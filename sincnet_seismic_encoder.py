import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SincConv1d_SeismicAdapted(nn.Module):
    """
    SincNet Convolution Layer adapted for seismic data processing.
    
    Key adaptations for seismic domain:
    - Frequency range: 2-150 Hz (seismic reflection band)
    - Sample rate: 500 Hz (2ms interval, based on Speed & Structure dataset research)
    - Filters: 40 (optimized for seismic multi-shot processing)
    - Kernel size: 251 samples (sufficient for low-frequency capture at 500Hz)
    - Linear frequency initialization (vs. Mel-scale for speech)
    
    CRITICAL FIX: Proper sinc filter implementation using sin(2πft)/(2πft)
    instead of torch.sinc() which computes sin(πx)/(πx).
    
    References:
    - Original SincNet: Ravanelli & Bengio (2018)
    - Seismic adaptations based on domain-specific research
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 sample_rate=500, min_low_hz=2, min_band_hz=3, window_func='hamming'):
        super().__init__()
        
        if in_channels != 1:
            raise ValueError("SincConv1d_SeismicAdapted only supports in_channels=1 for trace-wise processing")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.window_func = window_func
        
        # Learnable parameters for frequency bounds
        # Using relative parameterization for robust constraints
        self.low_hz_param = nn.Parameter(torch.Tensor(out_channels, 1))
        self.band_hz_param = nn.Parameter(torch.Tensor(out_channels, 1))
        
        # Initialize using linear spacing for seismic frequencies (2-150 Hz)
        self._initialize_seismic_linear_scale()
        
        # Pre-compute time vector for sinc function
        self.register_buffer('t_vect', self._get_time_vector())
        
        # Pre-compute window function
        if window_func == 'hamming':
            window = torch.hamming_window(kernel_size)
        elif window_func == 'hann':
            window = torch.hann_window(kernel_size)
        else:
            window = torch.ones(kernel_size)  # Rectangular window
        
        self.register_buffer('window', window.view(1, 1, -1))
    
    def _initialize_seismic_linear_scale(self):
        """Initialize frequencies using linear spacing for seismic range (2-150 Hz)"""
        
        # Seismic frequency range: 2-150 Hz (linear spacing more appropriate than Mel)
        low_freq_hz = self.min_low_hz
        high_freq_hz = min(150, self.sample_rate / 2 - self.min_band_hz - 1)  # Cap at reasonable seismic max
        
        # Create linearly-spaced frequency points
        freq_points = np.linspace(low_freq_hz, high_freq_hz, self.out_channels + 1)
        
        # Initialize learnable parameters
        low_hz_init = []
        band_hz_init = []
        
        for i in range(self.out_channels):
            f_low = freq_points[i]
            f_high = freq_points[i + 1]
            
            # Parameterize relative to constraints for robust learning
            low_hz_init.append(f_low - self.min_low_hz)
            band_hz_init.append(f_high - f_low - self.min_band_hz)
        
        # Initialize parameters
        with torch.no_grad():
            self.low_hz_param.data = torch.tensor(low_hz_init, dtype=torch.float32).view(-1, 1)
            self.band_hz_param.data = torch.tensor(band_hz_init, dtype=torch.float32).view(-1, 1)
    
    def _get_time_vector(self):
        """Compute time vector for sinc function"""
        # Center the time vector around 0 with exactly kernel_size elements
        n = (self.kernel_size - 1) // 2
        # Create symmetric time indices: [-n, -n+1, ..., -1, 0, 1, ..., n-1, n]
        time_indices = torch.arange(-n, n + 1, dtype=torch.float32)
        t_vect = time_indices / self.sample_rate
        return t_vect.view(1, 1, -1)
    
    def forward(self, x):
        """
        Forward pass with differentiable sinc filter generation
        
        Args:
            x: Input tensor (batch_size, 1, time_samples)
            
        Returns:
            Filtered output (batch_size, out_channels, time_samples_out)
        """
        
        # Compute actual frequencies with constraints
        # f_low = min_low_hz + |low_hz_param|
        f_low = self.min_low_hz + torch.abs(self.low_hz_param)
        
        # f_high = f_low + min_band_hz + |band_hz_param|, capped at Nyquist
        f_high = f_low + self.min_band_hz + torch.abs(self.band_hz_param)
        f_high = torch.clamp(f_high, max=self.sample_rate / 2 - 1)
        
        # Ensure f_low doesn't exceed f_high
        f_low = torch.clamp(f_low, max=f_high - self.min_band_hz)
        
        # Generate bandpass sinc filters
        filters = self._generate_sinc_filters(f_low, f_high)
        
        # Apply convolution
        output = F.conv1d(x, filters, stride=self.stride, padding=self.padding)
        
        return output
    
    def _generate_sinc_filters(self, f_low, f_high):
        """
        Generate bandpass sinc filters using correct sinc implementation.
        
        CRITICAL FIX: Direct sin(2πft)/(2πft) calculation with proper normalization
        """
        
        # Convert frequencies to normalized frequencies (relative to sample rate)
        f_low_norm = f_low / self.sample_rate   # Shape: (out_channels, 1)
        f_high_norm = f_high / self.sample_rate  # Shape: (out_channels, 1)
        
        # Time vector: (1, 1, kernel_size)
        t_actual = self.t_vect  # Already in seconds
        
        # Compute sinc functions using direct sin(2πft)/(2πft) calculation
        # Shape broadcasting: (out_channels, 1, 1) * (1, 1, kernel_size) → (out_channels, 1, kernel_size)
        
        # For high cutoff frequency
        arg_high = 2 * math.pi * f_high.unsqueeze(-1) * t_actual
        # For low cutoff frequency  
        arg_low = 2 * math.pi * f_low.unsqueeze(-1) * t_actual
        
        # Handle sinc(0) = 1 case (when t=0, arg=0)
        # Create safe denominators to avoid division by zero
        safe_arg_high = torch.where(torch.abs(arg_high) < 1e-8, torch.ones_like(arg_high), arg_high)
        safe_arg_low = torch.where(torch.abs(arg_low) < 1e-8, torch.ones_like(arg_low), arg_low)
        
        # Compute sinc values: sinc(arg) = sin(arg)/arg
        sinc_val_high = torch.sin(arg_high) / safe_arg_high
        sinc_val_low = torch.sin(arg_low) / safe_arg_low
        
        # Low-pass filters: h(t) = 2*f_norm * sinc(2πf_norm*t)
        low_pass_high = 2 * f_high_norm.unsqueeze(-1) * sinc_val_high
        low_pass_low = 2 * f_low_norm.unsqueeze(-1) * sinc_val_low
        
        # Bandpass = high_cutoff_lowpass - low_cutoff_lowpass
        bandpass_filters_raw = low_pass_high - low_pass_low  # (out_channels, 1, kernel_size)
        
        # Explicit center tap handling for numerical stability
        center_index = (self.kernel_size - 1) // 2
        center_tap_values = 2 * (f_high_norm - f_low_norm)  # At t=0: 2*(f_high - f_low)
        bandpass_filters_raw[:, 0, center_index] = center_tap_values.squeeze(-1)
        
        # Apply window function to reduce spectral leakage
        windowed_filters = bandpass_filters_raw * self.window  # (out_channels, 1, kernel_size) * (1, 1, kernel_size)
        
        # Normalize filters (unit energy)
        filter_norms = torch.sqrt(torch.sum(windowed_filters**2, dim=2, keepdim=True))
        normalized_filters = windowed_filters / (filter_norms + 1e-8)
        
        return normalized_filters


class PerShotTemporalEncoder(nn.Module):
    """
    Per-shot temporal encoder using SincNet + 2D CNN aggregation.
    
    Architecture:
    1. Trace-wise SincNet processing (31 traces → 31 × 40 features)
    2. 2D CNN aggregation across (sinc_features, time, receivers)
    3. Global average pooling → linear projection to embedding
    
    Input: (B, 10001, 31) - Single shot gather
    Output: (B, embedding_dim) - Shot-level embedding
    """
    
    def __init__(self, num_receivers=31,
                 # SincNet parameters
                 sinc_out_channels=40, sinc_kernel_size=251, sinc_stride=50, 
                 sample_rate=500, min_low_hz=2, min_band_hz=3,
                 # 2D CNN aggregation parameters  
                 cnn_channels_list=[64, 128, 256], cnn_kernel_size=3,
                 # Output embedding dimension
                 embedding_dim=128):
        super().__init__()
        
        self.num_receivers = num_receivers
        self.sinc_out_channels = sinc_out_channels
        self.sinc_stride = sinc_stride
        self.sinc_kernel_size = sinc_kernel_size
        
        # SincNet layer for trace-wise processing
        self.sinc_layer = SincConv1d_SeismicAdapted(
            in_channels=1,
            out_channels=sinc_out_channels,
            kernel_size=sinc_kernel_size,
            stride=sinc_stride,
            padding=sinc_kernel_size // 2,  # 'same' padding
            sample_rate=sample_rate,
            min_low_hz=min_low_hz,
            min_band_hz=min_band_hz,
            window_func='hamming'
        )
        
        # Calculate output time dimension after SincNet
        # For 'same' padding: output_length = (input_length + stride - 1) // stride
        # With 10001 input samples, stride=50: (10001 + 50 - 1) // 50 = 10050 // 50 = 201
        self.time_reduced = (10001 + sinc_stride - 1) // sinc_stride
        
        # Layer normalization after SincNet (per trace)
        self.sinc_norm = nn.LayerNorm([sinc_out_channels, self.time_reduced])
        self.sinc_activation = nn.LeakyReLU(0.2, inplace=True)
        
        # 2D CNN stack for spatial-temporal aggregation
        # Input shape to CNN: (B, sinc_out_channels, time_reduced, num_receivers)
        cnn_layers = []
        current_channels = sinc_out_channels
        
        for i, out_channels in enumerate(cnn_channels_list):
            # Convolution
            cnn_layers.append(
                nn.Conv2d(current_channels, out_channels, 
                         kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2)
            )
            
            # Group normalization (more stable than BatchNorm for varying batch sizes)
            num_groups = min(8, out_channels // 2) if out_channels >= 16 else out_channels
            cnn_layers.append(nn.GroupNorm(num_groups, out_channels))
            
            # Activation
            cnn_layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Pooling (reduce spatial dimensions progressively)
            if i < len(cnn_channels_list) - 1:  # Don't pool after last layer
                # Pool more aggressively in time dimension, preserve receivers
                pool_kernel = (2, 1) if i < 2 else (2, 2)
                cnn_layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel))
            
            current_channels = out_channels
        
        self.cnn_aggregator = nn.Sequential(*cnn_layers)
        
        # Global average pooling to fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to embedding dimension
        self.projection = nn.Linear(cnn_channels_list[-1], embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize CNN and projection weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Smaller variance for final projection layer
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_shot_gather):
        """
        Forward pass for single shot gather
        
        Args:
            x_shot_gather: (B, 10001, 31) - Batch of shot gathers
            
        Returns:
            embedding: (B, embedding_dim) - Shot-level embeddings
        """
        B, time_samples, num_receivers = x_shot_gather.shape
        
        if num_receivers != self.num_receivers:
            raise ValueError(f"Expected {self.num_receivers} receivers, got {num_receivers}")
        if time_samples != 10001:
            raise ValueError(f"Expected 10001 time samples, got {time_samples}")
        
        # Reshape for trace-wise SincNet processing
        # (B, 10001, 31) → (B, 31, 10001) → (B*31, 1, 10001)
        x_traces = x_shot_gather.permute(0, 2, 1).contiguous()  # (B, 31, 10001)
        x_traces = x_traces.view(B * num_receivers, 1, time_samples)  # (B*31, 1, 10001)
        
        # Apply SincNet to each trace
        sinc_features = self.sinc_layer(x_traces)  # (B*31, sinc_out_channels, time_reduced)
        
        # Normalize and activate SincNet features
        sinc_features = self.sinc_norm(sinc_features)
        sinc_features = self.sinc_activation(sinc_features)
        
        # Reshape back to shot structure
        # (B*31, sinc_out_channels, time_reduced) → (B, 31, sinc_out_channels, time_reduced)
        sinc_features = sinc_features.view(B, num_receivers, self.sinc_out_channels, self.time_reduced)
        
        # Permute for 2D CNN: (B, sinc_out_channels, time_reduced, num_receivers)
        cnn_input = sinc_features.permute(0, 2, 3, 1)  # (B, C, H, W) format
        
        # Apply 2D CNN aggregation
        cnn_features = self.cnn_aggregator(cnn_input)  # (B, final_channels, H_reduced, W_reduced)
        
        # Global average pooling
        pooled_features = self.global_pool(cnn_features)  # (B, final_channels, 1, 1)
        
        # Flatten and project to final embedding
        flattened = pooled_features.view(B, -1)  # (B, final_channels)
        embedding = self.projection(flattened)  # (B, embedding_dim)
        
        return embedding


# Test function for development
def test_sincnet_encoder():
    """Test the SincNet encoder with dummy data"""
    
    print("🧪 Testing SincNet Seismic Encoder...")
    
    # Create test data
    batch_size = 2
    dummy_shot = torch.randn(batch_size, 10001, 31)
    
    print(f"Input shape: {dummy_shot.shape}")
    
    # Create encoder
    encoder = PerShotTemporalEncoder(
        num_receivers=31,
        sinc_out_channels=40,
        sinc_kernel_size=251,
        sinc_stride=50,
        sample_rate=500,
        embedding_dim=128
    )
    
    print(f"Created encoder with {sum(p.numel() for p in encoder.parameters())} parameters")
    
    # Forward pass
    try:
        with torch.no_grad():
            embedding = encoder(dummy_shot)
            print(f"✅ Output shape: {embedding.shape}")
            print(f"✅ Output range: [{embedding.min():.3f}, {embedding.max():.3f}]")
            
            # Check for NaN/Inf
            if torch.isnan(embedding).any():
                print("❌ NaN detected in output!")
            elif torch.isinf(embedding).any():
                print("❌ Inf detected in output!")
            else:
                print("✅ Output is numerically stable")
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test SincNet frequency parameters
    sinc_layer = encoder.sinc_layer
    with torch.no_grad():
        f_low = sinc_layer.min_low_hz + torch.abs(sinc_layer.low_hz_param)
        f_high = f_low + sinc_layer.min_band_hz + torch.abs(sinc_layer.band_hz_param)
        f_high = torch.clamp(f_high, max=sinc_layer.sample_rate / 2 - 1)
        
        print(f"✅ SincNet frequency ranges:")
        print(f"   f_low: [{f_low.min():.1f}, {f_low.max():.1f}] Hz")
        print(f"   f_high: [{f_high.min():.1f}, {f_high.max():.1f}] Hz")
        print(f"   Valid seismic range: {sinc_layer.min_low_hz}-{sinc_layer.sample_rate/2:.0f} Hz")
    
    print("🎉 SincNet Seismic Encoder test completed successfully!")
    return True


if __name__ == "__main__":
    test_sincnet_encoder() 