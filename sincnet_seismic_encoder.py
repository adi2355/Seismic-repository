import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SincConv1d_SeismicAdapted(nn.Module):
    """
    SincNet Convolution Layer adapted for seismic data processing.
    
    Key adaptations for seismic domain:
    - Frequency range: 5-100 Hz (vs 85-3400 Hz in speech)
    - Filters: 40 (vs 64-80 in speech) 
    - Kernel size: 251 samples (maintained from speech for low-freq capture)
    - Mel-scale initialization adapted for seismic frequency range
    
    References:
    - Original SincNet: Ravanelli & Bengio (2018)
    - Seismic adaptations based on domain-specific research
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 sample_rate=1000, min_low_hz=5, min_band_hz=5, window_func='hamming'):
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
        
        # Initialize using adapted Mel-scale for seismic frequencies (5-100 Hz)
        self._initialize_seismic_mel_scale()
        
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
    
    def _initialize_seismic_mel_scale(self):
        """Initialize frequencies using Mel-scale adapted for seismic range (5-100 Hz)"""
        
        def hz_to_mel(hz):
            """Convert Hz to Mel scale"""
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            """Convert Mel scale to Hz"""
            return 700 * (10**(mel / 2595) - 1)
        
        # Seismic frequency range: 5-100 Hz
        low_freq_hz = self.min_low_hz
        high_freq_hz = min(100, self.sample_rate / 2 - 1)  # Cap at Nyquist
        
        # Convert to Mel scale
        low_freq_mel = hz_to_mel(low_freq_hz)
        high_freq_mel = hz_to_mel(high_freq_hz)
        
        # Create mel-spaced frequency points
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.out_channels + 1)
        hz_points = [mel_to_hz(mel) for mel in mel_points]
        
        # Initialize learnable parameters
        low_hz_init = []
        band_hz_init = []
        
        for i in range(self.out_channels):
            f_low = hz_points[i]
            f_high = hz_points[i + 1]
            
            # Parameterize relative to constraints for robust learning
            low_hz_init.append(f_low - self.min_low_hz)
            band_hz_init.append(f_high - f_low - self.min_band_hz)
        
        # Initialize parameters
        with torch.no_grad():
            self.low_hz_param.data = torch.tensor(low_hz_init, dtype=torch.float32).view(-1, 1)
            self.band_hz_param.data = torch.tensor(band_hz_init, dtype=torch.float32).view(-1, 1)
    
    def _get_time_vector(self):
        """Compute time vector for sinc function"""
        # Fix: Center the time vector around 0 with exactly kernel_size elements
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
        
        # Generate bandpass sinc filters
        filters = self._generate_sinc_filters(f_low, f_high)
        
        # Apply convolution
        output = F.conv1d(x, filters, stride=self.stride, padding=self.padding)
        
        return output
    
    def _generate_sinc_filters(self, f_low, f_high):
        """Generate bandpass sinc filters"""
        
        # Convert frequencies to normalized form (cycles per sample)
        f_low_norm = f_low / self.sample_rate  # Shape: (out_channels, 1)
        f_high_norm = f_high / self.sample_rate  # Shape: (out_channels, 1)
        
        # Compute sinc functions
        # sinc(2πft) where t is time vector
        # Need to broadcast: (out_channels, 1) * (1, 1, kernel_size) → (out_channels, 1, kernel_size)
        
        # Low-pass filter (high cutoff frequency)
        sinc_high = torch.sinc(2 * f_high_norm.unsqueeze(-1) * self.t_vect.squeeze(0))  # (out_channels, 1, kernel_size)
        
        # Low-pass filter (low cutoff frequency)  
        sinc_low = torch.sinc(2 * f_low_norm.unsqueeze(-1) * self.t_vect.squeeze(0))   # (out_channels, 1, kernel_size)
        
        # Bandpass = high_cutoff_lowpass - low_cutoff_lowpass
        bandpass_filters = sinc_high - sinc_low  # (out_channels, 1, kernel_size)
        
        # Handle sinc(0) = 1 case for numerical stability
        # When t=0, sinc(0) should be 1, but torch.sinc handles this
        
        # Apply window function (e.g., Hamming) to reduce spectral leakage
        windowed_filters = bandpass_filters * self.window  # (out_channels, 1, kernel_size) * (1, 1, kernel_size)
        
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
                 sample_rate=1000, min_low_hz=5, min_band_hz=5,
                 # 2D CNN aggregation parameters  
                 cnn_channels_list=[64, 128, 256], cnn_kernel_size=3,
                 # Output embedding dimension
                 embedding_dim=256):
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
        sample_rate=1000,
        embedding_dim=256
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