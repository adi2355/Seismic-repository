import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SincConv1d_SeismicAdapted(nn.Module):
    """
    SincNet Convolution Layer adapted for seismic data processing.
    
    Key adaptations for seismic domain:
    - Frequency range: Adaptively set based on sample_rate and kernel_size
    - Sample rate: Configurable, with 10001 Hz for Speed & Structure dataset
    - Filters: 40 (optimized for seismic multi-shot processing)
    - Kernel size: 251 samples (considerations for frequency resolution)
    - Linear frequency initialization (vs. Mel-scale for speech)
    
    CRITICAL FIX: Proper sinc filter implementation using sin(2πft)/(2πft)
    instead of torch.sinc() which computes sin(πx)/(πx).
    
    References:
    - Original SincNet: Ravanelli & Bengio (2018)
    - Seismic adaptations based on domain-specific research
    """
    
    def __init__(self, out_channels, kernel_size, sample_rate, in_channels=1, stride=10,  # CRITICAL FIX: stride=10
                 padding=None, dilation=1, bias=False, groups=1, min_low_hz=80, min_band_hz=10,
                 window_func='hamming', trainable_window=False, sample_normalization=True):
        super().__init__()
        
        if in_channels != 1:
            raise ValueError("SincConv1d only supports in_channels=1")
        
        # Set padding for 'same' output if not specified
        if padding is None:
            padding = kernel_size // 2  # 'same' padding for odd kernel_size
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride  # Now 10 instead of 50 - preserves more temporal resolution
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.window_func = window_func
        self.trainable_window = trainable_window
        self.sample_normalization = sample_normalization
        
        # Validate parameters
        if kernel_size % 2 == 0:
            raise ValueError("SincConv1d kernel_size must be odd")
        
        # Initialize learnable frequency parameters
        self._initialize_seismic_linear_scale()
        
        # Create window
        self._create_window()
        
        print(f"🔧 SincConv1d_SeismicAdapted initialized:")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Kernel size: {kernel_size} samples ({kernel_size/sample_rate*1000:.1f} ms)")
        print(f"   Padding: {padding} ('same' padding for consistent output size)")
        print(f"   Stride: {stride} (FIXED - was 50, now {stride} to prevent aliasing)")
        print(f"   Output sample rate: ~{sample_rate/stride:.0f} Hz")
        print(f"   Effective Nyquist: ~{sample_rate/stride/2:.0f} Hz")
        print(f"   Frequency range: {min_low_hz}-{self.max_learnable_hz:.0f} Hz")
        print(f"   Window: {window_func}")

    def _initialize_seismic_linear_scale(self):
        """Initialize frequency parameters with seismic-appropriate scaling."""
        # Calculate frequency limits based on kernel size and sample rate
        nyquist_freq = self.sample_rate / 2.0
        
        # Minimum resolvable frequency (conservative estimate)
        min_representable_hz = self.sample_rate / self.kernel_size
        
        # Effective max learnable frequency (conservative)
        # Reduced slightly to ensure good filter shape
        self.max_learnable_hz = min(nyquist_freq * 0.4, 300.0)  # Slightly higher ceiling
        
        # Ensure min_low_hz is reasonable
        effective_min_low = max(self.min_low_hz, min_representable_hz)
        
        print(f"📊 SincNet frequency initialization:")
        print(f"   Min representable: {min_representable_hz:.1f} Hz")
        print(f"   Effective min_low: {effective_min_low:.1f} Hz") 
        print(f"   Max learnable: {self.max_learnable_hz:.1f} Hz")
        print(f"   Frequency range: {self.max_learnable_hz - effective_min_low:.1f} Hz")
        
        # Linear spacing of center frequencies
        low_hz_init = np.linspace(
            effective_min_low, 
            self.max_learnable_hz - 2*self.min_band_hz,
            self.out_channels
        )
        
        # Band initialization - wider bands for better initial coverage
        band_hz_init = np.full(self.out_channels, self.min_band_hz * 2.0)
        
        # Ensure no filter exceeds max frequency
        for i in range(self.out_channels):
            if low_hz_init[i] + band_hz_init[i] > self.max_learnable_hz:
                band_hz_init[i] = self.max_learnable_hz - low_hz_init[i]
        
        # Convert to normalized frequencies [0, 1] where 1 = sample_rate/2
        self.f_low = nn.Parameter(torch.tensor(low_hz_init / nyquist_freq, dtype=torch.float32))
        self.band_hz = nn.Parameter(torch.tensor(band_hz_init / nyquist_freq, dtype=torch.float32))
        
        print(f"   Initialized {self.out_channels} filters")
        print(f"   Low freq range: {low_hz_init.min():.1f}-{low_hz_init.max():.1f} Hz")
        print(f"   Band range: {band_hz_init.min():.1f}-{band_hz_init.max():.1f} Hz")

    def _create_window(self):
        """Create windowing function for filter design."""
        n = self.kernel_size
        if self.window_func == 'hamming':
            window = torch.hamming_window(n, periodic=False)
        elif self.window_func == 'blackman':  # Better side-lobe suppression option
            window = torch.blackman_window(n, periodic=False)
        elif self.window_func == 'hann':
            window = torch.hann_window(n, periodic=False)
        else:
            window = torch.ones(n)
        
        if self.trainable_window:
            self.window = nn.Parameter(window)
        else:
            self.register_buffer('window', window)

    def _generate_sinc_filters(self):
        """Generate sinc bandpass filters based on learned parameters."""
        device = self.f_low.device
        dtype = self.f_low.dtype
        
        # Ensure valid frequency parameters
        f_low_clamped = torch.clamp(self.f_low, 0.0, 0.99)
        band_hz_clamped = torch.clamp(self.band_hz, 0.01, 0.99)
        
        # High frequency is low + bandwidth
        f_high = f_low_clamped + band_hz_clamped
        f_high_clamped = torch.clamp(f_high, f_low_clamped + 0.01, 0.99)
        
        # Time axis centered at zero
        n = self.kernel_size
        t = torch.arange(-(n//2), n//2 + 1, dtype=dtype, device=device).unsqueeze(0)
        
        # Convert normalized frequencies to actual frequencies
        f_low_hz = f_low_clamped * (self.sample_rate / 2.0)
        f_high_hz = f_high_clamped * (self.sample_rate / 2.0)
        
        # Generate sinc filters: sinc(2πf_high*t) - sinc(2πf_low*t)
        # Handle t=0 case separately to avoid division by zero
        filters = torch.zeros(self.out_channels, n, dtype=dtype, device=device)
        
        # Non-zero time points
        t_nonzero = t[:, t[0] != 0]
        
        if t_nonzero.numel() > 0:
            # High-frequency sinc
            sinc_high = torch.sin(2 * np.pi * f_high_hz.unsqueeze(1) * t_nonzero) / (
                np.pi * t_nonzero
            )
            
            # Low-frequency sinc  
            sinc_low = torch.sin(2 * np.pi * f_low_hz.unsqueeze(1) * t_nonzero) / (
                np.pi * t_nonzero
            )
            
            # Bandpass = high_sinc - low_sinc
            filters[:, t[0] != 0] = sinc_high - sinc_low
        
        # Handle t=0 case (limit as t->0)
        zero_idx = n // 2
        filters[:, zero_idx] = 2 * (f_high_hz - f_low_hz) / self.sample_rate
        
        # Apply window
        filters = filters * self.window.unsqueeze(0)
        
        # Normalize filters
        if self.sample_normalization:
            # L2 normalization
            filters = F.normalize(filters, p=2, dim=1)
        
        return filters.unsqueeze(1)  # Add input channel dimension

    def forward(self, x):
        """Forward pass through SincNet layer."""
        filters = self._generate_sinc_filters()
        
        return F.conv1d(
            x, filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class TemporalAntiAliasingDownsample(nn.Module):
    """
    Advanced temporal anti-aliasing module focusing on the time dimension.
    Implements 1D temporal filtering before downsampling as suggested in the analysis.
    """
    def __init__(self, channels, downsample_factor=2, filter_type='gaussian', filter_size=5):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.filter_type = filter_type
        
        if filter_type == 'gaussian':
            # Create 1D Gaussian filter for temporal anti-aliasing
            self.temporal_blur = nn.Conv1d(
                channels, channels,
                kernel_size=filter_size,
                stride=1,
                padding=filter_size//2,
                groups=channels,
                bias=False
            )
            
            # Initialize with Gaussian weights
            with torch.no_grad():
                sigma = 0.8  # Gaussian standard deviation
                x = torch.arange(filter_size, dtype=torch.float32) - filter_size // 2
                gaussian_kernel = torch.exp(-(x**2) / (2 * sigma**2))
                gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                
                for i in range(channels):
                    self.temporal_blur.weight[i, 0] = gaussian_kernel
                    
        elif filter_type == 'learnable':
            # Learnable 1D temporal filter
            self.temporal_blur = nn.Conv1d(
                channels, channels,
                kernel_size=filter_size,
                stride=1,
                padding=filter_size//2,
                groups=channels,
                bias=False
            )
            # Will be learned during training
            
        else:  # 'box' filter (fallback)
            self.temporal_blur = nn.Conv1d(
                channels, channels,
                kernel_size=filter_size,
                stride=1,
                padding=filter_size//2,
                groups=channels,
                bias=False
            )
            
            with torch.no_grad():
                box_kernel = torch.ones(filter_size) / filter_size
                for i in range(channels):
                    self.temporal_blur.weight[i, 0] = box_kernel
        
        # Freeze filter weights for non-learnable types
        if filter_type != 'learnable':
            self.temporal_blur.weight.requires_grad = False
            
    def forward(self, x):
        # x: (B, C, T, R) - Apply temporal anti-aliasing along T dimension
        B, C, T, R = x.shape
        
        # Reshape to apply 1D conv along temporal dimension
        # (B, C, T, R) → (B*R, C, T)
        x_temp = x.permute(0, 3, 1, 2).contiguous().view(B*R, C, T)
        
        # Apply temporal anti-aliasing filter
        x_filtered = self.temporal_blur(x_temp)
        
        # Reshape back: (B*R, C, T) → (B, C, T, R)
        x_filtered = x_filtered.view(B, R, C, T).permute(0, 2, 3, 1)
        
        # Downsample temporally only
        x_downsampled = x_filtered[:, :, ::self.downsample_factor, :]
        
        return x_downsampled


class AntiAliasingDownsample(nn.Module):
    """
    Anti-aliasing downsampling module to replace naive strided convolutions.
    Implements proper low-pass filtering before downsampling.
    """
    def __init__(self, channels, downsample_factor=2, filter_size=5):
        super().__init__()
        self.downsample_factor = downsample_factor
        
        # Create anti-aliasing filter (simple Gaussian-based)
        self.blur = nn.Conv2d(
            channels, channels, 
            kernel_size=filter_size,
            stride=1,
            padding=filter_size//2,
            groups=channels,
            bias=False
        )
        
        # Initialize with Gaussian-like weights
        with torch.no_grad():
            # Simple box filter approximation for anti-aliasing
            weight = torch.ones(filter_size, filter_size) / (filter_size * filter_size)
            for i in range(channels):
                self.blur.weight[i, 0] = weight
        
        # Freeze the anti-aliasing filter
        self.blur.weight.requires_grad = False
        
    def forward(self, x):
        # Apply anti-aliasing filter
        x = self.blur(x)
        
        # Downsample
        return x[:, :, ::self.downsample_factor, ::self.downsample_factor]


class PerShotTemporalEncoder(nn.Module):
    """
    FIXED VERSION: Implements hierarchical downsampling with anti-aliasing
    to address temporal resolution bottleneck identified in analysis.
    """
    def __init__(self, sample_rate=10001, num_receivers=31, time_samples=10001,
                 sinc_out_channels=40, sinc_kernel_size=251, sinc_stride=10,  # FIXED stride
                 sinc_min_low_hz=80, sinc_min_band_hz=10,
                 embedding_dim=128, window_func='hamming'):
        super().__init__()
        
        print(f"🔧 PerShotTemporalEncoder FIXED VERSION:")
        print(f"   Input: ({time_samples}, {num_receivers})")
        print(f"   SincNet stride: {sinc_stride} (FIXED - prevents aliasing)")
        
        # SincNet layer with FIXED stride
        self.sinc_layer = SincConv1d_SeismicAdapted(
            out_channels=sinc_out_channels,
            kernel_size=sinc_kernel_size,
            sample_rate=sample_rate,
            stride=sinc_stride,  # Now 10 instead of 50
            min_low_hz=sinc_min_low_hz,
            min_band_hz=sinc_min_band_hz,
            window_func=window_func
        )
        
        # Calculate dimensions after SincNet
        # With 'same' padding, output length = ceil(input_length / stride)
        sinc_output_length = (time_samples + sinc_stride - 1) // sinc_stride
        print(f"   After SincNet: ({sinc_output_length}, {num_receivers}) [with 'same' padding]")
        
        # HIERARCHICAL DOWNSAMPLING with anti-aliasing
        # Goal: Reduce temporal dimension gradually while preserving information
        
        # Stage 1: Initial 2D convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(sinc_out_channels, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(8, 64),
            nn.ELU(),
            nn.Dropout2d(0.1)
        )
        
        # Stage 2: First downsampling (temporal factor 2) - IMPROVED ANTI-ALIASING
        self.downsample1 = TemporalAntiAliasingDownsample(64, downsample_factor=2, filter_type='gaussian')
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(16, 128),
            nn.ELU(),
            nn.Dropout2d(0.1)
        )
        
        # Stage 3: Second downsampling (temporal factor 2) - IMPROVED ANTI-ALIASING  
        self.downsample2 = TemporalAntiAliasingDownsample(128, downsample_factor=2, filter_type='gaussian')
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(32, 256),
            nn.ELU(),
            nn.Dropout2d(0.1)
        )
        
        # Stage 4: Adaptive pooling to final manageable size
        target_temporal = max(8, sinc_output_length // 16)  # Reasonable final temporal dimension
        target_spatial = max(4, num_receivers // 8)         # Reasonable final spatial dimension
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_temporal, target_spatial))
        
        # Final embedding layer
        final_features = 256 * target_temporal * target_spatial
        self.embedding_projection = nn.Sequential(
            nn.Linear(final_features, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        print(f"   Hierarchical downsampling: {sinc_output_length} → {target_temporal}")
        print(f"   IMPROVED: 1D Gaussian temporal anti-aliasing (preserves high-freq SincNet content)")
        print(f"   Final embedding: {embedding_dim}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, shot_data):
        # shot_data: (batch_size, time_samples, num_receivers)
        batch_size = shot_data.shape[0]
        
        # Apply SincNet along time dimension
        # Reshape for 1D convolution: (batch_size * num_receivers, 1, time_samples)
        num_receivers = shot_data.shape[2]
        shot_reshaped = shot_data.transpose(1, 2).contiguous()  # (batch, receivers, time)
        shot_reshaped = shot_reshaped.view(-1, 1, shot_data.shape[1])  # (batch*receivers, 1, time)
        
        # SincNet processing
        sinc_output = self.sinc_layer(shot_reshaped)  # (batch*receivers, sinc_channels, time_reduced)
        
        # Reshape back to 2D: (batch_size, sinc_channels, time_reduced, num_receivers)
        sinc_channels, time_reduced = sinc_output.shape[1], sinc_output.shape[2]
        sinc_output = sinc_output.view(batch_size, num_receivers, sinc_channels, time_reduced)
        sinc_output = sinc_output.permute(0, 2, 3, 1)  # (batch, sinc_channels, time, receivers)
        
        # Hierarchical 2D CNN processing with anti-aliasing
        x = self.conv1(sinc_output)
        
        x = self.downsample1(x)  # Anti-aliased downsampling
        x = self.conv2(x)
        
        x = self.downsample2(x)  # Anti-aliased downsampling  
        x = self.conv3(x)
        
        # Adaptive pooling to target size
        x = self.adaptive_pool(x)
        
        # Global feature extraction
        x = x.view(batch_size, -1)
        
        # Final embedding
        embedding = self.embedding_projection(x)
        
        return embedding


# Initialize weights function
def initialize_seismic_weights(module):
    """Initialize weights for seismic processing modules."""
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


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
        sinc_stride=10,
        sample_rate=10001,  # Updated sample rate
        min_low_hz=80,      # Updated minimum frequency
        min_band_hz=10,     # Updated minimum bandwidth
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
        f_low = sinc_layer.min_low_hz + torch.abs(sinc_layer.f_low)
        f_high = f_low + sinc_layer.min_band_hz + torch.abs(sinc_layer.band_hz)
        f_high = torch.clamp(f_high, max=sinc_layer.sample_rate / 2 - 1)
        
        # Calculate the minimum representable frequency
        min_representable_hz = 2 * sinc_layer.sample_rate / sinc_layer.kernel_size
        
        print(f"✅ SincNet frequency information:")
        print(f"   Sample rate: {sinc_layer.sample_rate} Hz")
        print(f"   Kernel size: {sinc_layer.kernel_size} samples")
        print(f"   Minimum representable frequency: {min_representable_hz:.1f} Hz")
        print(f"   Configured min_low_hz: {sinc_layer.min_low_hz} Hz")
        print(f"   Nyquist frequency: {sinc_layer.sample_rate/2:.1f} Hz")
        print(f"   Learned f_low range: [{f_low.min():.1f}, {f_low.max():.1f}] Hz")
        print(f"   Learned f_high range: [{f_high.min():.1f}, {f_high.max():.1f}] Hz")
    
    print("🎉 SincNet Seismic Encoder test completed successfully!")
    return True


if __name__ == "__main__":
    test_sincnet_encoder() 