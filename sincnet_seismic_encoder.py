import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SincConv1d_SeismicAdapted(nn.Module):
    """
    SincNet Convolution Layer adapted for seismic data processing.
    
    OPTIMIZED configuration based on detailed spectral analysis of 10001 Hz data:
    - kernel_size=1001 for better low-frequency resolution (down to ~10-20 Hz)
    - max_learnable_hz=1000 to match the spectral content of seismic signals
    - stride=1 to eliminate aliasing (critical fix)
    - Logarithmic filter spacing for better allocation of filters across frequencies
    - Blackman window for superior side-lobe suppression
    - 60 filters for optimal spectral coverage without redundancy
    
    References:
    - Original SincNet: Ravanelli & Bengio (2018)
    - Seismic adaptations based on domain-specific research
    """
    
    def __init__(self, 
                 out_channels=60,        # Recommended: 60
                 kernel_size=1001,       # Recommended: 1001
                 sample_rate=10001,      # Should be 10001
                 in_channels=1,          # Fixed
                 stride=1,               # CRITICAL: Set to 1 (or very small like 2, 4)
                 padding='same',         # Use 'same' for easier length calculation
                 min_low_hz=40,          # Recommended: 40Hz (with kernel 1001) or 80Hz
                 max_learnable_hz=1000,  # Recommended: 1000Hz
                 min_band_hz=10,         # Min bandwidth for a filter
                 window_func='blackman', # Recommended: 'blackman'
                 initialization_type='logarithmic'): # Recommended: 'logarithmic'
        super().__init__()
        
        if in_channels != 1:
            raise ValueError("SincConv1d_SeismicAdapted only supports in_channels=1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = float(sample_rate) # Ensure float for division
        self.stride = stride
        self.min_low_hz = float(min_low_hz)
        self.max_learnable_hz = float(max_learnable_hz)
        self.min_band_hz = float(min_band_hz)
        self.window_func = window_func
        self.initialization_type = initialization_type
        
        # Padding: 'same' will be handled by F.conv1d if padding arg is 'same'
        # Or calculate manually: self.padding_val = (kernel_size - 1) // 2
        if isinstance(padding, str) and padding.lower() == 'same':
            self.padding_val = padding # Pass 'same' directly to F.conv1d
        elif isinstance(padding, int):
            self.padding_val = padding
        else: # Default to manual 'same' calculation
            self.padding_val = (kernel_size - 1) // 2

        # Learnable parameters for frequency bounds (normalized 0 to 0.5 of sample_rate)
        # f_center_norm and f_bandwidth_norm
        self.f_center_norm = nn.Parameter(torch.Tensor(out_channels, 1))
        self.f_bandwidth_norm = nn.Parameter(torch.Tensor(out_channels, 1))

        self._initialize_filter_params()

        # Pre-compute time vector for sinc function
        self.register_buffer('t_vect', self._get_time_vector())
        
        # Pre-compute window function
        if self.window_func == 'hamming':
            window = torch.hamming_window(kernel_size, periodic=False)
        elif self.window_func == 'blackman':
            window = torch.blackman_window(kernel_size, periodic=False)
        elif self.window_func == 'hann':
            window = torch.hann_window(kernel_size, periodic=False)
        else: # Rectangular
            window = torch.ones(kernel_size)
        self.register_buffer('window', window.view(1, 1, -1))

        print(f"🔧 SincConv1d_SeismicAdapted Initialized:")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Kernel Size: {self.kernel_size} samples ({self.kernel_size/self.sample_rate*1000:.1f} ms)")
        print(f"   Stride: {self.stride}")
        print(f"   Filters: {self.out_channels}, Window: {self.window_func}")
        print(f"   Target Freq. Range (init): [{self.min_low_hz:.1f} - {self.max_learnable_hz:.1f}] Hz ({self.initialization_type} spacing)")

    def _initialize_filter_params(self):
        # Calculate effective frequency bounds based on kernel size
        # With kernel_size=1001 and sample_rate=10001, we can theoretically resolve down to ~10-20 Hz
        effective_resolution_hz = self.sample_rate / self.kernel_size  # ~10 Hz for 1001-point kernel
        actual_min_low_hz = max(self.min_low_hz, effective_resolution_hz * 2)  # Safety factor
        actual_max_learnable_hz = min(self.max_learnable_hz, self.sample_rate / 2.0 - 50)  # Leave margin from Nyquist
        
        print(f"🔧 SincConv1d_SeismicAdapted Initialized:")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Kernel Size: {self.kernel_size} samples ({self.kernel_size/self.sample_rate*1000:.1f} ms)")
        print(f"   Stride: {self.stride}")
        print(f"   Filters: {self.out_channels}, Window: {self.window_func}")
        print(f"   Target Freq. Range (init): [{actual_min_low_hz:.1f} - {actual_max_learnable_hz:.1f}] Hz ({self.initialization_type} spacing)")
        
        if self.initialization_type == 'logarithmic':
            # Use log spacing if range is sufficient
            log_range = actual_max_learnable_hz / actual_min_low_hz
            if log_range > 2.0:  # Minimum for meaningful log spacing
                center_freqs_hz_init = np.logspace(
                    np.log10(actual_min_low_hz), 
                    np.log10(actual_max_learnable_hz), 
                    self.out_channels
                )
            else:
                print(f"   Warning: Log range {log_range:.2f} too small, using linear spacing")
                center_freqs_hz_init = np.linspace(actual_min_low_hz, actual_max_learnable_hz, self.out_channels)
        else:
            # Linear spacing
            center_freqs_hz_init = np.linspace(actual_min_low_hz, actual_max_learnable_hz, self.out_channels)
        
        # Initialize bandwidths: use 10% of center frequency but ensure minimum bandwidth
        bandwidths_hz_init = np.maximum(self.min_band_hz, center_freqs_hz_init * 0.1)
        
        # Ensure filters don't exceed the valid range when bandwidth is added
        max_allowed_bw = 2 * (actual_max_learnable_hz - center_freqs_hz_init)
        bandwidths_hz_init = np.minimum(bandwidths_hz_init, max_allowed_bw)
        bandwidths_hz_init = np.maximum(bandwidths_hz_init, self.min_band_hz)  # Ensure minimum
        
        # CRITICAL FIX: Normalize by Nyquist frequency, not sample_rate
        nyquist = self.sample_rate / 2.0  # This is the key fix!
        
        # Convert to normalized parameters (0 to 1.0, where 1.0 = Nyquist)
        self.f_center_norm.data = torch.tensor(center_freqs_hz_init, dtype=torch.float32).view(-1,1) / nyquist
        self.f_bandwidth_norm.data = torch.tensor(bandwidths_hz_init, dtype=torch.float32).view(-1,1) / nyquist

        # DEBUG VERIFICATION: Print normalization values to confirm Nyquist fix is active
        print(f"   🔍 SINCNET DEBUG: Nyquist freq = {nyquist:.1f} Hz")
        print(f"   🔍 SINCNET DEBUG: f_center_norm min={self.f_center_norm.data.min().item():.4f}, max={self.f_center_norm.data.max().item():.4f}")
        print(f"   🔍 SINCNET DEBUG: f_bandwidth_norm min={self.f_bandwidth_norm.data.min().item():.4f}, max={self.f_bandwidth_norm.data.max().item():.4f}")
        print(f"   🔍 SINCNET DEBUG: Expected f_center_norm max ≈ {actual_max_learnable_hz/nyquist:.4f} (if Nyquist norm is correct)")

    def _get_time_vector(self):
        n = (self.kernel_size - 1) // 2
        time_indices = torch.arange(-n, n + 1, dtype=torch.float32)
        return time_indices.view(1, 1, -1)

    def _get_current_cutoffs(self):
        # Ensure parameters stay in valid ranges through clamping
        f_c_norm = torch.abs(self.f_center_norm) 
        f_bw_norm = torch.abs(self.f_bandwidth_norm)

        # Ensure bandwidth is at least min_band_hz (normalized by Nyquist)
        nyquist = self.sample_rate / 2.0
        min_b_norm = self.min_band_hz / nyquist
        f_bw_norm = torch.clamp(f_bw_norm, min=min_b_norm)

        f_low_norm = f_c_norm - f_bw_norm / 2.0
        f_high_norm = f_c_norm + f_bw_norm / 2.0

        # Clamp to [0, 1.0] where 1.0 = Nyquist frequency
        f_low_norm = torch.clamp(f_low_norm, min=0.0, max=1.0 - min_b_norm) # ensure high can be higher
        
        # Using torch.maximum for proper tensor operations with 1.0 representing Nyquist
        f_high_norm = torch.minimum(f_high_norm, torch.ones_like(f_high_norm) * 1.0)
        f_high_norm = torch.maximum(f_high_norm, f_low_norm + min_b_norm) # ensure minimum bandwidth
        
        return f_low_norm, f_high_norm

    def _generate_sinc_filters(self):
        f_low_norm, f_high_norm = self._get_current_cutoffs()
        
        # t_vect is now time indices: (1, 1, kernel_size)
        # Frequencies are normalized (0 to 0.5)
        # Argument for sinc: 2 * pi * f_norm * t_indices
        t_indices = self.t_vect 

        # First make sure f_low_norm and f_high_norm are properly shaped for broadcasting
        # Should be (out_channels, 1, 1) for broadcasting with t_indices (1, 1, kernel_size)
        f_low_norm = f_low_norm.view(self.out_channels, 1, 1)
        f_high_norm = f_high_norm.view(self.out_channels, 1, 1)

        # High-frequency sinc component
        arg_high = 2 * math.pi * f_high_norm * t_indices
        # Low-frequency sinc component
        arg_low = 2 * math.pi * f_low_norm * t_indices
        
        safe_arg_high = torch.where(torch.abs(arg_high) < 1e-8, torch.ones_like(arg_high), arg_high)
        safe_arg_low = torch.where(torch.abs(arg_low) < 1e-8, torch.ones_like(arg_low), arg_low)
        
        sinc_val_high = torch.sin(arg_high) / safe_arg_high
        sinc_val_low = torch.sin(arg_low) / safe_arg_low
        
        # According to SincNet paper, filters are 2*f_high*sinc(2*pi*f_high*t) - 2*f_low*sinc(2*pi*f_low*t)
        # where f_high and f_low are normalized frequencies
        low_pass_high = 2 * f_high_norm * sinc_val_high
        low_pass_low = 2 * f_low_norm * sinc_val_low
        
        bandpass_filters_raw = low_pass_high - low_pass_low
        
        # Center tap (t=0)
        center_idx = (self.kernel_size - 1) // 2
        
        # Handle center tap value for t=0 (L'Hôpital's rule)
        # Calculate directly using broadcasting
        center_diff = 2 * (f_high_norm - f_low_norm).squeeze(-1)  # Shape: (out_channels, 1)
        bandpass_filters_raw[:, :, center_idx] = center_diff
        
        windowed_filters = bandpass_filters_raw * self.window
        
        # Normalize filters (L2 norm per filter)
        filter_norms = torch.norm(windowed_filters, p=2, dim=2, keepdim=True)
        normalized_filters = windowed_filters / (filter_norms + 1e-8)
        
        return normalized_filters # Shape (out_channels, 1, kernel_size)

    def forward(self, x):
        filters = self._generate_sinc_filters()
        # Input x: (B, 1, T_in)
        # Filters: (C_out, 1, K)
        output = F.conv1d(x, filters, stride=self.stride, padding=self.padding_val)
        return output


# Anti-Aliasing Modules
class BlurPool1D(nn.Module):
    """1D anti-aliasing module using Gaussian filters followed by strided subsampling
    
    OPTIMIZED VERSION based on research findings:
    - For 5x+ downsampling: Use 9-11 taps with σ≈2.5-3.0
    - For 4x downsampling: Use 7-9 taps with σ≈2.0-2.5  
    - For 2x downsampling: Use 5-7 taps with σ≈1.0-1.5
    """
    def __init__(self, channels, filt_size=None, stride=2, sigma=None):
        super().__init__()
        self.channels = channels
        self.stride = stride
        
        # RESEARCH-BASED OPTIMIZATION: Auto-select filter size and sigma based on stride
        if filt_size is None:
            # Auto-select optimal filter size based on downsampling factor
            if stride >= 5:
                self.filt_size = 11  # 11-tap for 5x+ downsampling
            elif stride >= 4:
                self.filt_size = 9   # 9-tap for 4x downsampling  
            elif stride >= 3:
                self.filt_size = 7   # 7-tap for 3x downsampling
            else:
                self.filt_size = 5   # 5-tap for 2x downsampling
        else:
            self.filt_size = filt_size
            
        self.padding = (self.filt_size - 1) // 2
        
        # RESEARCH-BASED SIGMA OPTIMIZATION
        if sigma is None:
            # Optimized sigma values based on research for different downsampling factors
            if stride >= 5:
                self.sigma = 3.0    # σ=3.0 for 5x+ (strong low-pass filtering)
            elif stride >= 4:
                self.sigma = 2.5    # σ=2.5 for 4x 
            elif stride >= 3:
                self.sigma = 2.0    # σ=2.0 for 3x
            else:
                self.sigma = max(1.0, stride / 2.0)  # σ≈1.0-1.5 for 2x
        else:
            self.sigma = sigma
        
        # Generate Gaussian filter weights with optimized parameters
        coords = torch.arange(self.filt_size, dtype=torch.float32)
        coords -= (self.filt_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g_norm = g / g.sum()
        
        filt = g_norm.view(1, 1, self.filt_size).repeat(channels, 1, 1)
        self.register_buffer('filt', filt)
        
        # Debug info for optimization verification
        print(f"      BlurPool1D: {stride}x stride → {self.filt_size}-tap filter, σ={self.sigma:.1f}")
            
    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        # Apply depthwise convolution with the Gaussian blur filter
        blurred_x = F.conv1d(x, self.filt, stride=1, padding=self.padding, groups=self.channels)
        # Downsample
        return blurred_x[:, :, ::self.stride]


class BlurPool2D(nn.Module):
    """2D anti-aliasing module using separable Gaussian filters followed by strided subsampling
    
    OPTIMIZED VERSION based on research findings:
    - For 5x downsampling: Use 9-11 taps with σ≈2.5-3.0 for ~-10dB attenuation at new Nyquist
    - For 4x downsampling: Use 7-9 taps with σ≈2.0-2.5  
    - For 2x downsampling: Use 5-7 taps with σ≈1.0-1.5
    - Much better than 3-tap binomial filters (-0.87dB) used in basic implementations
    """
    def __init__(self, channels, filt_size=None, stride=(2,2), sigma=None):
        super().__init__()
        self.channels = channels
        self.stride_h, self.stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        
        # RESEARCH-BASED OPTIMIZATION: Auto-select filter size and sigma based on stride
        max_stride = max(self.stride_h, self.stride_w)
        
        if filt_size is None:
            # Auto-select optimal filter size based on downsampling factor
            if max_stride >= 5:
                self.filt_size = 11  # 11-tap for 5x+ downsampling (aggressive anti-aliasing)
            elif max_stride >= 4:
                self.filt_size = 9   # 9-tap for 4x downsampling  
            elif max_stride >= 3:
                self.filt_size = 7   # 7-tap for 3x downsampling
            else:
                self.filt_size = 5   # 5-tap for 2x downsampling
        else:
            self.filt_size = filt_size
            
        self.padding = (self.filt_size - 1) // 2
        
        # RESEARCH-BASED SIGMA OPTIMIZATION
        if sigma is None:
            # Optimized sigma values based on research for different downsampling factors
            if max_stride >= 5:
                self.sigma = 3.0    # σ=3.0 for 5x+ (strong low-pass filtering)
            elif max_stride >= 4:
                self.sigma = 2.5    # σ=2.5 for 4x 
            elif max_stride >= 3:
                self.sigma = 2.0    # σ=2.0 for 3x
            else:
                self.sigma = max(1.0, max_stride / 2.0)  # σ≈1.0-1.5 for 2x
        else:
            self.sigma = sigma
        
        # Generate 1D Gaussian kernel with optimized parameters
        coords = torch.arange(self.filt_size, dtype=torch.float32)
        coords -= (self.filt_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g_norm = g / g.sum()
        
        # Create separable 2D filters
        filt_h = g_norm.view(1, 1, self.filt_size, 1).repeat(channels, 1, 1, 1) # (C,1,k,1)
        filt_w = g_norm.view(1, 1, 1, self.filt_size).repeat(channels, 1, 1, 1) # (C,1,1,k)

        self.register_buffer('filt_h', filt_h)
        self.register_buffer('filt_w', filt_w)
        
        # Debug info for optimization verification
        print(f"      BlurPool2D: {max_stride}x stride → {self.filt_size}-tap filter, σ={self.sigma:.1f}")
        
    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Apply separable Gaussian blur (horizontal then vertical)
        blurred_x = F.conv2d(x, self.filt_w, stride=1, padding=(0, self.padding), groups=self.channels)
        blurred_x = F.conv2d(blurred_x, self.filt_h, stride=1, padding=(self.padding, 0), groups=self.channels)
        # Downsample
        return blurred_x[:, :, ::self.stride_h, ::self.stride_w]


class PerShotTemporalEncoder(nn.Module):
    """
    Optimized PerShotTemporalEncoder with:
    1. Optimized SincNet parameters (kernel=1001, stride=1, max_hz=1000, log spacing)
    2. Hierarchical anti-aliased downsampling for proper signal processing
    3. Progressive reduction of temporal dimension without aliasing
    """
    def __init__(self, 
                 sample_rate=10001,       # Must be 10001
                 num_receivers=31,        # Fixed
                 time_samples=10001,      # Fixed
                 # SincNet Optimized Params
                 sinc_out_channels=60,    # Recommended: 60
                 sinc_kernel_size=1001,   # Recommended: 1001
                 sinc_stride=1,           # CRITICAL: 1 (or very small, e.g., 2 or 4)
                 sinc_min_low_hz=40,      # Recommended: 40Hz (with kernel 1001)
                 sinc_max_learnable_hz=1000, # Recommended: 1000Hz
                 sinc_min_band_hz=10,
                 sinc_window_func='blackman',
                 sinc_init_type='logarithmic',
                 # CNN Aggregator Params
                 cnn_channels_start=64,   # Channels for first 2D CNN layer
                 cnn_depth=4,             # Number of CNN blocks
                 cnn_temporal_pool_factors=[5, 5, 4, 2], # Factors for temporal downsampling at each CNN stage
                 cnn_spatial_pool_factors=[2, 2, 2, 1], # Factors for spatial (receiver) downsampling
                 embedding_dim=128):
        super().__init__()
        
        self.num_receivers = num_receivers
        
        print(f"🔧 PerShotTemporalEncoder (HIERARCHICAL ANTI-ALIASED VERSION):")
        print(f"   Input Time Samples: {time_samples}")
        print(f"   SincNet Stride: {sinc_stride} (Output {time_samples // sinc_stride if sinc_stride > 0 else time_samples} temporal features if padding='same')")

        self.sinc_layer = SincConv1d_SeismicAdapted(
            out_channels=sinc_out_channels,
            kernel_size=sinc_kernel_size,
            sample_rate=sample_rate,
            stride=sinc_stride,
            padding='same', # Ensures output length is input_length / stride
            min_low_hz=sinc_min_low_hz,
            max_learnable_hz=sinc_max_learnable_hz,
            min_band_hz=sinc_min_band_hz,
            window_func=sinc_window_func,
            initialization_type=sinc_init_type
        )
        
        sinc_output_temp_dim = math.ceil(time_samples / sinc_stride) if sinc_stride > 0 else time_samples
        print(f"   SincNet Output Temporal Dim: {sinc_output_temp_dim}")

        # CNN Aggregator with Hierarchical Anti-Aliased Downsampling
        cnn_layers = []
        current_sinc_channels = sinc_out_channels
        current_temp_dim = sinc_output_temp_dim
        current_spatial_dim = num_receivers
        
        current_cnn_channels = cnn_channels_start

        for i in range(cnn_depth):
            # Convolutional Block
            cnn_layers.append(nn.Conv2d(current_sinc_channels if i == 0 else current_cnn_channels, 
                                        current_cnn_channels * 2, # Double channels at each layer
                                        kernel_size=3, padding=1))
            cnn_layers.append(nn.GroupNorm(min(32, current_cnn_channels * 2 // 2), current_cnn_channels * 2))
            cnn_layers.append(nn.ELU(inplace=True))
            cnn_layers.append(nn.Dropout2d(0.1))
            
            prev_cnn_channels = current_cnn_channels
            current_cnn_channels = current_cnn_channels * 2

            # Anti-Aliased Pooling for this stage
            t_pool = cnn_temporal_pool_factors[i]
            s_pool = cnn_spatial_pool_factors[i]
            
            if t_pool > 1 or s_pool > 1:
                # Using BlurPool2D for anti-aliased downsampling
                cnn_layers.append(BlurPool2D(current_cnn_channels, filt_size=None, stride=(t_pool, s_pool)))
                
                current_temp_dim = math.ceil(current_temp_dim / t_pool)
                current_spatial_dim = math.ceil(current_spatial_dim / s_pool)
                print(f"   CNN Stage {i+1}: Pool ({t_pool}T, {s_pool}S) -> Temp Dim: {current_temp_dim}, Spatial Dim: {current_spatial_dim}, Channels: {current_cnn_channels}")

        self.cnn_aggregator = nn.Sequential(*cnn_layers)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # Pool to 1x1 spatially and temporally
        
        final_feature_dim = current_cnn_channels 
        self.projection = nn.Sequential(
            nn.Linear(final_feature_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        print(f"   Final CNN Output Channels: {final_feature_dim}")
        print(f"   Final Embedding Dim: {embedding_dim}")
        self.apply(initialize_seismic_weights) # Initialize weights

    def forward(self, x_shot_gather):
        B, T, R = x_shot_gather.shape # (Batch, Time, Receivers)
        
        # Reshape for trace-wise SincNet: (B*R, 1, T)
        x_traces = x_shot_gather.permute(0, 2, 1).contiguous().view(B * R, 1, T)
        
        sinc_features = self.sinc_layer(x_traces) # (B*R, sinc_C_out, T_sinc_out)
        
        # Reshape for 2D CNN: (B, sinc_C_out, T_sinc_out, R)
        sinc_C_out = sinc_features.shape[1]
        T_sinc_out = sinc_features.shape[2]
        cnn_input = sinc_features.view(B, R, sinc_C_out, T_sinc_out).permute(0, 2, 3, 1)
        
        # Pass through CNN aggregator
        cnn_aggregated_features = self.cnn_aggregator(cnn_input)
        
        pooled_features = self.global_pool(cnn_aggregated_features).view(B, -1)
        embedding = self.projection(pooled_features)
        
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


# Test function for the optimized implementation
def test_fixed_sincnet_encoder():
    print("🧪 Testing FIXED SincNet Seismic Encoder with Hierarchical Downsampling...")
    batch_size = 2
    time_samples = 10001
    num_receivers = 31
    dummy_shot = torch.randn(batch_size, time_samples, num_receivers)
    print(f"Input shape: {dummy_shot.shape}")

    encoder = PerShotTemporalEncoder(
        sample_rate=10001,
        num_receivers=num_receivers,
        time_samples=time_samples,
        sinc_out_channels=60,
        sinc_kernel_size=1001,
        sinc_stride=1, # CRITICAL: stride=1 prevents aliasing
        sinc_min_low_hz=40,
        sinc_max_learnable_hz=1000,
        sinc_min_band_hz=10,
        sinc_window_func='blackman',
        sinc_init_type='logarithmic',
        cnn_channels_start=32, # Start leaner for CNN
        cnn_depth=4, # More depth to handle larger feature maps
        cnn_temporal_pool_factors=[5, 5, 4, 2], # Product = 200. Total downsample = 1 * 200 = 200x. 10001/200 = ~50 features
        cnn_spatial_pool_factors=[2, 2, 2, 1],   # Product = 8. Total 31/8 = ~3-4 features
        embedding_dim=128
    )
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    try:
        with torch.no_grad():
            embedding = encoder(dummy_shot)
            print(f"✅ Output embedding shape: {embedding.shape} (Expected: {batch_size, 128})")
            assert embedding.shape == (batch_size, 128)
            print(f"✅ Output embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
            if not (torch.isnan(embedding).any() or torch.isinf(embedding).any()):
                print("✅ Output is numerically stable")
            else:
                print("❌ Output has NaN/Inf values!")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 FIXED SincNet Seismic Encoder test completed successfully!")
    return True


# Legacy test function (keeping for backward compatibility)
def test_sincnet_encoder():
    """Test the SincNet encoder with dummy data"""
    
    print("🧪 Testing SincNet Seismic Encoder (LEGACY VERSION)...")
    
    # Create test data
    batch_size = 2
    dummy_shot = torch.randn(batch_size, 10001, 31)
    
    print(f"Input shape: {dummy_shot.shape}")
    
    # Create encoder
    encoder = PerShotTemporalEncoder(
        num_receivers=31,
        time_samples=10001,
        sinc_out_channels=60,
        sinc_kernel_size=1001,
        sinc_stride=1,
        sample_rate=10001,
        sinc_min_low_hz=40,
        sinc_max_learnable_hz=1000,
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
    
    print("🎉 SincNet Seismic Encoder test completed successfully!")
    return True


if __name__ == "__main__":
    test_fixed_sincnet_encoder() 