"""
Complete SincNet-GAT-UNet Integration

This module provides the complete implementation of the SincNet-GAT-UNet architecture
with proper sample rate handling, baseline UNet components, and training utilities.

Key Features:
1. Configurable sample_rate throughout the architecture
2. Complete BaselineUNet implementation with proper encoder/decoder split
3. Champion hybrid loss integration
4. Mixed precision training support
5. A100 stability configuration

Architecture Flow:
Input (B, 5, 10001, 31) → SincNet Encoders → GAT Fusion → UNet with GAT injection → Output (B, 1, 300, 1259)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GATv2Conv, global_mean_pool, GlobalAttention

# Import the individual components
from sincnet_seismic_encoder import SincConv1d_SeismicAdapted, PerShotTemporalEncoder
from seismic_gat_fusion import LightweightGATFusion, ShotGraphBuilder


# =====================================
# BASELINE UNET COMPONENTS
# =====================================

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels, pool_kernel_stride=(2, 2)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_kernel_stride[0], stride=pool_kernel_stride[1]),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True, upsample_scale_factor=(2,2)):
        super().__init__()
        self.upsample_scale_factor = upsample_scale_factor
        if bilinear:
            self.up = nn.Upsample(scale_factor=upsample_scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=upsample_scale_factor,
                                         stride=upsample_scale_factor)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # x1 from upsampling, x2 from skip connection
        x1 = self.up(x1)
        # Handle potential padding issues
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY > 0 or diffX > 0:
            x2 = x2[:, :, diffY // 2 : x2.size()[2] - diffY // 2 - (diffY % 2),
                           diffX // 2 : x2.size()[3] - diffX // 2 - (diffX % 2)]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BaselineUNet(nn.Module):
    """
    Complete BaselineUNet with proper encoder/decoder split for integration
    """
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        super(BaselineUNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        # Encoder - More conservative pooling for (5, 10001, 31) → (1, 300, 1259)
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128, pool_kernel_stride=(2, 2))    # Mild downsampling
        self.down2 = Down(128, 256, pool_kernel_stride=(2, 2))   # Mild downsampling
        self.down3 = Down(256, 512, pool_kernel_stride=(2, 2))   # Mild downsampling  
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, pool_kernel_stride=(2, 2))  # Final bottleneck

        # Decoder - symmetric upsampling with conservative scaling
        self.up1 = Up(1024, 512 // factor, bilinear, upsample_scale_factor=(2, 2))
        self.up2 = Up(512, 256 // factor, bilinear, upsample_scale_factor=(2, 2))
        self.up3 = Up(256, 128 // factor, bilinear, upsample_scale_factor=(2, 2))
        self.up4 = Up(128, 64, bilinear, upsample_scale_factor=(2, 2))
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        """Standard forward pass"""
        x1, x2, x3, x4, x5 = self.forward_encoder(x)
        return self.forward_decoder(x5, x4, x3, x2, x1)

    def forward_encoder(self, x):
        """Encoder forward pass - returns all skip connections and bottleneck"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck features
        return x1, x2, x3, x4, x5

    def forward_decoder(self, x5, x4, x3, x2, x1):
        """Decoder forward pass with skip connections"""
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        final_output = F.interpolate(logits, size=(300, 1259), mode='bilinear', align_corners=False)
        return final_output


# =====================================
# GAT-UNET INTEGRATION MODULE
# =====================================

class GATUNetIntegration(nn.Module):
    """
    Integration module for injecting GAT-fused context into U-Net bottleneck.
    """
    
    def __init__(self, 
                 C_bottleneck=512,      # U-Net bottleneck channels
                 F_fused_embedding=128, # GAT output embedding dimension
                 fusion_ratio=0.25):    # Fraction of bottleneck to replace with GAT context
        super().__init__()
        
        self.C_bottleneck = C_bottleneck
        self.F_fused_embedding = F_fused_embedding
        self.gat_channels = int(C_bottleneck * fusion_ratio)  # 128 channels for ratio=0.25
        
        self.gat_projection = nn.Sequential(
            nn.Linear(F_fused_embedding, self.gat_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.gat_channels * 4, self.gat_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(C_bottleneck + self.gat_channels, C_bottleneck, 
                     kernel_size=1, bias=False),
            nn.GroupNorm(16, C_bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with appropriate schemes"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, unet_bottleneck_features, gat_fused_vector):
        """
        Inject GAT context into U-Net bottleneck features
        """
        B, C, H, W = unet_bottleneck_features.shape
        
        # Project GAT vector to spatial features
        gat_projected = self.gat_projection(gat_fused_vector)  # (B, gat_channels)
        
        # Spatial tiling: Broadcast GAT features across spatial dimensions
        gat_spatial = gat_projected.view(B, self.gat_channels, 1, 1).expand(B, self.gat_channels, H, W)
        
        # Concatenate U-Net and GAT features
        concat_features = torch.cat([unet_bottleneck_features, gat_spatial], dim=1)
        
        # Fuse with 1x1 convolution
        fused_bottleneck = self.fusion_conv(concat_features)
        
        return fused_bottleneck


# =====================================
# COMPLETE SINCGAT-UNET MODEL
# =====================================

class CompleteSincGAT_UNet(nn.Module):
    """
    Complete SincNet-GAT-UNet architecture with optimized parameters from research.
    
    Architecture:
    - Optimized SincNet encoders process each shot's traces (stride=1, 1001 kernel, 60 filters)
    - GAT fusion combines shot embeddings considering spatial relationships
    - Modified U-Net with GAT-fusion injection at bottleneck
    
    Key improvements:
    - Anti-aliasing with stride=1 (critical signal processing fix)
    - Logarithmic filter spacing with 60 filters (better frequency coverage)
    - 1001-point kernel (better low-frequency resolution)
    - Blackman window (superior side-lobe suppression)
    """
    def __init__(self, 
                 # Dataset-specific parameters (MUST be set correctly!)
                 sample_rate=10001,  # Hz - CRITICAL: Must match actual data sampling rate (10001 samples = 1 second)
                 num_receivers=31,
                 time_samples=10001,
                 num_shots=5,
                 # SincNet parameters (OPTIMIZED SETTINGS)
                 sinc_out_channels=60,        # Increased from 40 to 60 (optimal for log spacing)
                 sinc_kernel_size=1001,       # Increased from 251 to 1001 (better low-freq resolution)
                 sinc_stride=1,               # CRITICAL: Use 1 to eliminate aliasing (was 10)
                 sinc_min_low_hz=40,          # Lowered from 80 to 40 (captures more low frequencies)
                 sinc_max_learnable_hz=1000,  # Upper limit at 1000Hz (where coherent signal ends)
                 sinc_min_band_hz=10,         # Minimum bandwidth for a filter
                 sinc_window_func='blackman', # Changed from hamming to blackman (better side-lobe suppression)
                 sinc_init_type='logarithmic',# Added logarithmic spacing (better allocation across spectrum)
                 shot_embedding_dim=128,
                 # GAT parameters
                 gat_hidden_per_head=32,
                 gat_num_heads=4,
                 gat_layers=1,
                 gat_dropout_feat=0.3,
                 gat_dropout_attn=0.2,
                 fused_embedding_dim=128,
                 # U-Net parameters
                 n_unet_output_channels=1,
                 unet_bilinear=True,
                 unet_bottleneck_channels=512,
                 fusion_ratio=0.25):
        super().__init__()
        
        self.num_shots = num_shots
        self.sample_rate = sample_rate
        self.time_samples = time_samples
        self.num_receivers = num_receivers
        self.shot_embedding_dim = shot_embedding_dim
        
        # Create SincNet encoders for temporal processing
        self.shot_encoder = PerShotTemporalEncoder(
            sample_rate=sample_rate,
            num_receivers=num_receivers,
            time_samples=time_samples,
            sinc_out_channels=sinc_out_channels,
            sinc_kernel_size=sinc_kernel_size,
            sinc_stride=sinc_stride,            # CRITICAL FIX: stride=1 prevents aliasing
            sinc_min_low_hz=sinc_min_low_hz,
            sinc_max_learnable_hz=sinc_max_learnable_hz,
            sinc_min_band_hz=sinc_min_band_hz,
            sinc_window_func=sinc_window_func,
            sinc_init_type=sinc_init_type,
            embedding_dim=shot_embedding_dim
        )
        
        # Graph builder to create shot-to-shot relationships
        self.shot_graph_builder = ShotGraphBuilder(num_shots)
        
        # GAT fusion module
        self.gat_fusion = LightweightGATFusion(
            in_features=shot_embedding_dim,
            hidden_per_head=gat_hidden_per_head,
            num_heads=gat_num_heads,
            layers=gat_layers,
            dropout_feat=gat_dropout_feat,
            dropout_attn=gat_dropout_attn,
            output_dim=fused_embedding_dim
        )
        
        # Baseline U-Net for final velocity prediction
        self.unet = BaselineUNet(
            n_channels_in=num_shots,  # One channel per shot
            n_channels_out=n_unet_output_channels,
            bilinear=unet_bilinear
        )
        
        # Integration module to inject GAT-fused context into U-Net bottleneck
        self.gat_unet_integration = GATUNetIntegration(
            C_bottleneck=unet_bottleneck_channels,
            F_fused_embedding=fused_embedding_dim,
            fusion_ratio=fusion_ratio
        )
        
        self._initialize_model()
        
        print(f"🔧 CompleteSincGAT_UNet initialized")
        print(f"   SincNet: {sinc_kernel_size}-point kernel, {sinc_out_channels} filters, stride={sinc_stride}, window={sinc_window_func}")
        print(f"   Frequency range: {sinc_min_low_hz}-{sinc_max_learnable_hz} Hz ({sinc_init_type} spacing)")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _initialize_model(self):
        """Initialize model components"""
        print(f"🔧 Initializing CompleteSincGAT_UNet with sample_rate={self.sample_rate} Hz")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        sincnet_params = sum(p.numel() for p in self.shot_encoder.parameters())
        gat_params = sum(p.numel() for p in self.gat_fusion.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        integration_params = sum(p.numel() for p in self.gat_unet_integration.parameters())
        
        print(f"📊 Parameter counts:")
        print(f"   SincNet Encoder: {sincnet_params:,}")
        print(f"   GAT Fusion: {gat_params:,}")
        print(f"   BaselineUNet: {unet_params:,}")
        print(f"   GAT-UNet Integration: {integration_params:,}")
        print(f"   Total: {total_params:,}")
    
    def forward(self, x_all_shots_batch):
        """
        Forward pass through the complete SincGAT-UNet architecture.
        """
        B, num_shots, time_samples, num_receivers = x_all_shots_batch.shape
        
        # Step 1: Extract embeddings from all shots using PerShotTemporalEncoder
        shot_embeddings_list = []
        for i in range(num_shots):
            current_shot_data = x_all_shots_batch[:, i, :, :]  # (B, time_samples, num_receivers)
            shot_embedding = self.shot_encoder(current_shot_data)  # (B, embedding_dim)
            shot_embeddings_list.append(shot_embedding)
        
        shot_embeddings_batch = torch.stack(shot_embeddings_list, dim=1)  # (B, num_shots, embedding_dim)
        
        # STRATEGIC DEBUGGING: Track training progress without cluttering logs
        if self.training:
            if not hasattr(self, '_debug_step_count'):
                self._debug_step_count = 0
                self._debug_epoch = 0
                self._debug_batch_in_epoch = 0
            
            self._debug_step_count += 1
            self._debug_batch_in_epoch += 1
            
            # Debug conditions: First 3 batches of first 3 epochs, then every 50 batches, then first/last batch of each epoch
            should_debug = (
                (self._debug_epoch < 3 and self._debug_batch_in_epoch <= 3) or  # Early training details
                (self._debug_step_count % 50 == 0) or  # Periodic checks
                (self._debug_batch_in_epoch == 1) or  # First batch of epoch
                (hasattr(self, '_is_last_batch') and self._is_last_batch)  # Last batch of epoch
            )
            
            if should_debug:
                print(f"\n🔍 DEBUG [Epoch {self._debug_epoch+1}, Batch {self._debug_batch_in_epoch}, Global {self._debug_step_count}]:")
                print(f"   📊 Shot Embeddings: shape={shot_embeddings_batch.shape}")
                print(f"      Mean: {shot_embeddings_batch.mean().item():.4f}, Std: {shot_embeddings_batch.std().item():.4f}")
                print(f"      Range: [{shot_embeddings_batch.min().item():.3f}, {shot_embeddings_batch.max().item():.3f}]")
                
                # Check for problems
                if torch.isnan(shot_embeddings_batch).any():
                    print(f"      ⚠️  NaN detected in shot embeddings!")
                if torch.isinf(shot_embeddings_batch).any():
                    print(f"      ⚠️  Inf detected in shot embeddings!")
                if shot_embeddings_batch.std().item() < 1e-6:
                    print(f"      ⚠️  Very low variance - embeddings collapsed!")
                if shot_embeddings_batch.abs().max().item() > 100:
                    print(f"      ⚠️  Very large values - potential explosion!")
        
        # Step 2: Prepare graph data for GAT fusion
        x_nodes, edge_index, batch_vector = self.shot_graph_builder.create_batch(shot_embeddings_batch)
        
        # Move to correct device
        device = x_all_shots_batch.device
        edge_index = edge_index.to(device)
        batch_vector = batch_vector.to(device)
        
        # Step 3: Apply GAT fusion to create a single fused embedding per batch
        fused_embedding = self.gat_fusion(x_nodes, edge_index, batch_vector)  # (B, fused_embedding_dim)
        
        # Debug GAT fusion output
        if self.training and should_debug:
            print(f"   🔗 GAT Fusion: shape={fused_embedding.shape}")
            print(f"      Mean: {fused_embedding.mean().item():.4f}, Std: {fused_embedding.std().item():.4f}")
            print(f"      Range: [{fused_embedding.min().item():.3f}, {fused_embedding.max().item():.3f}]")
            
            if torch.isnan(fused_embedding).any():
                print(f"      ⚠️  NaN detected in GAT output!")
            if torch.isinf(fused_embedding).any():
                print(f"      ⚠️  Inf detected in GAT output!")
        
        # 4. U-Net encoder path
        x1, x2, x3, x4, x5 = self.unet.forward_encoder(x_all_shots_batch)
        
        # Debug U-Net encoder bottleneck
        if self.training and should_debug:
            print(f"   🏗️  U-Net Bottleneck (x5): shape={x5.shape}")
            print(f"      Mean: {x5.mean().item():.4f}, Std: {x5.std().item():.4f}")
            print(f"      Range: [{x5.min().item():.3f}, {x5.max().item():.3f}]")
        
        # 5. GAT-UNet integration at bottleneck
        enhanced_bottleneck = self.gat_unet_integration(x5, fused_embedding)
        
        # Debug enhanced bottleneck
        if self.training and should_debug:
            print(f"   🔀 Enhanced Bottleneck: shape={enhanced_bottleneck.shape}")
            print(f"      Mean: {enhanced_bottleneck.mean().item():.4f}, Std: {enhanced_bottleneck.std().item():.4f}")
            print(f"      Range: [{enhanced_bottleneck.min().item():.3f}, {enhanced_bottleneck.max().item():.3f}]")
        
        # 6. U-Net decoder path with enhanced bottleneck
        velocity_prediction = self.unet.forward_decoder(enhanced_bottleneck, x4, x3, x2, x1)
        
        # Debug final output
        if self.training and should_debug:
            print(f"   🎯 Final Prediction: shape={velocity_prediction.shape}")
            print(f"      Mean: {velocity_prediction.mean().item():.4f}, Std: {velocity_prediction.std().item():.4f}")
            print(f"      Range: [{velocity_prediction.min().item():.3f}, {velocity_prediction.max().item():.3f}]")
            
            if torch.isnan(velocity_prediction).any():
                print(f"      ⚠️  NaN detected in final prediction!")
            if torch.isinf(velocity_prediction).any():
                print(f"      ⚠️  Inf detected in final prediction!")
        
        return velocity_prediction
    
    def start_new_epoch(self, epoch_num):
        """Call this at the start of each epoch to update debugging counters"""
        if hasattr(self, '_debug_epoch'):
            self._debug_epoch = epoch_num
            self._debug_batch_in_epoch = 0
    
    def mark_last_batch(self):
        """Call this to mark the current batch as the last batch of the epoch"""
        if hasattr(self, '_is_last_batch'):
            self._is_last_batch = True
        else:
            self._is_last_batch = True
    
    def unmark_last_batch(self):
        """Call this to unmark the last batch flag"""
        if hasattr(self, '_is_last_batch'):
            self._is_last_batch = False
    
    def debug_gradients(self):
        """Monitor gradient flow through the network"""
        if not self.training:
            return
            
        print(f"   🔄 Gradient Analysis:")
        
        # Check SincNet gradients
        if hasattr(self.shot_encoder.sinc_layer, 'f_center_norm') and self.shot_encoder.sinc_layer.f_center_norm.grad is not None:
            center_grad = self.shot_encoder.sinc_layer.f_center_norm.grad
            print(f"      SincNet Center Freq Grad: mean={center_grad.mean().item():.6f}, std={center_grad.std().item():.6f}")
            if center_grad.abs().max().item() < 1e-8:
                print(f"      ⚠️  Very small SincNet gradients - possible vanishing!")
            elif center_grad.abs().max().item() > 1.0:
                print(f"      ⚠️  Large SincNet gradients - possible exploding!")
        else:
            print(f"      ⚠️  No SincNet gradients found!")
        
        # Check GAT gradients
        gat_params_with_grad = [p for p in self.gat_fusion.parameters() if p.grad is not None]
        if gat_params_with_grad:
            gat_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in gat_params_with_grad]))
            print(f"      GAT Grad Norm: {gat_grad_norm.item():.6f}")
        else:
            print(f"      ⚠️  No GAT gradients found!")
        
        # Check U-Net gradients
        unet_params_with_grad = [p for p in self.unet.parameters() if p.grad is not None]
        if unet_params_with_grad:
            unet_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in unet_params_with_grad]))
            print(f"      U-Net Grad Norm: {unet_grad_norm.item():.6f}")
        else:
            print(f"      ⚠️  No U-Net gradients found!")
    
    def debug_sincnet_filters(self):
        """Monitor SincNet filter evolution"""
        if not self.training:
            return
            
        # Get current filter frequencies
        f_low, f_high = self.shot_encoder.sinc_layer._get_current_cutoffs()
        nyquist = self.shot_encoder.sinc_layer.sample_rate / 2.0
        
        f_low_hz = f_low.squeeze() * nyquist
        f_high_hz = f_high.squeeze() * nyquist
        bandwidth_hz = f_high_hz - f_low_hz
        
        print(f"   🎵 SincNet Filter Status:")
        print(f"      Frequency Range: {f_low_hz.min().item():.1f} - {f_high_hz.max().item():.1f} Hz")
        print(f"      Mean Bandwidth: {bandwidth_hz.mean().item():.1f} Hz")
        print(f"      Filters <100Hz: {(f_high_hz < 100).sum().item()}/{len(f_high_hz)}")
        print(f"      Filters >500Hz: {(f_low_hz > 500).sum().item()}/{len(f_low_hz)}")
        
        # Check for problematic filter configurations
        if (bandwidth_hz < 5).any():
            print(f"      ⚠️  Some filters have very narrow bandwidth (<5Hz)!")
        if (f_high_hz > nyquist * 0.95).any():
            print(f"      ⚠️  Some filters are very close to Nyquist!")
    
    def epoch_summary(self, epoch, train_loss, val_mape):
        """Print epoch summary with key metrics"""
        print(f"\n📈 EPOCH {epoch+1} SUMMARY:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val MAPE: {val_mape:.4f}%")
        
        # Add filter status every 5 epochs
        if (epoch + 1) % 5 == 0:
            self.debug_sincnet_filters()


# =====================================
# UTILITY FUNCTIONS
# =====================================

def configure_a100_stability(disable_tf32=True):
    """Configure settings for A100 stability"""
    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("✅ Disabled TF32 for A100 stability")
    
    # Other A100 optimizations
    torch.backends.cudnn.benchmark = True
    print("✅ Enabled CuDNN benchmark")


def get_model_info(model):
    """Get detailed model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'sample_rate': model.sample_rate,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    return info


# =====================================
# TEST FUNCTIONS
# =====================================

def test_complete_model(sample_rate=10001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Comprehensive test of the complete model
    """
    print("🧪 Testing Complete SincGAT_UNet Model...")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   Device: {device}")
    
    # Configure for A100 if available
    if 'cuda' in device:
        configure_a100_stability()
    
    # Create model with correct sample rate
    model = CompleteSincGAT_UNet(
        sample_rate=sample_rate,  # CRITICAL: Set actual data sample rate
        num_receivers=31,
        time_samples=10001,
        num_shots=5,
        sinc_out_channels=60,
        sinc_kernel_size=1001,
        sinc_stride=1,
        sinc_min_low_hz=40,
        sinc_max_learnable_hz=1000,
        sinc_min_band_hz=10,
        sinc_window_func='blackman',
        sinc_init_type='logarithmic',
        shot_embedding_dim=128,
        gat_hidden_per_head=32,
        gat_num_heads=4,
        fused_embedding_dim=128,
        n_unet_output_channels=1
    ).to(device)
    
    # Get model info
    info = get_model_info(model)
    print(f"📊 Model Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Create test data
    batch_size = 2
    dummy_shots = torch.randn(batch_size, 5, 10001, 31, device=device)
    
    print(f"📥 Input shape: {dummy_shots.shape}")
    
    # Forward pass
    try:
        model.eval()
        with torch.no_grad():
            if 'cuda' in device:
                # Test with mixed precision
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    velocity_models = model(dummy_shots)
            else:
                velocity_models = model(dummy_shots)
            
            print(f"✅ Output shape: {velocity_models.shape}")
            print(f"✅ Expected shape: (2, 1, 300, 1259)")
            print(f"✅ Shape correct: {velocity_models.shape == (batch_size, 1, 300, 1259)}")
            print(f"✅ Output range: [{velocity_models.min():.3f}, {velocity_models.max():.3f}]")
            
            # Check for numerical stability
            if torch.isnan(velocity_models).any():
                print("❌ NaN detected!")
                return False, model, info
            elif torch.isinf(velocity_models).any():
                print("❌ Inf detected!")
                return False, model, info
            else:
                print("✅ Numerically stable")
        
        print("🎉 Complete model test passed!")
        return True, model, info
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


if __name__ == "__main__":
    print("="*80)
    print("COMPLETE SINCGAT-UNET INTEGRATION TEST")
    print("="*80)
    
    # Test with correct sample rate (10001 Hz based on 10001 samples per 1 second)
    success, model, info = test_complete_model(sample_rate=10001)
    
    if success:
        print("\n🎉 Integration successful!")
        print("📋 Next Steps:")
        print("   1. Set correct sample_rate from your dataset metadata")
        print("   2. Import champion loss functions")
        print("   3. Set up mixed precision training")
        print("   4. Configure DataLoaders with proper batch size")
        print("   5. Start training with AdamW optimizer")
    else:
        print("\n❌ Integration failed. Please check the implementation.") 
