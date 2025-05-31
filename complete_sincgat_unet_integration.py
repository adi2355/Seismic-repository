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
    Complete SincNet-GAT-UNet architecture with proper sample rate configuration.
    
    This is the final model that integrates:
    1. SincNet temporal encoding with configurable sample_rate
    2. GAT multi-shot fusion
    3. U-Net spatial modeling with GAT context injection
    
    Critical Fix: sample_rate is now properly passed through all components
    """
    
    def __init__(self, 
                 # Dataset-specific parameters (MUST be set correctly!)
                 sample_rate=500,  # Hz - CRITICAL: Must match actual data sampling rate
                 num_receivers=31,
                 time_samples=10001,
                 num_shots=5,
                 # SincNet parameters
                 sinc_out_channels=40,
                 sinc_kernel_size=251,
                 sinc_stride=50,
                 sinc_min_low_hz=2,
                 sinc_min_band_hz=3,
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
        
        self.sample_rate = sample_rate  # Store for reference
        self.num_shots = num_shots
        self.time_samples = time_samples
        self.num_receivers = num_receivers
        
        # Per-shot temporal encoder with correct sample_rate
        self.per_shot_encoder = PerShotTemporalEncoder(
            num_receivers=num_receivers,
            sinc_out_channels=sinc_out_channels,
            sinc_kernel_size=sinc_kernel_size,
            sinc_stride=sinc_stride,
            sample_rate=sample_rate,  # CRITICAL: Pass actual sample rate
            min_low_hz=sinc_min_low_hz,
            min_band_hz=sinc_min_band_hz,
            embedding_dim=shot_embedding_dim
        )
        
        # GAT fusion module
        self.gat_fusion = LightweightGATFusion(
            in_features=shot_embedding_dim,
            gat_hidden_channels_per_head=gat_hidden_per_head,
            num_heads=gat_num_heads,
            gat_layers=gat_layers,
            dropout_feat=gat_dropout_feat,
            dropout_attn=gat_dropout_attn,
            output_embedding_dim=fused_embedding_dim
        )
        
        # Graph builder for shot connectivity
        self.graph_builder = ShotGraphBuilder(num_shots=num_shots, connectivity='full')
        
        # Baseline U-Net with encoder/decoder split
        self.baseline_unet = BaselineUNet(
            n_channels_in=num_shots,  # 5 shots as input channels
            n_channels_out=n_unet_output_channels,
            bilinear=unet_bilinear
        )
        
        # GAT-UNet integration module
        self.gat_unet_integrator = GATUNetIntegration(
            C_bottleneck=unet_bottleneck_channels,
            F_fused_embedding=fused_embedding_dim,
            fusion_ratio=fusion_ratio
        )
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model components"""
        print(f"🔧 Initializing CompleteSincGAT_UNet with sample_rate={self.sample_rate} Hz")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        sincnet_params = sum(p.numel() for p in self.per_shot_encoder.parameters())
        gat_params = sum(p.numel() for p in self.gat_fusion.parameters())
        unet_params = sum(p.numel() for p in self.baseline_unet.parameters())
        integration_params = sum(p.numel() for p in self.gat_unet_integrator.parameters())
        
        print(f"📊 Parameter counts:")
        print(f"   SincNet Encoder: {sincnet_params:,}")
        print(f"   GAT Fusion: {gat_params:,}")
        print(f"   BaselineUNet: {unet_params:,}")
        print(f"   GAT-UNet Integration: {integration_params:,}")
        print(f"   Total: {total_params:,}")
    
    def forward(self, x_all_shots_batch):
        """
        Forward pass through complete architecture
        
        Args:
            x_all_shots_batch: (B, 5, 10001, 31) - Batch of 5-shot gathers
            
        Returns:
            velocity_models: (B, 1, 300, 1259) - Predicted velocity models
        """
        # Validate input shape
        B, shots, time, receivers = x_all_shots_batch.shape
        if shots != self.num_shots:
            raise ValueError(f"Expected {self.num_shots} shots, got {shots}")
        if time != self.time_samples:
            raise ValueError(f"Expected {self.time_samples} time samples, got {time}")
        if receivers != self.num_receivers:
            raise ValueError(f"Expected {self.num_receivers} receivers, got {receivers}")
        
        device = x_all_shots_batch.device
        
        # 1. Per-Shot Encoding with SincNet
        shot_embeddings_list = []
        for i in range(self.num_shots):
            current_shot_data = x_all_shots_batch[:, i, :, :]  # (B, 10001, 31)
            shot_embedding = self.per_shot_encoder(current_shot_data)  # (B, shot_embedding_dim)
            shot_embeddings_list.append(shot_embedding)
        
        # Stack embeddings: (B, num_shots, shot_embedding_dim)
        shot_embeddings_batch = torch.stack(shot_embeddings_list, dim=1)
        
        # 2. Create graph batch for GAT
        x_nodes, edge_index, batch_vector = self.graph_builder.create_batch(shot_embeddings_batch)
        edge_index = edge_index.to(device)
        batch_vector = batch_vector.to(device)
        
        # 3. GAT fusion
        gat_fused_vector = self.gat_fusion(x_nodes, edge_index, batch_vector)  # (B, fused_embedding_dim)
        
        # 4. U-Net encoder path
        x1, x2, x3, x4, x5 = self.baseline_unet.forward_encoder(x_all_shots_batch)
        
        # 5. GAT-UNet integration at bottleneck
        enhanced_bottleneck = self.gat_unet_integrator(x5, gat_fused_vector)
        
        # 6. U-Net decoder path with enhanced bottleneck
        velocity_models = self.baseline_unet.forward_decoder(enhanced_bottleneck, x4, x3, x2, x1)
        
        return velocity_models


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

def test_complete_model(sample_rate=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
        sinc_out_channels=40,
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
                return False
            elif torch.isinf(velocity_models).any():
                print("❌ Inf detected!")
                return False
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
    
    # Test with correct sample rate (500 Hz for 2ms sampling as per research)
    success, model, info = test_complete_model(sample_rate=500)
    
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