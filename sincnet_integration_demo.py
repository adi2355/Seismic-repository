"""
SincNet + GAT Integration Demo

This script demonstrates how to integrate the SincNet+GAT encoder 
with the existing champion BaselineUNet architecture that achieved 0.0862% MAPE.

Architecture Comparison:
1. Original Champion: Concatenate 5 shots → U-Net → (B, 1, 300, 1259)
2. Enhanced SincNet: SincNet+GAT → Features → Modified Decoder → (B, 1, 300, 1259)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from seismic_gat_fusion import SeismicSincNetGAT


class SincNetEnhancedDecoder(nn.Module):
    """
    Enhanced decoder that takes SincNet+GAT features and produces velocity models.
    
    Adapts the proven U-Net decoder structure to work with SincNet+GAT embeddings.
    Maintains compatibility with the champion loss function.
    """
    
    def __init__(self, 
                 input_embedding_dim=128,  # From SincNet+GAT
                 target_height=300,
                 target_width=1259,
                 hidden_channels=[256, 512, 256, 128, 64]):
        super().__init__()
        
        self.target_height = target_height
        self.target_width = target_width
        
        # Project embedding to initial feature map
        # Start with a reasonable spatial size that can be upsampled
        initial_h, initial_w = 10, 40  # Will be upsampled to 300x1259
        self.initial_spatial_size = initial_h * initial_w
        
        self.embedding_to_features = nn.Sequential(
            nn.Linear(input_embedding_dim, hidden_channels[0] * self.initial_spatial_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder layers - progressive upsampling
        self.decoder_layers = nn.ModuleList()
        current_channels = hidden_channels[0]
        
        for i, out_channels in enumerate(hidden_channels[1:]):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, out_channels, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(min(8, out_channels//2) if out_channels >= 16 else out_channels, 
                                out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = out_channels
        
        # Final layer to get single channel output
        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=3, padding=1)
        
        # Adaptive interpolation to exact target size
        self.final_interpolation = True  # Use interpolation for exact sizing
        
    def forward(self, embeddings):
        """
        Convert SincNet+GAT embeddings to velocity models
        
        Args:
            embeddings: (B, input_embedding_dim) from SincNet+GAT
            
        Returns:
            velocity_models: (B, 1, target_height, target_width)
        """
        B = embeddings.size(0)
        
        # Project to feature map
        features = self.embedding_to_features(embeddings)  # (B, C*H*W)
        
        # Reshape to spatial feature map
        features = features.view(B, -1, int(self.initial_spatial_size**0.5), 
                               int(self.initial_spatial_size**0.5))
        
        # Progressive upsampling
        for decoder_layer in self.decoder_layers:
            features = decoder_layer(features)
        
        # Final convolution
        output = self.final_conv(features)
        
        # Interpolate to exact target size
        if self.final_interpolation:
            output = F.interpolate(output, size=(self.target_height, self.target_width), 
                                 mode='bilinear', align_corners=False)
        
        return output


class SincNetEnhancedSeismicModel(nn.Module):
    """
    Complete enhanced seismic model combining SincNet+GAT encoder with decoder.
    
    This replaces the original BaselineUNet while maintaining output compatibility
    for seamless integration with the champion loss function.
    """
    
    def __init__(self,
                 # SincNet+GAT parameters
                 num_receivers=31,
                 sinc_out_channels=40,
                 shot_embedding_dim=128,
                 gat_embedding_dim=128,
                 # Decoder parameters
                 target_height=300,
                 target_width=1259):
        super().__init__()
        
        # SincNet+GAT encoder
        self.encoder = SeismicSincNetGAT(
            num_receivers=num_receivers,
            sinc_out_channels=sinc_out_channels,
            shot_embedding_dim=shot_embedding_dim,
            final_embedding_dim=gat_embedding_dim,
            num_shots=5
        )
        
        # Enhanced decoder
        self.decoder = SincNetEnhancedDecoder(
            input_embedding_dim=gat_embedding_dim,
            target_height=target_height,
            target_width=target_width
        )
        
    def forward(self, shot_gathers):
        """
        Forward pass compatible with existing data format
        
        Args:
            shot_gathers: Can be either:
                - List of 5 tensors, each (B, 10001, 31)  
                - Tensor (B, 5, 10001, 31)
                - Tensor (B, 5, 10001, 31) concatenated as (B, 5, 10001, 31)
                
        Returns:
            velocity_models: (B, 1, 300, 1259) - Same format as BaselineUNet
        """
        # Encode with SincNet+GAT
        embeddings = self.encoder(shot_gathers)
        
        # Decode to velocity model
        velocity_models = self.decoder(embeddings)
        
        return velocity_models


def create_champion_comparison():
    """Create both models for comparison"""
    
    print("🏗️ Creating Model Architectures for Comparison...")
    
    # Enhanced SincNet+GAT model
    sincnet_model = SincNetEnhancedSeismicModel(
        num_receivers=31,
        sinc_out_channels=40,
        shot_embedding_dim=128,
        gat_embedding_dim=128
    )
    
    sincnet_params = sum(p.numel() for p in sincnet_model.parameters())
    print(f"✅ SincNet+GAT Enhanced Model: {sincnet_params:,} parameters")
    
    # For comparison - would be BaselineUNet (but we don't import it here)
    print(f"📊 Comparison with Champion BaselineUNet:")
    print(f"   - Champion achieved: 0.0862% MAPE")
    print(f"   - Champion loss weights: [1.0, 0.12, 0.007]")
    print(f"   - Champion architecture: 5-channel input U-Net")
    
    return sincnet_model


def test_integration_compatibility():
    """Test integration compatibility with existing data pipeline"""
    
    print("\n🧪 Testing Integration Compatibility...")
    
    # Create model
    model = SincNetEnhancedSeismicModel()
    
    # Test with different input formats that might come from existing pipeline
    batch_size = 2
    
    # Format 1: List of shots (common in custom data loaders)
    print("\n📋 Testing Format 1: List of shots")
    shot_list = [torch.randn(batch_size, 10001, 31) for _ in range(5)]
    
    with torch.no_grad():
        output1 = model(shot_list)
        print(f"   Input: List of 5 shots, each {shot_list[0].shape}")
        print(f"   Output: {output1.shape}")
        
    # Format 2: Stacked tensor (common in batched processing)
    print("\n📋 Testing Format 2: Stacked tensor")
    shot_tensor = torch.stack(shot_list, dim=1)  # (B, 5, 10001, 31)
    
    with torch.no_grad():
        output2 = model(shot_tensor)
        print(f"   Input: {shot_tensor.shape}")
        print(f"   Output: {output2.shape}")
        
    # Verify compatibility
    if output1.shape == output2.shape and output1.shape == (batch_size, 1, 300, 1259):
        print("✅ Perfect compatibility with BaselineUNet output format!")
        print("✅ Ready for integration with champion loss function")
    else:
        print("❌ Output format mismatch - needs adjustment")
        
    # Check numerical stability
    if not (torch.isnan(output1).any() or torch.isnan(output2).any()):
        print("✅ Numerically stable outputs")
    else:
        print("❌ Numerical instability detected")
        
    print(f"\n📈 Output Statistics:")
    print(f"   Range: [{output1.min():.3f}, {output1.max():.3f}]")
    print(f"   Mean: {output1.mean():.3f}, Std: {output1.std():.3f}")
    
    return model


def demonstrate_training_integration():
    """Demonstrate how this integrates with existing training loop"""
    
    print("\n🎯 Training Integration Demonstration...")
    print("="*60)
    
    print("Original Champion Training Loop:")
    print("""
    # Original approach
    for batch in dataloader:
        inputs, targets = batch  # inputs: (B, 5, 10001, 31), targets: (B, 1, 300, 1259)
        
        # Concatenate shots as channels
        model_input = inputs.view(B, 5, 10001, 31)  # Current BaselineUNet format
        outputs = baseline_unet(model_input)
        
        # Apply champion loss
        loss = champion_loss(outputs, targets)  # [1.0, 0.12, 0.007] weights
        loss.backward()
        optimizer.step()
    """)
    
    print("\nEnhanced SincNet Integration:")
    print("""
    # Enhanced approach - drop-in replacement!
    for batch in dataloader:
        inputs, targets = batch  # Same data format!
        
        # SincNet+GAT processing (handles 5-shot input automatically)
        outputs = sincnet_enhanced_model(inputs)  # Same output shape!
        
        # Use SAME champion loss function - no changes needed!
        loss = champion_loss(outputs, targets)  # Same [1.0, 0.12, 0.007] weights
        loss.backward()
        optimizer.step()
    """)
    
    print("🎊 Key Advantages:")
    print("✅ Drop-in replacement for BaselineUNet")
    print("✅ Same data format and loss function")
    print("✅ Enhanced temporal-frequency learning")
    print("✅ Spatial attention across shot positions")
    print("✅ Preserves champion optimization strategies")


if __name__ == "__main__":
    print("🚀 SincNet + GAT Integration Demonstration")
    print("="*60)
    
    # Create models
    model = create_champion_comparison()
    
    # Test compatibility
    test_integration_compatibility()
    
    # Show training integration
    demonstrate_training_integration()
    
    print("\n🎉 Integration demonstration complete!")
    print("Ready to replace BaselineUNet with SincNet+GAT enhanced architecture!") 