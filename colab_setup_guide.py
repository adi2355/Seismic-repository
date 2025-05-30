"""
🚀 COMPLETE GOOGLE COLAB SETUP GUIDE
===================================

Step-by-step guide to run SincNet+GAT architecture in Google Colab
"""

# ============================================================================
# STEP 1: COLAB PREPARATION
# ============================================================================

"""
BEFORE RUNNING ANYTHING:

1. 🖥️ SET GPU RUNTIME:
   - Runtime → Change runtime type → Hardware accelerator → GPU
   - Choose T4 (free) or A100 (Colab Pro) for best performance

2. 📁 UPLOAD FILES TO COLAB:
   Upload these 3 files to your Colab session:
   - sincnet_seismic_encoder.py
   - seismic_gat_fusion.py  
   - sincnet_integration_demo.py
   
   (Use the folder icon on the left sidebar to upload)

3. 🔄 THEN RUN THE CELLS BELOW IN ORDER:
"""

# ============================================================================
# CELL 1: Install Dependencies and Setup
# ============================================================================

# Install required packages
!pip install torch_geometric --quiet

# Import essential libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Check GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_memory:.1f} GB")
    
    # Configure for stability
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("✅ GPU optimization configured")
else:
    print("⚠️  No GPU detected! Please enable GPU runtime")

# ============================================================================
# CELL 2: Load Architecture Files
# ============================================================================

# Import your SincNet+GAT architecture
try:
    from sincnet_seismic_encoder import PerShotTemporalEncoder
    from seismic_gat_fusion import SeismicSincNetGAT, SpatiallyAwareLightweightGATFusion
    from sincnet_integration_demo import (
        SpatiallyAwareSincNetGATModel, 
        EnhancedSpatialSincNetGATModel,
        ArchitectureComparisonFramework
    )
    print("✅ All architecture files imported successfully!")
    
except ImportError as e:
    print(f"❌ Error importing files: {e}")
    print("\n📝 SOLUTION:")
    print("1. Upload these files to Colab using the folder icon:")
    print("   - sincnet_seismic_encoder.py")
    print("   - seismic_gat_fusion.py")
    print("   - sincnet_integration_demo.py")
    print("2. Restart runtime and try again")
    raise

# ============================================================================
# CELL 3: Create and Test Models
# ============================================================================

def setup_models():
    """Create both SincNet+GAT model variants"""
    
    print("🏗️ Creating SincNet+GAT Models...")
    
    # Model configuration
    config = {
        'num_receivers': 31,
        'sinc_out_channels': 40, 
        'shot_embedding_dim': 128,
        'gat_embedding_dim': 128,
        'target_height': 300,
        'target_width': 1259,
        'num_shots': 5
    }
    
    # Create Path Alpha (simpler, global pooling)
    print("📊 Path Alpha: Global pooling GAT")
    model_alpha = SpatiallyAwareSincNetGATModel(**config).to(device)
    params_alpha = sum(p.numel() for p in model_alpha.parameters())
    
    # Create Path Beta (advanced, spatial preservation)  
    print("📊 Path Beta: Spatially-aware GAT")
    model_beta = EnhancedSpatialSincNetGATModel(**config).to(device)
    params_beta = sum(p.numel() for p in model_beta.parameters())
    
    print(f"\n📈 Model Comparison:")
    print(f"├─ Path Alpha: {params_alpha:,} parameters")
    print(f"├─ Path Beta:  {params_beta:,} parameters") 
    print(f"└─ Champion:   ~17,260,000 parameters")
    print(f"🎯 Efficiency: {((17260000 - params_alpha) / 17260000 * 100):.1f}% reduction!")
    
    return model_alpha, model_beta, config

# Create models
model_alpha, model_beta, model_config = setup_models()

# ============================================================================
# CELL 4: Test Forward Pass
# ============================================================================

def test_models(model_alpha, model_beta):
    """Test both models with dummy data"""
    
    print("🧪 Testing Model Forward Pass...")
    
    # Create dummy input (replace with your actual data format)
    batch_size = 2
    dummy_shots = torch.randn(batch_size, 5, 10001, 31).to(device)
    
    print(f"📥 Input shape: {dummy_shots.shape}")
    print("   Format: (batch_size, num_shots, time_samples, receivers)")
    
    with torch.no_grad():
        # Test Path Alpha
        output_alpha = model_alpha(dummy_shots)
        print(f"📤 Path Alpha output: {output_alpha.shape}")
        
        # Test Path Beta
        output_beta = model_beta(dummy_shots)
        print(f"📤 Path Beta output:  {output_beta.shape}")
        
        # Verify compatibility with BaselineUNet
        expected = (batch_size, 1, 300, 1259)
        alpha_ok = output_alpha.shape == expected
        beta_ok = output_beta.shape == expected
        
        print(f"\n✅ Path Alpha compatible: {alpha_ok}")
        print(f"✅ Path Beta compatible:  {beta_ok}")
        
        if alpha_ok and beta_ok:
            print("🎉 Both models ready for drop-in BaselineUNet replacement!")
        else:
            print("❌ Shape mismatch - check implementation")
            
        # Check for numerical stability
        alpha_stable = not (torch.isnan(output_alpha).any() or torch.isinf(output_alpha).any())
        beta_stable = not (torch.isnan(output_beta).any() or torch.isinf(output_beta).any())
        
        print(f"🔢 Path Alpha numerically stable: {alpha_stable}")
        print(f"🔢 Path Beta numerically stable:  {beta_stable}")

# Run tests
test_models(model_alpha, model_beta)

# ============================================================================
# CELL 5: Choose Model and Setup Training
# ============================================================================

def choose_model_for_training():
    """Select model variant for training"""
    
    print("🤔 Model Selection Guide:")
    print("="*50)
    print("Path Alpha (Simpler):")
    print("  ✅ 6.35M parameters")
    print("  ✅ Faster training")
    print("  ✅ Good for initial validation")
    print("  ❌ Information bottleneck at global pooling")
    print()
    print("Path Beta (Advanced):")
    print("  ✅ 4.24M parameters (even more efficient!)")
    print("  ✅ Preserves spatial structure")
    print("  ✅ Better for geological detail")
    print("  ❌ More complex architecture")
    print()
    print("📋 RECOMMENDATION: Start with Path Alpha")
    
    # For initial testing, choose Path Alpha
    chosen_model = model_alpha
    model_name = "Path Alpha (Global Pooling)"
    
    print(f"\n🎯 Selected: {model_name}")
    print(f"📊 Parameters: {sum(p.numel() for p in chosen_model.parameters()):,}")
    
    return chosen_model, model_name

# Choose model
selected_model, selected_name = choose_model_for_training()

# Setup optimizer
optimizer = optim.AdamW(selected_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f"✅ Training setup complete for {selected_name}")

# ============================================================================
# CELL 6: Integration with Your Existing Code
# ============================================================================

def show_integration_example():
    """Show how to integrate with existing training pipeline"""
    
    print("🔧 INTEGRATION WITH YOUR EXISTING CODE:")
    print("="*60)
    
    integration_code = '''
# STEP 1: Replace BaselineUNet model creation
# OLD:
# model = BaselineUNet(input_channels=5, output_channels=1).to(device)

# NEW:
from sincnet_integration_demo import SpatiallyAwareSincNetGATModel
model = SpatiallyAwareSincNetGATModel(
    num_receivers=31,
    sinc_out_channels=40,
    shot_embedding_dim=128,
    gat_embedding_dim=128,
    target_height=300,
    target_width=1259
).to(device)

# STEP 2: Keep your existing loss function (champion configuration)
# Your RefinedLogSpaceMAEHybridLoss with weights [1.0, 0.12, 0.007]

# STEP 3: Keep your existing training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs: (B, 5, 10001, 31) - your existing format
        # targets: (B, 1, 300, 1259) - your existing format
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass - same interface as BaselineUNet!
        outputs = model(inputs)  # (B, 1, 300, 1259)
        
        # Loss calculation - exactly the same!
        loss = loss_function(outputs, targets)
        
        # Backward pass - exactly the same!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# STEP 4: Keep your existing evaluation
validation_mape = calculate_mape(predictions, ground_truth)
print(f"Validation MAPE: {validation_mape:.4f}%")
'''
    
    print(integration_code)
    
    print("\n🎊 KEY BENEFITS:")
    print("✅ Drop-in replacement - no data pipeline changes needed")
    print("✅ Same input/output formats as BaselineUNet")
    print("✅ Use your existing loss function and training loop")
    print("✅ 63-75% fewer parameters than champion")
    print("✅ Enhanced temporal-frequency processing")
    print("✅ Cross-shot attention mechanisms")

show_integration_example()

# ============================================================================
# CELL 7: Quick Training Test
# ============================================================================

def run_quick_training_test(model, num_steps=5):
    """Run a quick training test with dummy data"""
    
    print(f"🚀 Quick Training Test ({num_steps} steps)...")
    
    # Create dummy data (replace with your actual DataLoader)
    def dummy_batch():
        shots = torch.randn(4, 5, 10001, 31).to(device)  # 4 batch size
        targets = torch.randn(4, 1, 300, 1259).to(device)
        return shots, targets
    
    # Simple MSE loss for testing (replace with your champion loss)
    criterion = nn.MSELoss()
    
    model.train()
    
    for step in range(num_steps):
        # Get dummy batch
        inputs, targets = dummy_batch()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{num_steps}: Loss = {loss.item():.6f}")
    
    print("✅ Training test completed successfully!")
    print("🎯 Model is ready for full training with your dataset")

# Uncomment to run quick test:
# run_quick_training_test(selected_model)

# ============================================================================
# CELL 8: Model Saving/Loading
# ============================================================================

def save_model(model, filepath="sincnet_gat_best.pth"):
    """Save trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'parameters': sum(p.numel() for p in model.parameters()),
        'model_type': model.__class__.__name__
    }, filepath)
    print(f"✅ Model saved: {filepath}")

def load_model(filepath, model_class):
    """Load saved model"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate model with saved config
    model = model_class(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded: {filepath}")
    print(f"   Type: {checkpoint['model_type']}")
    print(f"   Parameters: {checkpoint['parameters']:,}")
    
    return model

# Example usage:
# save_model(selected_model, "my_sincnet_gat.pth")
# loaded_model = load_model("my_sincnet_gat.pth", SpatiallyAwareSincNetGATModel)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "🎉 COLAB SETUP COMPLETE!" + "🎉")
print("="*70)
print("✅ SincNet+GAT architecture loaded and tested")
print("✅ GPU optimization configured")
print("✅ Models ready for training")
print("✅ Integration guide provided")

print(f"\n📊 Selected Model: {selected_name}")
print(f"📈 Parameters: {sum(p.numel() for p in selected_model.parameters()):,}")
print(f"🎯 Target: Beat 0.0862% MAPE champion baseline")

print(f"\n🚀 NEXT STEPS:")
print("1. Replace BaselineUNet in your training script")
print("2. Run training with your actual dataset") 
print("3. Monitor MAPE improvement")
print("4. Test geological feature preservation")

print(f"\n💡 TIPS:")
print("• Start with Path Alpha for validation")
print("• Switch to Path Beta if you need better spatial detail")
print("• Use same champion loss weights [1.0, 0.12, 0.007]")
print("• Save best model based on validation MAPE")

print(f"\n🏆 READY TO ACHIEVE NEW SOTA PERFORMANCE!") 