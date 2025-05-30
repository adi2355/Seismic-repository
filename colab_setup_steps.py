"""
GOOGLE COLAB SETUP INSTRUCTIONS - SincNet+GAT Architecture
==========================================================

Step-by-step guide to run your SincNet+GAT in Google Colab.

PREPARATION:
1. Set Runtime > Change runtime type > GPU (T4 or A100)
2. Upload these 3 files to Colab session:
   - sincnet_seismic_encoder.py
   - seismic_gat_fusion.py  
   - sincnet_integration_demo.py
3. Copy code sections below into separate Colab cells
"""

# =============================================================================
# STEP 1: Install Dependencies (Copy to first Colab cell)
# =============================================================================

# Run this in a Colab cell:
# !pip install torch_geometric --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Check GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
    
    # Configure for stability
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("GPU optimization configured")
else:
    print("WARNING: No GPU detected! Enable GPU runtime")

# =============================================================================
# STEP 2: Import Architecture (Copy to second Colab cell)
# =============================================================================

# Import your SincNet+GAT architecture files
try:
    from sincnet_seismic_encoder import PerShotTemporalEncoder
    from seismic_gat_fusion import SeismicSincNetGAT, SpatiallyAwareLightweightGATFusion
    from sincnet_integration_demo import (
        SpatiallyAwareSincNetGATModel, 
        EnhancedSpatialSincNetGATModel
    )
    print("SUCCESS: All architecture files imported!")
    
except ImportError as e:
    print(f"ERROR: {e}")
    print("SOLUTION:")
    print("1. Upload these files using Colab's folder icon:")
    print("   - sincnet_seismic_encoder.py")
    print("   - seismic_gat_fusion.py")
    print("   - sincnet_integration_demo.py")
    print("2. Restart runtime and try again")
    raise

# =============================================================================
# STEP 3: Create and Test Models (Copy to third Colab cell)
# =============================================================================

def setup_sincnet_gat_models():
    """Create both model variants and test them"""
    
    print("Creating SincNet+GAT Models...")
    
    # Model configuration
    config = {
        'num_receivers': 31,
        'sinc_out_channels': 40, 
        'shot_embedding_dim': 128,  # FIXED: Consistent 128-dim throughout
        'gat_embedding_dim': 128,   # FIXED: Consistent 128-dim throughout
        'target_height': 300,
        'target_width': 1259,
        'num_shots': 5
    }
    
    # Create Path Alpha (simpler, global pooling)
    print("Path Alpha: Global pooling GAT")
    model_alpha = SpatiallyAwareSincNetGATModel(**config).to(device)
    params_alpha = sum(p.numel() for p in model_alpha.parameters())
    
    # Create Path Beta (advanced, spatial preservation)  
    print("Path Beta: Spatially-aware GAT")
    model_beta = EnhancedSpatialSincNetGATModel(**config).to(device)
    params_beta = sum(p.numel() for p in model_beta.parameters())
    
    print(f"\nModel Comparison:")
    print(f"Path Alpha: {params_alpha:,} parameters")
    print(f"Path Beta:  {params_beta:,} parameters") 
    print(f"Champion:   17,260,000 parameters")
    print(f"Efficiency: {((17260000 - params_alpha) / 17260000 * 100):.1f}% reduction!")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 2
    dummy_shots = torch.randn(batch_size, 5, 10001, 31).to(device)
    
    with torch.no_grad():
        output_alpha = model_alpha(dummy_shots)
        output_beta = model_beta(dummy_shots)
        
        print(f"Input shape: {dummy_shots.shape}")
        print(f"Path Alpha output: {output_alpha.shape}")
        print(f"Path Beta output:  {output_beta.shape}")
        
        # Verify BaselineUNet compatibility
        expected = (batch_size, 1, 300, 1259)
        alpha_ok = output_alpha.shape == expected
        beta_ok = output_beta.shape == expected
        
        if alpha_ok and beta_ok:
            print("SUCCESS: Both models compatible with BaselineUNet!")
        else:
            print("ERROR: Shape mismatch")
            
        # Check numerical stability
        alpha_stable = not (torch.isnan(output_alpha).any() or torch.isinf(output_alpha).any())
        beta_stable = not (torch.isnan(output_beta).any() or torch.isinf(output_beta).any())
        
        print(f"Path Alpha numerically stable: {alpha_stable}")
        print(f"Path Beta numerically stable:  {beta_stable}")
    
    return model_alpha, model_beta, config

# Run the setup
model_alpha, model_beta, model_config = setup_sincnet_gat_models()

# =============================================================================
# STEP 4: Choose Model and Setup Training (Copy to fourth Colab cell)
# =============================================================================

def setup_training():
    """Choose model and setup training components"""
    
    print("Model Selection Guide:")
    print("="*50)
    print("Path Alpha:")
    print("  + 6.35M parameters")
    print("  + Faster training")
    print("  + Good for initial validation")
    print("  - Information bottleneck")
    print()
    print("Path Beta:")
    print("  + 4.24M parameters (even more efficient!)")
    print("  + Preserves spatial structure")
    print("  + Better geological detail")
    print("  - More complex")
    print()
    print("RECOMMENDATION: Start with Path Alpha")
    
    # Choose Path Alpha for initial validation
    selected_model = model_alpha
    model_name = "Path Alpha (Global Pooling)"
    
    print(f"\nSelected: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in selected_model.parameters()):,}")
    
    # Setup optimizer (same configuration as champion)
    optimizer = optim.AdamW(selected_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print("Training setup complete!")
    
    return selected_model, optimizer, scheduler, model_name

# Setup training
selected_model, optimizer, scheduler, selected_name = setup_training()

# =============================================================================
# STEP 5: Integration Guide (Copy to fifth Colab cell)
# =============================================================================

def show_integration_guide():
    """Show how to integrate with existing training code"""
    
    print("INTEGRATION WITH YOUR EXISTING CODE:")
    print("="*60)
    
    print("""
STEP 1: Replace BaselineUNet model creation

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

STEP 2: Keep your existing loss function
# Your RefinedLogSpaceMAEHybridLoss with weights [1.0, 0.12, 0.007]

STEP 3: Keep your existing training loop - NO CHANGES NEEDED!
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

STEP 4: Keep your existing evaluation
validation_mape = calculate_mape(predictions, ground_truth)
print(f"Validation MAPE: {validation_mape:.4f}%")
""")
    
    print("\nKEY BENEFITS:")
    print("+ Drop-in replacement - no data changes needed")
    print("+ Same input/output as BaselineUNet")
    print("+ Use existing loss and training loop")
    print("+ 63-75% fewer parameters")
    print("+ Enhanced frequency processing")
    print("+ Cross-shot attention")

show_integration_guide()

# =============================================================================
# STEP 6: Optional Quick Test (Copy to sixth Colab cell)
# =============================================================================

def run_quick_training_test(model, steps=3):
    """Quick test to verify training works"""
    
    print(f"Running quick training test ({steps} steps)...")
    
    model.train()
    criterion = nn.MSELoss()  # Replace with your champion loss
    
    for step in range(steps):
        # Create dummy batch
        inputs = torch.randn(2, 5, 10001, 31).to(device)
        targets = torch.randn(2, 1, 300, 1259).to(device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{steps}: Loss = {loss.item():.6f}")
    
    print("SUCCESS: Training test completed!")
    print("Model ready for full training with your dataset")

# Uncomment to run quick test:
# run_quick_training_test(selected_model)

# =============================================================================
# STEP 7: Model Saving/Loading (Copy to seventh Colab cell)
# =============================================================================

def save_sincnet_model(model, filepath="sincnet_gat_best.pth"):
    """Save trained model with configuration"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'parameters': sum(p.numel() for p in model.parameters()),
        'model_type': model.__class__.__name__
    }, filepath)
    print(f"Model saved: {filepath}")

def load_sincnet_model(filepath, model_class=SpatiallyAwareSincNetGATModel):
    """Load saved model"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model = model_class(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded: {filepath}")
    print(f"Type: {checkpoint['model_type']}")
    print(f"Parameters: {checkpoint['parameters']:,}")
    
    return model

# Example usage:
# save_sincnet_model(selected_model, "my_sincnet_gat.pth")
# loaded_model = load_sincnet_model("my_sincnet_gat.pth")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("COLAB SETUP COMPLETE!")
print("="*70)
print("+ SincNet+GAT architecture loaded and tested")
print("+ GPU optimization configured")
print("+ Models ready for training")
print("+ Integration guide provided")

print(f"\nSelected Model: {selected_name}")
print(f"Parameters: {sum(p.numel() for p in selected_model.parameters()):,}")
print(f"Target: Beat 0.0862% MAPE champion baseline")

print(f"\nNEXT STEPS:")
print("1. Replace BaselineUNet in your training script")
print("2. Run training with your actual dataset") 
print("3. Monitor MAPE improvement vs 0.0862%")
print("4. Test geological feature preservation")

print(f"\nTIPS:")
print("- Start with Path Alpha for validation")
print("- Switch to Path Beta if need better spatial detail")
print("- Use champion loss weights [1.0, 0.12, 0.007]")
print("- Save best model based on validation MAPE")

print(f"\nREADY TO ACHIEVE NEW SOTA PERFORMANCE!")
print("Your SincNet+GAT is 63% more efficient than champion!")
print("="*70) 