"""
SincGAT Sample Rate Test

This script demonstrates how to properly initialize and use the CompleteSincGAT_UNet
with the correct sample_rate (10001 Hz) based on the dataset characteristics.

It includes a diagnostic test and example code for integration into experimental framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from complete_sincgat_unet_integration import CompleteSincGAT_UNet, configure_a100_stability, get_model_info

def test_sincgat_init_and_inference():
    """Test proper initialization and inference with CompleteSincGAT_UNet"""
    print("\n" + "="*70)
    print("SINCGAT SAMPLE RATE VERIFICATION TEST")
    print("="*70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configure for A100 stability if using CUDA
    if device.type == 'cuda':
        configure_a100_stability(disable_tf32=True)
    
    # Create model with the correct sample rate
    print("\n📡 Creating CompleteSincGAT_UNet with correct sample_rate (10001 Hz)...")
    model = CompleteSincGAT_UNet(
        sample_rate=10001,  # CRITICAL: 10001 Hz (10001 samples = 1 second)
        num_receivers=31,
        time_samples=10001,
        num_shots=5,
        sinc_out_channels=40,
        sinc_kernel_size=251,
        sinc_stride=50,
        sinc_min_low_hz=80,   # Based on kernel size constraints
        sinc_min_band_hz=10,  # Minimum bandwidth parameter
        shot_embedding_dim=128,
        gat_hidden_per_head=32,
        gat_num_heads=4,
        fused_embedding_dim=128,
        n_unet_output_channels=1
    ).to(device)
    
    # Print model info
    info = get_model_info(model)
    print("\n📊 Model Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Create dummy input data
    print("\n🔢 Testing with dummy data...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 5, 10001, 31, device=device)
    
    # Run forward pass
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        try:
            # Test with mixed precision if CUDA is available
            if device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    output = model(dummy_input)
            else:
                output = model(dummy_input)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            print(f"✅ Forward pass successful!")
            print(f"✅ Inference time: {inference_time:.4f} seconds")
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Output dtype: {output.dtype}")
            print(f"✅ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Check for numerical issues
            if torch.isnan(output).any():
                print("❌ Warning: NaN values detected in output")
            elif torch.isinf(output).any():
                print("❌ Warning: Inf values detected in output")
            else:
                print("✅ Output is numerically stable")
            
            return True, model
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None

def generate_notebook_code_example():
    """Generate code example for Colab notebook integration"""
    print("\n" + "="*70)
    print("EXAMPLE CODE FOR COLAB NOTEBOOK INTEGRATION")
    print("="*70)
    
    code_example = """
# In your Colab notebook:

# 1. Install torch_geometric if not already installed
!pip install torch-geometric

# 2. Import the CompleteSincGAT_UNet model
from complete_sincgat_unet_integration import CompleteSincGAT_UNet, configure_a100_stability

# 3. Define test functions for SincGAT_UNet with correct sample_rate

def test_sincgat_unet_default_loss(num_epochs=25, sample_rate=10001, batch_size_override=None):
    print("🚀 Testing CompleteSincGAT_UNet with Default L1 Loss...")
    
    # Configure A100 stability
    if 'cuda' in str(device):
        configure_a100_stability(disable_tf32=True, verbose=True)

    # Setup data loaders (reuse your existing function)
    current_batch_size = batch_size_override if batch_size_override else 4  # Default to 4 for SincGAT
    train_loader, val_loader = setup_phase2_data_loaders(batch_size=current_batch_size)
    if train_loader is None or val_loader is None:
        return None

    print(f"🔬 Training with {num_epochs} epochs, sample_rate={sample_rate} Hz, batch_size={current_batch_size}...")

    # Instantiate CompleteSincGAT_UNet with correct sample_rate
    model = CompleteSincGAT_UNet(
        sample_rate=sample_rate,  # CRITICAL: 10001 Hz (10001 samples = 1 second)
        sinc_min_low_hz=80,       # Based on kernel size constraints
        sinc_min_band_hz=10,      # Minimum bandwidth parameter
        # Other parameters use defaults
    ).to(device)
    
    print(f"Model: CompleteSincGAT_UNet, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Default Loss: L1Loss
    criterion = nn.L1Loss().to(device)
    print("Loss Function: nn.L1Loss() (MAE)")

    best_mape, history = train_validate_model_with_checkpoints(
        "Test_SincGAT_UNet_L1Loss", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape,
        checkpoint_freq=5
    )

    print(f"\\n✅ SincGAT_UNet with L1 Loss test completed!")
    print(f"🎯 Best Validation MAPE: {best_mape:.4f}%")
    return {'SincGAT_L1_MAPE': best_mape, 'history': history}


def test_sincgat_unet_champion_loss(num_epochs=45, sample_rate=10001, batch_size_override=None):
    print("👑 Testing CompleteSincGAT_UNet with CHAMPION Hybrid Loss...")

    # Configure A100 stability
    if 'cuda' in str(device):
        configure_a100_stability(disable_tf32=True, verbose=True)

    # Setup data loaders
    current_batch_size = batch_size_override if batch_size_override else 4
    train_loader, val_loader = setup_phase2_data_loaders(batch_size=current_batch_size)
    if train_loader is None or val_loader is None:
        return None

    print(f"🔬 Training with {num_epochs} epochs, sample_rate={sample_rate} Hz, batch_size={current_batch_size}...")

    # Instantiate CompleteSincGAT_UNet with correct sample_rate
    model = CompleteSincGAT_UNet(
        sample_rate=sample_rate,  # CRITICAL: 10001 Hz (10001 samples = 1 second)
        sinc_min_low_hz=80,       # Based on kernel size constraints
        sinc_min_band_hz=10,      # Minimum bandwidth parameter
        # Other parameters use defaults
    ).to(device)
    
    print(f"Model: CompleteSincGAT_UNet, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Champion Hybrid Loss (from your framework)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=False, # Champion uses fixed weights
        logmae_momentum=0,            # For FixedCLogSpaceMAE behavior
        initial_c_logmae=0.1,         # Champion c value
        fixed_weights_list=[1.0, 0.12, 0.007] # Champion weights
    ).to(device)
    
    print("Loss Function: Champion RefinedLogSpaceMAEHybridLoss")
    print(f"   Weights: {criterion.fixed_weights.cpu().numpy()}")

    best_mape, history = train_validate_model_with_checkpoints(
        "Test_SincGAT_UNet_ChampionLoss", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape,
        checkpoint_freq=5
    )

    print(f"\\n✅ SincGAT_UNet with Champion Loss test completed!")
    print(f"🎯 Best Validation MAPE: {best_mape:.4f}%")
    return {'SincGAT_Champion_MAPE': best_mape, 'history': history}


# 4. Run the experiments
ACTUAL_SAMPLE_RATE = 10001  # 10001 samples = 1 second
BATCH_SIZE_SincGAT = 4      # Start with smaller batch size due to model size

# Run Test 1: SincGAT_UNet + L1 Loss first
results_sincgat_l1 = test_sincgat_unet_default_loss(
    num_epochs=25, 
    sample_rate=ACTUAL_SAMPLE_RATE, 
    batch_size_override=BATCH_SIZE_SincGAT
)

# Run Test 2: SincGAT_UNet + Champion Hybrid Loss
results_sincgat_champion = test_sincgat_unet_champion_loss(
    num_epochs=45, 
    sample_rate=ACTUAL_SAMPLE_RATE, 
    batch_size_override=BATCH_SIZE_SincGAT
)
"""
    
    print(code_example)
    print("\n" + "="*70)
    print("END OF EXAMPLE CODE")
    print("="*70)
    print("\nCopy the example code above into your Colab notebook and adapt as needed.")

if __name__ == "__main__":
    # Run the test
    success, model = test_sincgat_init_and_inference()
    
    if success:
        print("\n🎉 Test completed successfully!")
        
        # Generate example code for notebook integration
        generate_notebook_code_example()
        
        print("\n✅ Key Points to Remember:")
        print("  1. Always use sample_rate=10001 (based on 10001 samples = 1 second)")
        print("  2. Use sinc_min_low_hz=80 (based on kernel size constraints)")
        print("  3. Use sinc_min_band_hz=10 (appropriate bandwidth for seismic data)")
        print("  4. Start with smaller batch size (4) due to larger model size")
        print("  5. Ensure torch-geometric is installed in your environment")
    else:
        print("\n❌ Test failed!") 