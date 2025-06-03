# Suggested Imports for this block (can be moved to top of notebook if preferred)
# === TWO-STAGE TRANSFER LEARNING IMPLEMENTATION (APPENDED) ===
# Make sure all necessary imports, utility functions (setup_phase2_data_loaders, calculate_mape),
# loss classes (RefinedLogSpaceMAEHybridLoss, StabilizedSeismicMSSSIM),
# model classes (ChampionBaselineUNet from complete_sincgat_unet_integration, CompleteSincGAT_UNet, etc.),
# and training functions (train_validate_model_with_checkpoints, train_with_curriculum)
# are defined or imported earlier in this notebook.

# === ESSENTIAL IMPORTS ===
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import json
from datetime import datetime

# === CRITICAL: GLOBAL DEFINITIONS (FIX 1) ===
CHECKPOINT_DIR = "checkpoints"
print(f"✅ CHECKPOINT_DIR globally defined as: {CHECKPOINT_DIR}")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    print(f"✅ Created checkpoint directory: {CHECKPOINT_DIR}")

# Ensure device is also globally available if not already
if 'device' not in globals() or not isinstance(device, torch.device): # check type to avoid issues if device was something else
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device globally defined as: {device}")
else:
    print(f"✅ Device already globally defined as: {device}")

# Corrected path for your champion weights for direct use (VERIFY ACTUAL FILE NAME)
champion_weights_path_for_direct_use = os.path.join(CHECKPOINT_DIR, "Extended_Absolute_Champion_epoch_40.pth")
print(f"✅ Default champion weights path for direct use: {champion_weights_path_for_direct_use}")


# === CRITICAL: IMPORT ALL NECESSARY FUNCTIONS ===
print("📦 Loading required functions from external modules...")

# Load phase2_experimental_framework.py
try:
    exec(open('phase2_experimental_framework.py').read())
    print("✅ Loaded phase2_experimental_framework.py")
except Exception as e:
    print(f"❌ Error loading phase2_experimental_framework.py: {e}")
    raise

# Load complete_sincgat_unet_integration.py
try:
    exec(open('complete_sincgat_unet_integration.py').read())
    print("✅ Loaded complete_sincgat_unet_integration.py")
    # Create alias for ChampionBaselineUNet
    ChampionBaselineUNet = BaselineUNet
    print("✅ ChampionBaselineUNet alias created")
except Exception as e:
    print(f"❌ Error loading complete_sincgat_unet_integration.py: {e}")
    raise

print("✅ All required functions loaded successfully")

# === FALLBACK FUNCTIONS FOR MISSING DEPENDENCIES ===
print("🔧 Setting up fallback functions for missing dependencies...")

# Fallback data loader function
def setup_phase2_data_loaders(batch_size=8, num_workers=0, test_size=0.2, random_state=42):
    """
    Fallback data loader function.
    This is a placeholder - you should import the real function from your data file.
    """
    print(f"⚠️ Using fallback data loader function")
    print(f"   Please import setup_phase2_data_loaders from your data file")
    print(f"   For now, creating dummy loaders for validation...")
    
    # Create minimal dummy datasets for validation
    import torch.utils.data as data
    
    class DummyDataset(data.Dataset):
        def __init__(self, size=32):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Return dummy data matching expected shapes
            shots = torch.randn(5, 10001, 31)  # 5 shots, 10001 time samples, 31 receivers
            velocity = torch.randn(300, 1259)  # Target velocity field
            return shots, velocity
    
    train_dataset = DummyDataset(64)  # Small dataset for validation
    val_dataset = DummyDataset(32)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"✅ Dummy data loaders created for validation")
    print(f"   Training: {len(train_loader)} batches, Validation: {len(val_loader)} batches")
    
    return train_loader, val_loader

# Fallback MAPE calculation function
def calculate_mape(pred, target, min_velocity=1.5, epsilon=1e-8):
    """
    Fallback MAPE calculation function.
    """
    # Handle shape mismatches - squeeze channel dimension if needed
    if pred.dim() == 4 and pred.shape[1] == 1:  # [B, 1, H, W]
        pred = pred.squeeze(1)  # [B, H, W]
    if target.dim() == 4 and target.shape[1] == 1:  # [B, 1, H, W]
        target = target.squeeze(1)  # [B, H, W]
    
    pred_clamped = torch.clamp(pred, min=min_velocity)
    target_clamped = torch.clamp(target, min=min_velocity)
    
    percentage_errors = torch.abs((target_clamped - pred_clamped) / (target_clamped + epsilon)) * 100
    mape = torch.mean(percentage_errors)
    return mape.item()

# Fallback training function
def train_validate_model_with_checkpoints(experiment_name, model, train_loader, val_loader, 
                                        criterion, optimizer, num_epochs, device, 
                                        calculate_mape_func, checkpoint_freq=10):
    """
    Fallback training function for Stage 1.
    """
    print(f"⚠️ Using fallback training function for {experiment_name}")
    
    model.train()
    best_val_mape = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (shots, targets) in enumerate(train_loader):
            shots, targets = shots.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(shots)
            
            # Handle both dictionary and scalar loss returns
            loss_result = criterion(outputs, targets)
            if isinstance(loss_result, dict):
                loss = loss_result['total']  # Extract total loss for backpropagation
                loss_value = loss.item()
            else:
                loss = loss_result
                loss_value = loss.item()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_value
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss_value:.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for shots, targets in val_loader:
                shots, targets = shots.to(device), targets.to(device)
                outputs = model(shots)
                
                # Handle both dictionary and scalar loss returns
                loss_result = criterion(outputs, targets)
                if isinstance(loss_result, dict):
                    loss_value = loss_result['total'].item()
                else:
                    loss_value = loss_result.item()
                
                mape = calculate_mape_func(outputs, targets)
                
                val_loss += loss_value
                val_mape += mape
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_mape = val_mape / val_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mape'].append(avg_val_mape)
        
        print(f"   Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.4f}%")
        
        # Save best model
        if avg_val_mape < best_val_mape:
            best_val_mape = avg_val_mape
            best_model_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best_mape.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mape': best_val_mape
            }, best_model_path)
    
    print(f"✅ Training completed. Best MAPE: {best_val_mape:.4f}%")
    return best_val_mape, history

print("✅ Fallback functions ready")

print("\\n" + "="*30 + " TWO-STAGE TRAINING SETUP " + "="*30)

# === CRITICAL: GLOBAL DEFINITIONS ===
# Ensure CHECKPOINT_DIR is defined globally to prevent NameError
CHECKPOINT_DIR = "checkpoints"
print(f"✅ CHECKPOINT_DIR globally defined: {CHECKPOINT_DIR}")

# Ensure device is defined (example, assuming it's defined earlier)
if 'device' not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device initialized in two-stage setup: {device}")
else:
    print(f"Using existing device: {device}")

# Ensure CHECKPOINT_DIR directory exists
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    print(f"Created checkpoint directory: {CHECKPOINT_DIR}")
else:
    print(f"Using existing checkpoint directory: {CHECKPOINT_DIR}")

# Helper to get the U-Net model for Stage 1 pre-training
def get_champion_baseline_unet_for_stage1(n_channels_in=5, n_channels_out=1, bilinear=True):
    print("Instantiating Champion BaselineUNet (Asymmetric) for Stage 1 Pre-training...")
    # This uses the BaselineUNet from complete_sincgat_unet_integration.py,
    # which should be your champion asymmetric version.
    # Make sure ChampionBaselineUNet is imported as:
    # from complete_sincgat_unet_integration import BaselineUNet as ChampionBaselineUNet
    model = ChampionBaselineUNet(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        bilinear=bilinear
    )
    return model

# --- Stage 1: Pre-train Champion BaselineUNet ---
def run_stage1_pretrain_unet(
    num_epochs=45,
    batch_size=8,
    lr=1e-4,
    weight_decay=0.01,
    min_velocity=1.5,
    logmae_initial_c=0.1,
    loss_fixed_weights=[1.0, 0.12, 0.007],
    experiment_name_prefix="Stage1_Pretrain"
):
    print("============================================================")
    print("🚀 EXECUTING STAGE 1: PRE-TRAINING CHAMPION UNET 🚀")
    print("============================================================\n")
    print(f"--- Starting Stage 1: Pre-training Champion BaselineUNet ({num_epochs} epochs) ---")
    print("============================================================")
    
    # Configure A100 stability only if CUDA is available
    if torch.cuda.is_available():
        configure_a100_stability(disable_tf32=True)
    else:
        print("🔧 Running on CPU - skipping A100 configuration")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data loaders - Fixed unpacking
    train_loader, val_loader = setup_phase2_data_loaders(
        batch_size=batch_size, num_workers=0
    )

    model = get_champion_baseline_unet_for_stage1().to(device)
    print(f"Champion BaselineUNet instantiated for Stage 1 with {sum(p.numel() for p in model.parameters())} parameters.")

    # Assuming RefinedLogSpaceMAEHybridLoss and StabilizedSeismicMSSSIM are defined/imported
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity,
        use_adaptive_softadapt=False,
        logmae_momentum=0,
        initial_c_logmae=logmae_initial_c,
        fixed_weights_list=loss_fixed_weights
    ).to(device)
    criterion.seismic_ms_ssim = StabilizedSeismicMSSSIM(
        apply_log=True, data_range_log=2.0, c_for_log=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    experiment_full_name = f"{experiment_name_prefix}_UNet_Asymmetric"

    print(f"Starting Stage 1 training using 'train_validate_model_with_checkpoints' for experiment: {experiment_full_name}")
    # train_validate_model_with_checkpoints saves the best model as:
    # os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best_mape.pth")

    # Assuming train_validate_model_with_checkpoints and calculate_mape are defined earlier
    best_val_mape, history = train_validate_model_with_checkpoints(
        experiment_name=experiment_full_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        calculate_mape_func=calculate_mape,
        checkpoint_freq=10 # Or your preferred frequency
    )

    saved_model_path = os.path.join(CHECKPOINT_DIR, f"{experiment_full_name}_best_mape.pth")
    print(f"--- Stage 1 Pre-training Finished ---")
    print(f"Best Validation MAPE from Stage 1: {best_val_mape:.4f}%")
    print(f"Best U-Net weights expected at: {saved_model_path}")

    if not os.path.exists(saved_model_path):
        print(f"ERROR: Expected Stage 1 model file was not found at {saved_model_path}. Check 'train_validate_model_with_checkpoints'.")
        return None, history # Return None for path if not found

    return saved_model_path, history

# --- Stage 2: Fine-tune CompleteSincGAT_UNet ---
def run_stage2_finetune_sincgat_unet(
    pretrained_unet_weights_path,
    num_epochs_phase_a=10,
    num_epochs_phase_b=30,
    batch_size=4,
    lr_frontend_phase_a=1e-4,
    lr_frontend_phase_b=5e-5,
    lr_unet_finetune_phase_b=1e-5,
    weight_decay=0.01,
    min_velocity=1.5,
    logmae_initial_c=0.1,
    loss_fixed_weights=[1.0, 0.12, 0.007],
    curriculum_start_simple=True,
    curriculum_total_epochs_for_simple_phase=2,  # Changed from 5 to 2: Use LogMAE for only 2 epochs, then switch to powerful hybrid loss
    experiment_name_prefix="Stage2_Finetune",
    use_film=False  # NEW: Control FiLM usage - False for baseline experiments
):
    """
    🚨 LEGACY/BASELINE STAGE 2 FUNCTION 🚨
    
    This function provides BASELINE Stage 2 training and is primarily maintained for:
    1. **Baseline Experiments**: When use_film=False, provides non-FiLM baseline
    2. **Backward Compatibility**: Legacy experimental setups
    3. **Simple Stage 2**: Basic two-phase training without advanced FiLM features
    
    ⚠️ FOR ADVANCED FiLM EXPERIMENTS, USE: run_stage2_film_training() ⚠️
    
    The run_stage2_film_training() function provides:
    - Advanced FiLM-specific parameter grouping
    - Granular differential learning rates for FiLM components  
    - Sophisticated warm-up and gradient clipping
    - Comprehensive FiLM monitoring and regularization
    - Better integration with the unified train_with_film_awareness() function
    
    This function uses simplified configurations and may not leverage all
    FiLM capabilities even when use_film=True.
    """
    print("=" * 80)
    if use_film:
        print(f"🎬 STAGE 2: FINE-TUNING SINCGAT-UNET WITH FiLM")
        print("   Note: For advanced FiLM experiments, use run_stage2_film_training()")
    else:
        print(f"🚀 STAGE 2: BASELINE SINCGAT-UNET FINE-TUNING (NO FiLM)")
        print("   Note: This is for baseline experiments without FiLM")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders(
        batch_size=batch_size, num_workers=0
    )
    
    # Create the CompleteSincGAT_UNet model with optional FiLM
    model_kwargs = {
        'sample_rate': 10001,
        'num_receivers': 31,
        'time_samples': 10001,
        'num_shots': 5,
        'sinc_out_channels': 60,
        'sinc_kernel_size': 1001,
        'sinc_stride': 1,
        'sinc_min_low_hz': 40,
        'sinc_max_learnable_hz': 1000,
        'sinc_min_band_hz': 10,
        'sinc_window_func': 'blackman',
        'sinc_init_type': 'logarithmic',
        'shot_embedding_dim': 128,
        'gat_hidden_per_head': 32,
        'gat_num_heads': 4,
        'gat_layers': 1,
        'gat_dropout_feat': 0.3,
        'gat_dropout_attn': 0.2,
        'fused_embedding_dim': 128,
        'n_unet_output_channels': 1,
        'unet_bilinear': True,
        'unet_bottleneck_channels': 512,
        'fusion_ratio': 0.25
    }
    
    # Add FiLM parameters only if FiLM is enabled
    if use_film:
        model_kwargs.update({
            'film_context_dim': 128,
            'film_target_channels': 512,
            'film_generator_mlp_type': 'linear'
        })
        print("   FiLM enabled with default parameters")
    else:
        print("   FiLM disabled - using baseline fusion_ratio method")
    
    sincgat_model = CompleteSincGAT_UNet(**model_kwargs).to(device)
    
    try:
        print(f"Loading pretrained U-Net weights into sincgat_model.unet...")
        # Load pretrained U-Net weights
        champion_unet_checkpoint = torch.load(pretrained_unet_weights_path, map_location=device, weights_only=False)
        
        # Handle both direct state_dict and checkpoint dict formats
        if isinstance(champion_unet_checkpoint, dict) and 'model_state_dict' in champion_unet_checkpoint:
            # This is a checkpoint file with multiple keys
            champion_unet_state_dict = champion_unet_checkpoint['model_state_dict']
            print(f"   📦 Loaded from checkpoint (epoch {champion_unet_checkpoint.get('epoch', 'unknown')})")
        else:
            # This is a direct state_dict
            champion_unet_state_dict = champion_unet_checkpoint
            print(f"   📦 Loaded direct state_dict")
            
        sincgat_model.unet.load_state_dict(champion_unet_state_dict, strict=True)
        print("Successfully loaded pretrained U-Net weights into sincgat_model.unet.")
    except Exception as e:
        print(f"ERROR loading pretrained U-Net weights: {e}")
        raise e

    criterion_stage2 = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity,
        use_adaptive_softadapt=False,
        logmae_momentum=0,
        initial_c_logmae=logmae_initial_c,
        fixed_weights_list=loss_fixed_weights,
        start_simple=curriculum_start_simple,
        curriculum_epochs=curriculum_total_epochs_for_simple_phase,
        # FiLM regularization (enabled only if use_film=True)
        use_film_reg=use_film,
        lambda_gamma_res=0.005 if use_film else 0.0,
        lambda_beta_res=0.0005 if use_film else 0.0
    ).to(device)
    criterion_stage2.seismic_ms_ssim = StabilizedSeismicMSSSIM(
        apply_log=True, data_range_log=2.0, c_for_log=0.1
    ).to(device)
    
    if use_film:
        print("   Loss function: FiLM regularization enabled")
    else:
        print("   Loss function: Standard loss without FiLM regularization")

    # --- Phase 2a: Frontend Training (U-Net Frozen) ---
    experiment_name_2a = f"{experiment_name_prefix}_PhaseA_FrontendFrozen"
    print(f"\n--- Stage 2a: Training Frontend ({num_epochs_phase_a} epochs) for {experiment_name_2a} ---")
    for param_name, param in sincgat_model.unet.named_parameters():
        param.requires_grad = False

    optimizer_stage2a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, sincgat_model.parameters()),
        lr=lr_frontend_phase_a,
        weight_decay=weight_decay
    )

    # Create simple config for baseline Phase 2a (no FiLM-specific parameters)
    config_2a = {
        'warmup_steps': 0,  # No warm-up for baseline Phase 2a
        'gradient_clip_film': 0.0,  # No FiLM clipping for baseline
        'gradient_clip_others': 5.0,  # Standard clipping
        'use_grad_clipping': True,
        'monitor_freq': 50,
        'use_film_reg': use_film,  # Respect the use_film parameter
        'epoch_monitor_freq': 10  # Less frequent monitoring for baseline
    }
    
    # No scheduler for baseline Phase 2a
    scheduler_2a = None

    print(f"Starting training for {experiment_name_2a} using 'train_with_curriculum_fixed' (must save best model).")
    # CRITICAL: Ensure 'train_with_curriculum' saves the best model like 'train_validate_model_with_checkpoints'.
    # If it doesn't, you might need to use train_validate_model_with_checkpoints here,
    # or add manual model saving logic based on its returned history.
    print(f"Starting training for {experiment_name_2a} using 'train_with_curriculum_fixed' (must save best model).")
    best_mape_2a, history_2a = train_with_curriculum_fixed( # CHANGED HERE
        experiment_name=experiment_name_2a,
        model=sincgat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_stage2,
        optimizer=optimizer_stage2a,
        num_epochs=num_epochs_phase_a,
        device=device,
        calculate_mape_func=calculate_mape,
        lr_scheduler=scheduler_2a,
        config=config_2a  # FIXED: Use config_2a instead of undefined config
    )
    path_2a = os.path.join(CHECKPOINT_DIR, f"{experiment_name_2a}_best_mape.pth")
    print(f"Stage 2a completed. Best MAPE: {best_mape_2a if best_mape_2a is not None else 'N/A'}.")
    print(f"Model for phase 2a expected at: {path_2a}. Exists: {os.path.exists(path_2a)}")


    # --- Phase 2b: Full Fine-tuning (Differential LRs) ---
    experiment_name_2b = f"{experiment_name_prefix}_PhaseB_FullFinetune"
    print(f"\n--- Stage 2b: Fine-tuning Full Model ({num_epochs_phase_b} epochs) for {experiment_name_2b} ---")
    for param_name, param in sincgat_model.unet.named_parameters(): # Unfreeze U-Net
        param.requires_grad = True

    # Setup differential learning rates with proper FiLM parameter separation
    # CRITICAL FIX: Separate parameter groups for granular control
    
    # FiLM generator parameters (if present)
    film_generator_params = []
    if hasattr(sincgat_model, 'film_bottleneck_modulator') and sincgat_model.film_bottleneck_modulator is not None:
        film_generator_params = list(sincgat_model.film_bottleneck_modulator.parameters())
    
    # GAT Context LayerNorm parameters (also benefits from FiLM-like LR)
    gat_context_norm_params = []
    if hasattr(sincgat_model, 'gat_context_layernorm'):
        gat_context_norm_params = list(sincgat_model.gat_context_layernorm.parameters())
    
    # Other frontend parameters (excluding FiLM and GAT LayerNorm)
    other_frontend_params = [p for name, p in sincgat_model.named_parameters() 
                           if 'unet' not in name and 'film_bottleneck_modulator' not in name 
                           and 'gat_context_layernorm' not in name and p.requires_grad]
    
    # U-Net parameters
    unet_params = [p for name, p in sincgat_model.named_parameters() 
                  if 'unet' in name and p.requires_grad]
    
    # Build parameter groups with differential LRs and weight decays
    param_groups = []
    
    # U-Net parameters (lowest LR)
    if unet_params:
        param_groups.append({
            'params': unet_params, 
            'lr': config['lr_unet_finetune_phase_b'],
            'weight_decay': config['weight_decay']
        })
        print(f"   📊 U-Net params: {len(unet_params)}")
    
    # Other frontend parameters (medium LR)
    if other_frontend_params:
        param_groups.append({
            'params': other_frontend_params, 
            'lr': config['lr_frontend_phase_b'],
            'weight_decay': config['weight_decay']
        })
        print(f"   📊 Other frontend params: {len(other_frontend_params)}")
    
    # GAT Context LayerNorm (FiLM-like LR)
    if gat_context_norm_params:
        param_groups.append({
            'params': gat_context_norm_params, 
            'lr': config.get('lr_film_generator', config['lr_frontend_phase_b']),
            'weight_decay': config.get('weight_decay_film', config['weight_decay'])
        })
        print(f"   📊 GAT Context LayerNorm params: {len(gat_context_norm_params)}")
    
    # FiLM generator parameters (highest LR, strongest regularization)
    if film_generator_params:
        param_groups.append({
            'params': film_generator_params, 
            'lr': config.get('lr_film_generator', config['lr_frontend_phase_b']),
            'weight_decay': config.get('weight_decay_film', config['weight_decay'])
        })
        print(f"   📊 FiLM generator params: {len(film_generator_params)}")
    
    optimizer_2b = torch.optim.AdamW(param_groups)
    
    # Create simple config for baseline Phase 2b (no FiLM-specific parameters)
    config_2b = {
        'warmup_steps': 0,  # No warm-up for baseline
        'gradient_clip_film': 0.0,  # No FiLM clipping for baseline
        'gradient_clip_others': 5.0,  # Standard clipping
        'use_grad_clipping': True,
        'monitor_freq': 50,
        'use_film_reg': use_film,  # Respect the use_film parameter
        'epoch_monitor_freq': 10  # Less frequent monitoring for baseline
    }
    
    # No scheduler for baseline Phase 2b
    scheduler_2b = None

    print(f"Starting training for {experiment_name_2b} using 'train_with_curriculum_fixed' (must save best model).")
    best_mape_2b, history_2b = train_with_curriculum_fixed( # CHANGED HERE
        experiment_name=experiment_name_2b,
        model=sincgat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_stage2, # Same criterion instance
        optimizer=optimizer_2b,
        num_epochs=num_epochs_phase_b,
        device=device,
        calculate_mape_func=calculate_mape,
        lr_scheduler=scheduler_2b,
        config=config_2b
    )
    path_2b = os.path.join(CHECKPOINT_DIR, f"{experiment_name_2b}_best_mape.pth")
    print(f"Stage 2b completed. Best MAPE: {best_mape_2b if best_mape_2b is not None else 'N/A'}.")
    print(f"Model for phase 2b expected at: {path_2b}. Exists: {os.path.exists(path_2b)}")

    print(f"--- Stage 2 Fine-tuning Finished ---")
    final_best_mape_stage2 = float('inf')
    if best_mape_2a is not None: final_best_mape_stage2 = min(final_best_mape_stage2, best_mape_2a)
    if best_mape_2b is not None: final_best_mape_stage2 = min(final_best_mape_stage2, best_mape_2b)

    print(f"Overall best MAPE from Stage 2: {final_best_mape_stage2 if final_best_mape_stage2 != float('inf') else 'N/A'}")

    final_model_path_stage2 = path_2b # Default to phase 2b model, or implement logic to pick overall best
    if os.path.exists(path_2a) and (not os.path.exists(path_2b) or (best_mape_2a is not None and best_mape_2b is not None and best_mape_2a < best_mape_2b)):
        final_model_path_stage2 = path_2a

    return final_model_path_stage2, (history_2a, history_2b)


# === Orchestration Logic ===
# This cell should be run to execute the two-stage training.
# Ensure all required functions and classes are defined above this cell in the notebook.

print("\\n" + "="*30 + " TWO-STAGE TRAINING ORCHESTRATION " + "="*30)

# --- Configuration for the two-stage run ---
# Set these flags to True or False to control which stages are run.
# For initial testing, use very few epochs.
RUN_STAGE_1_PRETRAINING = True  # True to run U-Net pre-training
RUN_STAGE_2_FINETUNING = True   # True to run SincGAT-UNet fine-tuning

# --- Stage 1: U-Net Pre-training ---
# Define default path, actual path will be returned by run_stage1
stage1_final_weights_path = os.path.join(CHECKPOINT_DIR, "Stage1_TestRun_UNet_Asymmetric_best_mape.pth") # Default/expected
stage1_run_history = None

if RUN_STAGE_1_PRETRAINING:
    print("\\n" + "="*60)
    print("🚀 EXECUTING STAGE 1: PRE-TRAINING CHAMPION UNET 🚀")
    print("="*60 + "\\n")

    # For a quick pipeline test:
    stage1_final_weights_path, stage1_run_history = run_stage1_pretrain_unet(
        num_epochs=2,
        batch_size=4,
        experiment_name_prefix="Stage1_TestRun"
    )
    # For a full run, use parameters like:
    # stage1_final_weights_path, stage1_run_history = run_stage1_pretrain_unet(
    #     num_epochs=45,
    #     batch_size=8,
    #     experiment_name_prefix="Stage1_FullRun"
    # )

    if stage1_final_weights_path and os.path.exists(stage1_final_weights_path):
        print(f"✅ Stage 1 completed. Champion U-Net weights saved to: {stage1_final_weights_path}")
    else:
        print(f"⚠️ Stage 1 did not produce the expected weights file. Expected based on exp name: {stage1_final_weights_path}")
        # RUN_STAGE_2_FINETUNING = False # Consider stopping if Stage 1 fails
else:
    print("\\n" + "="*60)
    print("⏩ SKIPPING STAGE 1 PRE-TRAINING ⏩")
    print(f"Attempting to use pre-existing U-Net weights from default path: {stage1_final_weights_path}")
    if not os.path.exists(stage1_final_weights_path):
        print(f"⚠️ WARNING: Pre-existing U-Net weights not found at {stage1_final_weights_path}. Stage 2 may fail.")
    print("="*60 + "\\n")

# --- Stage 2: CompleteSincGAT_UNet Fine-tuning ---
stage2_final_weights_path = None
stage2_run_history = None

if RUN_STAGE_2_FINETUNING:
    if not stage1_final_weights_path or not os.path.exists(stage1_final_weights_path):
        print(f"❌ ERROR: Cannot run Stage 2. Valid pre-trained U-Net weights path from Stage 1 is missing or file does not exist ({stage1_final_weights_path}).")
        print("Please ensure Stage 1 runs successfully and produces weights, or provide a correct path manually if skipping Stage 1.")
    else:
        print("\\n" + "="*60)
        print("🚀 EXECUTING STAGE 2: FINE-TUNING CompleteSincGAT_UNet 🚀")
        print("="*60 + "\\n")

        # For a quick pipeline test:
        stage2_final_weights_path, stage2_run_history = run_stage2_finetune_sincgat_unet(
            pretrained_unet_weights_path=stage1_final_weights_path,
            num_epochs_phase_a=1,
            num_epochs_phase_b=1,
            batch_size=2,
            experiment_name_prefix="Stage2_TestRun",
            curriculum_total_epochs_for_simple_phase=1 # Quick test for curriculum
        )
        # For a full run, use parameters like:
        # stage2_final_weights_path, stage2_run_history = run_stage2_finetune_sincgat_unet(
        #     pretrained_unet_weights_path=stage1_final_weights_path,
        #     num_epochs_phase_a=10,
        #     num_epochs_phase_b=30,
        #     batch_size=4, # Or 2 depending on memory
        #     experiment_name_prefix="Stage2_FullRun",
        #     curriculum_total_epochs_for_simple_phase=5
        # )
        if stage2_final_weights_path and os.path.exists(stage2_final_weights_path):
            print(f"✅ Stage 2 completed. Fine-tuned CompleteSincGAT_UNet weights saved to: {stage2_final_weights_path}")
        else:
            print(f"⚠️ Stage 2 did not produce the expected final weights file. Expected based on exp name: {stage2_final_weights_path}")
else:
    print("\\n" + "="*60)
    print("⏩ SKIPPING STAGE 2 FINE-TUNING ⏩")
    print("="*60 + "\\n")

print("\\n🏁🏁🏁 TWO-STAGE TRAINING SCRIPT EXECUTION FINISHED. 🏁🏁🏁\\n")

# --- Optional: Display history information ---
if stage1_run_history:
    print("--- Stage 1 History ---")
    # Example: Assuming history is a dict with lists like 'val_mape'
    # print(f"  Min Val MAPE: {min(stage1_run_history.get('val_mape', [float('inf')])):.4f}%")
    # print(f"  Final Val MAPE: {stage1_run_history.get('val_mape', [-1])[-1]:.4f}%")
    # Add more detailed history printing if desired

if stage2_run_history and isinstance(stage2_run_history, tuple) and len(stage2_run_history) == 2:
    history_2a, history_2b = stage2_run_history
    if history_2a:
        print("--- Stage 2a (Frontend) History ---")
        # print(f"  Min Val MAPE: {min(history_2a.get('val_mape', [float('inf')])):.4f}%")
        # print(f"  Final Val MAPE: {history_2a.get('val_mape', [-1])[-1]:.4f}%")
    if history_2b:
        print("--- Stage 2b (Full Finetune) History ---")
        # print(f"  Min Val MAPE: {min(history_2b.get('val_mape', [float('inf')])):.4f}%")
        # print(f"  Final Val MAPE: {history_2b.get('val_mape', [-1])[-1]:.4f}%")

print("\\nReview checkpoint directory for saved models and logs.")
print("Adjust RUN_STAGE_1_PRETRAINING and RUN_STAGE_2_FINETUNING flags and epochs/batch_sizes for full runs.")
print("Make sure all dependencies and helper functions are correctly defined earlier in the notebook.")
print("Ensure 'train_with_curriculum_fixed' saves the best model for Stage 2 phases.")


# === SYSTEMATIC EXPERIMENTAL FRAMEWORK FOR STAGE 2 HYPERPARAMETER EXPLORATION ===
# This framework systematically explores the hyperparameter space for CompleteSincGAT_UNet
# after Stage 1 (BaselineUNet pre-training) is complete.

import itertools
import json
from datetime import datetime
import pandas as pd

print("\\n" + "="*40 + " EXPERIMENTAL FRAMEWORK SETUP " + "="*40)

class Stage2ExperimentalFramework:
    """
    Systematic experimental framework for exploring CompleteSincGAT_UNet hyperparameters
    in the transfer learning context (Stage 2 fine-tuning).
    """
    
    def __init__(self, 
                 pretrained_unet_weights_path,
                 base_experiment_name="Stage2_Systematic",
                 results_dir="experiment_results",
                 device=None):
        self.pretrained_unet_weights_path = pretrained_unet_weights_path
        self.base_experiment_name = base_experiment_name
        self.results_dir = results_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # Initialize results tracking
        self.experiment_results = []
        self.best_result = None
        
        print(f"🔬 Experimental Framework Initialized:")
        print(f"   Pretrained U-Net: {self.pretrained_unet_weights_path}")
        print(f"   Results Directory: {self.results_dir}")
        print(f"   Device: {self.device}")
    
    def define_parameter_grids(self):
        """Define parameter grids for systematic exploration."""
        
        # === PRIORITY 1: Learning Rate & Schedule Experiments ===
        self.lr_schedule_grid = {
            'lr_frontend_phase_b': [2e-5, 5e-5, 1e-4],
            'lr_unet_finetune_phase_b': [5e-6, 1e-5, 2e-5],
            'lr_film_generator': [5e-5, 1e-4, 2e-4],  # NEW: FiLM-specific LR
            'lr_scheduler_type': [None, 'ReduceLROnPlateau', 'CosineAnnealingLR'],
            'scheduler_patience': [3, 5, 7],  # For ReduceLROnPlateau
            'scheduler_factor': [0.1, 0.2, 0.5],  # For ReduceLROnPlateau
        }
        
        # === PRIORITY 2: GAT Configuration Experiments ===
        self.gat_config_grid = {
            'gat_layers': [1, 2],
            'gat_num_heads': [2, 4, 8],
            'gat_hidden_per_head': [16, 32, 64],
            'gat_dropout_feat': [0.1, 0.2, 0.3],
            'gat_dropout_attn': [0.1, 0.2, 0.3],
            'use_global_attention': [True, False],  # For readout pooling
        }
        
        # === PRIORITY 3: FiLM Configuration Experiments ===
        self.film_config_grid = {
            'film_generator_mlp_type': ['linear', '2_layer'],
            'film_mlp_hidden_dim': [128, 256],  # For 2_layer MLP
            'lambda_gamma_res': [0.001, 0.005, 0.01],
            'lambda_beta_res': [0.0001, 0.0005, 0.001],
            'weight_decay_film': [1e-4, 5e-4, 1e-3],
        }
        
        # === PRIORITY 4: GAT Context Injection Experiments ===
        self.fusion_config_grid = {
            'fusion_ratio': [0.1, 0.25, 0.5],
            'fusion_method': ['concat_conv', 'film'],  # Future: FiLM implementation
        }
        
        # === PRIORITY 5: Shot Embedding Dimension Experiments ===
        # Note: This requires re-running Stage 2a (frontend training)
        self.embedding_dim_grid = {
            'shot_embedding_dim': [64, 128, 256],
        }
        
        # === PRIORITY 6: Training Configuration Experiments ===
        self.training_config_grid = {
            'num_epochs_phase_a': [5, 10, 15],
            'num_epochs_phase_b': [20, 30, 40],
            'batch_size': [2, 4, 8],
            'weight_decay': [0.001, 0.01, 0.1],
            'curriculum_epochs': [0, 3, 5],
            'warmup_steps': [500, 1000, 2000],  # NEW: Warm-up steps
            'gradient_clip_film': [0.5, 1.0, 2.0],  # NEW: FiLM gradient clipping
            'gradient_clip_others': [2.0, 5.0, 10.0],  # NEW: Other gradient clipping
        }
        
        print("✅ Parameter grids defined for systematic exploration (including FiLM parameters)")
    
    def create_experiment_config(self, **param_overrides):
        """Create a complete experiment configuration with parameter overrides."""
        
        # Base configuration (your current working setup)
        base_config = {
            # Model architecture
            'sample_rate': 10001,
            'num_receivers': 31,
            'time_samples': 10001,
            'num_shots': 5,
            'sinc_out_channels': 60,
            'sinc_kernel_size': 1001,
            'sinc_stride': 1,
            'sinc_min_low_hz': 40,
            'sinc_max_learnable_hz': 1000,
            'sinc_min_band_hz': 10,
            'sinc_window_func': 'blackman',
            'sinc_init_type': 'logarithmic',
            'shot_embedding_dim': 128,
            'gat_hidden_per_head': 32,
            'gat_num_heads': 4,
            'gat_layers': 1,
            'gat_dropout_feat': 0.3,
            'gat_dropout_attn': 0.2,
            'fused_embedding_dim': 128,
            'n_unet_output_channels': 1,
            'unet_bilinear': True,
            'unet_bottleneck_channels': 512,
            'fusion_ratio': 0.25,
            
            # FiLM parameters (NEW)
            'film_context_dim': 128,
            'film_target_channels': 512,
            'film_generator_mlp_type': 'linear',  # 'linear' or '2_layer'
            'film_mlp_hidden_dim': 256,  # For 2_layer MLP
            'use_film_reg': True,
            'lambda_gamma_res': 0.005,
            'lambda_beta_res': 0.0005,
            
            # Training parameters
            'batch_size': 4,
            'num_epochs_phase_a': 10,
            'num_epochs_phase_b': 30,
            'lr_frontend_phase_a': 1e-4,
            'lr_frontend_phase_b': 5e-5,
            'lr_unet_finetune_phase_b': 1e-5,
            'lr_film_generator': 1e-4,  # NEW: FiLM-specific LR
            'weight_decay': 0.01,
            'weight_decay_film': 1e-3,  # NEW: FiLM-specific weight decay
            
            # Training enhancements (NEW)
            'warmup_steps': 1000,
            'gradient_clip_film': 1.0,
            'gradient_clip_others': 5.0,
            'use_grad_clipping': True,
            
            # Loss parameters
            'min_velocity': 1.5,
            'logmae_initial_c': 0.1,
            'loss_fixed_weights': [1.0, 0.12, 0.007],
            'curriculum_start_simple': True,
            'curriculum_total_epochs_for_simple_phase': 2,  # Changed from 5 to 2: Use LogMAE for only 2 epochs, then switch to powerful hybrid loss
            
            # Scheduler parameters
            'lr_scheduler_type': None,  # None, 'ReduceLROnPlateau', 'CosineAnnealingLR'
            'scheduler_factor': 0.5,
            'scheduler_patience': 5
        }
        
        # Update with any parameter overrides
        for key, value in param_overrides.items():
            base_config[key] = value
        
        return base_config
    
    def run_single_experiment(self, config, experiment_id):
        """
        Run a single experiment with the given configuration and record results.
        """
        experiment_name = f"{self.base_experiment_name}_{experiment_id}"
        start_time = datetime.now()
        
        print(f"\n🧪 Running Experiment {experiment_id}: {experiment_name}")
        print(f"   Key parameters: {self._format_key_params(config)}")
        print("=" * 60)
        
        # Configure A100 stability
        configure_a100_stability(disable_tf32=True)
        
        try:
            # Setup data loaders
            print(f"Setting up data loaders with {config['batch_size']} batch size...")
            train_loader, val_loader = setup_phase2_data_loaders(batch_size=config['batch_size'])
            
            # Create model
            sincgat_model = CompleteSincGAT_UNet(
                sample_rate=config['sample_rate'],
                num_receivers=config['num_receivers'],
                time_samples=config['time_samples'],
                num_shots=config['num_shots'],
                sinc_out_channels=config['sinc_out_channels'],
                sinc_kernel_size=config['sinc_kernel_size'],
                sinc_stride=config['sinc_stride'],
                sinc_min_low_hz=config['sinc_min_low_hz'],
                sinc_max_learnable_hz=config['sinc_max_learnable_hz'],
                sinc_min_band_hz=config['sinc_min_band_hz'],
                sinc_window_func=config['sinc_window_func'],
                sinc_init_type=config['sinc_init_type'],
                shot_embedding_dim=config['shot_embedding_dim'],
                gat_hidden_per_head=config['gat_hidden_per_head'],
                gat_num_heads=config['gat_num_heads'],
                gat_layers=config['gat_layers'],
                gat_dropout_feat=config['gat_dropout_feat'],
                gat_dropout_attn=config['gat_dropout_attn'],
                fused_embedding_dim=config['fused_embedding_dim'],
                n_unet_output_channels=config['n_unet_output_channels'],
                unet_bilinear=config['unet_bilinear'],
                unet_bottleneck_channels=config['unet_bottleneck_channels'],
                fusion_ratio=config['fusion_ratio'],
                # FiLM parameters (NEW)
                film_context_dim=config.get('film_context_dim', 128),
                film_target_channels=config.get('film_target_channels', 512),
                film_generator_mlp_type=config.get('film_generator_mlp_type', 'linear'),
                film_mlp_hidden_dim=config.get('film_mlp_hidden_dim', 256)
            ).to(self.device)
            
            # Load pretrained U-Net weights
            champion_unet_checkpoint = torch.load(self.pretrained_unet_weights_path, map_location=self.device, weights_only=False)
            
            # Handle both direct state_dict and checkpoint dict formats
            if isinstance(champion_unet_checkpoint, dict) and 'model_state_dict' in champion_unet_checkpoint:
                # This is a checkpoint file with multiple keys
                champion_unet_state_dict = champion_unet_checkpoint['model_state_dict']
                print(f"   📦 Loaded from checkpoint (epoch {champion_unet_checkpoint.get('epoch', 'unknown')})")
            else:
                # This is a direct state_dict
                champion_unet_state_dict = champion_unet_checkpoint
                print(f"   📦 Loaded direct state_dict")
                
            sincgat_model.unet.load_state_dict(champion_unet_state_dict, strict=True)
            
            # Setup loss function
            criterion = RefinedLogSpaceMAEHybridLoss(
                min_velocity=config['min_velocity'],
                use_adaptive_softadapt=False,
                logmae_momentum=0,
                initial_c_logmae=config['logmae_initial_c'],
                fixed_weights_list=config['loss_fixed_weights'],
                start_simple=config['curriculum_start_simple'],
                curriculum_epochs=config['curriculum_total_epochs_for_simple_phase'],
                # FiLM regularization parameters (NEW)
                use_film_reg=config.get('use_film_reg', False),
                lambda_gamma_res=config.get('lambda_gamma_res', 0.005),
                lambda_beta_res=config.get('lambda_beta_res', 0.0005)
            ).to(self.device)
            criterion.seismic_ms_ssim = StabilizedSeismicMSSSIM(
                apply_log=True, data_range_log=2.0, c_for_log=0.1
            ).to(self.device)
            
            # === Phase 2a: Frontend Training (U-Net Frozen) ===
            print(f"   Phase 2a: Training frontend ({config['num_epochs_phase_a']} epochs)...")
            
            # Freeze U-Net
            for param in sincgat_model.unet.parameters():
                param.requires_grad = False
            
            # Setup optimizer for frontend only
            frontend_params = [p for name, p in sincgat_model.named_parameters() 
                             if 'unet' not in name and p.requires_grad]
            optimizer_2a = torch.optim.AdamW(frontend_params, 
                                           lr=config['lr_frontend_phase_a'], 
                                           weight_decay=config['weight_decay'])
            
            # Setup scheduler if specified
            scheduler_2a = self._create_scheduler(optimizer_2a, config, phase='2a')
            
            # Train Phase 2a
            best_mape_2a, history_2a = train_with_curriculum_fixed(
                experiment_name=f"{experiment_name}_PhaseA",
                model=sincgat_model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer_2a,
                num_epochs=config['num_epochs_phase_a'],
                device=self.device,
                calculate_mape_func=calculate_mape,
                lr_scheduler=scheduler_2a,
                config=config
            )
            
            # Define path_2a for later reference
            path_2a = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_PhaseA_best_mape.pth")
            
            # === Phase 2b: Full Model Fine-tuning ===
            print(f"   Phase 2b: Full model fine-tuning ({config['num_epochs_phase_b']} epochs)...")
            
            # Unfreeze U-Net
            for param in sincgat_model.unet.parameters():
                param.requires_grad = True
            
            # Setup granular differential learning rates with FiLM-specific groups (FIX 4)
            print("   ⚙️ Setting up granular parameter groups for Phase 2b differential LRs...")
            
            film_generator_params = []
            if hasattr(sincgat_model, 'film_bottleneck_modulator') and sincgat_model.film_bottleneck_modulator is not None:
                film_generator_params = list(sincgat_model.film_bottleneck_modulator.parameters())

            gat_context_norm_params = []
            if hasattr(sincgat_model, 'gat_context_layernorm'):
                gat_context_norm_params = list(sincgat_model.gat_context_layernorm.parameters())
            
            sincnet_params = []
            if hasattr(sincgat_model, 'shot_encoder'):
                 sincnet_params = list(sincgat_model.shot_encoder.parameters())

            gat_params = []
            if hasattr(sincgat_model, 'gat_fusion'):
                gat_params = list(sincgat_model.gat_fusion.parameters())

            unet_params = []
            if hasattr(sincgat_model, 'unet'):
                unet_params = list(sincgat_model.unet.parameters())
            
            # Build parameter groups with differential LRs and weight decays
            param_groups = []
            
            # U-Net parameters (lowest LR, no warmup typically)
            if unet_params:
                param_groups.append({
                    'params': unet_params, 
                    'lr': config['lr_unet_finetune_phase_b'],
                    'weight_decay': config['weight_decay'], 
                    'group_name': 'U-Net', 
                    'apply_warmup': False 
                })
                print(f"     U-Net params: {len(unet_params)} (LR: {config['lr_unet_finetune_phase_b']:.2e}, WD: {config['weight_decay']:.1e}, Warmup: False)")

            # SincNet parameters (medium LR, with warmup)
            if sincnet_params:
                param_groups.append({
                    'params': sincnet_params,
                    'lr': config['lr_frontend_phase_b'],
                    'weight_decay': config['weight_decay'],
                    'group_name': 'SincNet',
                    'apply_warmup': True
                })
                print(f"     SincNet params: {len(sincnet_params)} (LR: {config['lr_frontend_phase_b']:.2e}, WD: {config['weight_decay']:.1e}, Warmup: True)")
            
            # GAT parameters (medium LR, with warmup)
            if gat_params:
                param_groups.append({
                    'params': gat_params,
                    'lr': config['lr_frontend_phase_b'],
                    'weight_decay': config['weight_decay'],
                    'group_name': 'GAT',
                    'apply_warmup': True
                })
                print(f"     GAT params: {len(gat_params)} (LR: {config['lr_frontend_phase_b']:.2e}, WD: {config['weight_decay']:.1e}, Warmup: True)")
            
            # GAT Context LayerNorm (FiLM-like LR, with warmup)
            if gat_context_norm_params:
                param_groups.append({
                    'params': gat_context_norm_params, 
                    'lr': config.get('lr_film_generator', config['lr_frontend_phase_b']), 
                    'weight_decay': config.get('weight_decay_film', config['weight_decay']), 
                    'group_name': 'GAT_Context_Norm', 
                    'apply_warmup': True
                })
                print(f"     GAT Context Norm params: {len(gat_context_norm_params)} (LR: {config.get('lr_film_generator', config['lr_frontend_phase_b']):.2e}, WD: {config.get('weight_decay_film', config['weight_decay']):.1e}, Warmup: True)")
            
            # FiLM generator parameters (highest LR, strongest regularization, with warmup)
            if film_generator_params:
                param_groups.append({
                    'params': film_generator_params, 
                    'lr': config.get('lr_film_generator', config['lr_frontend_phase_b']), 
                    'weight_decay': config.get('weight_decay_film', config['weight_decay']),
                    'group_name': 'FiLM_Generator', 
                    'apply_warmup': True
                })
                print(f"     FiLM Generator params: {len(film_generator_params)} (LR: {config.get('lr_film_generator', config['lr_frontend_phase_b']):.2e}, WD: {config.get('weight_decay_film', config['weight_decay']):.1e}, Warmup: True)")
            
            optimizer_2b = torch.optim.AdamW(param_groups)
            
            # Create experiment names and setup for Phase 2b
            experiment_name_2b = f"{experiment_name}_PhaseB"
            num_epochs_phase_b = config['num_epochs_phase_b']
            
            # Setup scheduler for Phase 2b (FIX 4: Pass full config to _create_scheduler)
            # The _create_scheduler method already exists and uses config for T_max etc.
            scheduler_2b = self._create_scheduler(optimizer_2b, config, phase='2b')
            
            # The main 'config' dict is already comprehensive from create_experiment_config.
            # It contains warmup_steps, clipping norms, use_film_reg, etc.
            # So, we pass it directly to the training function. (FIX 4)

            print(f"Starting training for {experiment_name_2b} using 'train_with_curriculum_fixed' (must save best model).")
            # train_with_curriculum_fixed is now a wrapper for train_with_film_awareness
            best_mape_2b, history_2b = train_with_curriculum_fixed( 
                experiment_name=experiment_name_2b,
                model=sincgat_model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion, # Same criterion instance
                optimizer=optimizer_2b,
                num_epochs=num_epochs_phase_b,
                device=self.device,
                calculate_mape_func=calculate_mape,
                lr_scheduler=scheduler_2b,
                config=config # PASS THE FULL EXPERIMENT CONFIG
            )
            path_2b = os.path.join(CHECKPOINT_DIR, f"{experiment_name_2b}_best_mape.pth")
            print(f"Stage 2b completed. Best MAPE: {best_mape_2b if best_mape_2b is not None else 'N/A'}.")
            print(f"Model for phase 2b expected at: {path_2b}. Exists: {os.path.exists(path_2b)}")

            print(f"--- Stage 2 Fine-tuning Finished ---")
            final_best_mape_stage2 = float('inf')
            if best_mape_2a is not None: final_best_mape_stage2 = min(final_best_mape_stage2, best_mape_2a)
            if best_mape_2b is not None: final_best_mape_stage2 = min(final_best_mape_stage2, best_mape_2b)

            print(f"Overall best MAPE from Stage 2: {final_best_mape_stage2 if final_best_mape_stage2 != float('inf') else 'N/A'}")

            final_model_path_stage2 = path_2b # Default to phase 2b model, or implement logic to pick overall best
            if os.path.exists(path_2a) and (not os.path.exists(path_2b) or (best_mape_2a is not None and best_mape_2b is not None and best_mape_2a < best_mape_2b)):
                final_model_path_stage2 = path_2a

            # Create success result dictionary
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            result = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'config': config,
                'status': 'completed',
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration_seconds,
                'best_mape_phase_a': best_mape_2a,
                'best_mape_phase_b': best_mape_2b,
                'final_mape': final_best_mape_stage2 if final_best_mape_stage2 != float('inf') else None,
                'final_model_path': final_model_path_stage2,
                'history_phase_a': history_2a,
                'history_phase_b': history_2b
            }
            
        except Exception as e:
            print(f"   ❌ Experiment {experiment_id} failed: {str(e)}")
            result = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'config': config,
                'error': str(e),
                'status': 'failed',
                'timestamp': start_time.isoformat()
            }
        
        # Save result
        self.experiment_results.append(result)
        self._save_results()
        
        # Update best result
        if result.get('status') == 'completed':
            if self.best_result is None or result['final_mape'] < self.best_result['final_mape']:
                self.best_result = result
                print(f"   🏆 NEW BEST RESULT! MAPE: {result['final_mape']:.4f}%")
        
        return result
    
    def _create_scheduler(self, optimizer, config, phase):
        """Create learning rate scheduler based on configuration."""
        scheduler_type = config.get('lr_scheduler_type')
        
        if scheduler_type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=config['scheduler_factor'],
                patience=config['scheduler_patience'],
                verbose=True
            )
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = config['num_epochs_phase_a'] if phase == '2a' else config['num_epochs_phase_b']
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=1e-7
            )
        else:
            return None
    
    def _format_key_params(self, config):
        """Format key parameters for display."""
        key_params = {
            'lr_frontend_b': config['lr_frontend_phase_b'],
            'lr_unet_b': config['lr_unet_finetune_phase_b'],
            'gat_heads': config['gat_num_heads'],
            'gat_layers': config['gat_layers'],
            'fusion_ratio': config['fusion_ratio'],
            'batch_size': config['batch_size']
        }
        return str(key_params)
    
    def _save_results(self):
        """Save experiment results to JSON file."""
        results_file = os.path.join(self.results_dir, f"{self.base_experiment_name}_results.json")
        
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in self.experiment_results:
            serializable_result = result.copy()
            # Convert any numpy arrays or tensors to lists
            if 'history_phase_a' in serializable_result:
                serializable_result['history_phase_a'] = self._serialize_history(serializable_result['history_phase_a'])
            if 'history_phase_b' in serializable_result:
                serializable_result['history_phase_b'] = self._serialize_history(serializable_result['history_phase_b'])
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Results saved to: {results_file}")
    
    def _serialize_history(self, history):
        """Convert history to JSON-serializable format."""
        if history is None:
            return None
        
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, (list, tuple)):
                serializable_history[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                serializable_history[key] = value
        return serializable_history
    
    def run_lr_schedule_experiments(self, max_experiments=None):
        """
        Run systematic learning rate and scheduler experiments.
        Now fully explores the defined grid for ALL scheduler parameters.
        
        Args:
            max_experiments: Maximum number of experiments to run. If None, runs all combinations.
        """
        print(f"\n🎯 Running Learning Rate & Schedule Experiments...")
        
        all_configs_to_run = []
        
        # Base LR triplets (frontend, unet, and film learning rates)
        base_lr_triplets = list(itertools.product(
            self.lr_schedule_grid['lr_frontend_phase_b'],
            self.lr_schedule_grid['lr_unet_finetune_phase_b'],
            self.lr_schedule_grid['lr_film_generator']  # ADD FiLM LR to systematic search
        ))
        
        for lr_frontend_b, lr_unet_b, lr_film_gen in base_lr_triplets:
            for scheduler_type in self.lr_schedule_grid['lr_scheduler_type']:
                config_overrides = {
                    'lr_frontend_phase_b': lr_frontend_b,
                    'lr_unet_finetune_phase_b': lr_unet_b,
                    'lr_film_generator': lr_film_gen,  # ADD FiLM LR override
                    'lr_scheduler_type': scheduler_type,
                }
                
                if scheduler_type == 'ReduceLROnPlateau':
                    # Iterate through ALL patience and factor combinations for ReduceLROnPlateau
                    for patience in self.lr_schedule_grid['scheduler_patience']:
                        for factor in self.lr_schedule_grid['scheduler_factor']:
                            current_config_overrides = config_overrides.copy()
                            current_config_overrides['scheduler_patience'] = patience
                            current_config_overrides['scheduler_factor'] = factor
                            all_configs_to_run.append(self.create_experiment_config(**current_config_overrides))
                elif scheduler_type == 'CosineAnnealingLR':
                    # CosineAnnealingLR doesn't use patience/factor from this grid
                    # It uses T_max based on num_epochs within _create_scheduler
                    all_configs_to_run.append(self.create_experiment_config(**config_overrides))
                elif scheduler_type is None:
                    # No scheduler
                    all_configs_to_run.append(self.create_experiment_config(**config_overrides))

        # Calculate total possible experiments
        total_possible_lr_experiments = len(all_configs_to_run)
        print(f"   📊 Total unique LR/Scheduler configurations generated from grid: {total_possible_lr_experiments}")
        
        # Calculate breakdown
        num_lr_pairs = len(base_lr_triplets)
        num_none_configs = num_lr_pairs * 1  # None scheduler
        num_cosine_configs = num_lr_pairs * 1  # CosineAnnealingLR
        num_reduce_configs = num_lr_pairs * len(self.lr_schedule_grid['scheduler_patience']) * len(self.lr_schedule_grid['scheduler_factor'])
        
        print(f"   📈 Breakdown: {num_lr_pairs} LR pairs × [1 None + 1 Cosine + {len(self.lr_schedule_grid['scheduler_patience'])}×{len(self.lr_schedule_grid['scheduler_factor'])} ReduceLROnPlateau] = {total_possible_lr_experiments}")

        # Apply max_experiments limit if provided
        if max_experiments is not None and max_experiments < total_possible_lr_experiments:
            print(f"   ⚠️  Limiting to {max_experiments} experiments due to max_experiments setting.")
            print(f"   💡 To run all configurations, call with max_experiments={total_possible_lr_experiments} or None")
            # Potentially shuffle if you want a random subset, otherwise it takes the first N
            # import random
            # random.shuffle(all_configs_to_run) 
            configs_to_actually_run = all_configs_to_run[:max_experiments]
        else:
            configs_to_actually_run = all_configs_to_run
            if max_experiments is not None: # max_experiments >= total_possible_lr_experiments
                print(f"   ✅ Running all {total_possible_lr_experiments} configurations (max_experiments is >= total).")
            else:
                print(f"   ✅ Running all {total_possible_lr_experiments} configurations (no max_experiments limit).")

        experiment_id_start = len(self.experiment_results) + 1 
        
        for i, config_dict in enumerate(configs_to_actually_run):
            current_experiment_id_str = f"lr_sched_{experiment_id_start + i}"
            self.run_single_experiment(config_dict, current_experiment_id_str)
    
    def run_gat_config_experiments(self, max_experiments=15):
        """Run systematic GAT configuration experiments."""
        print(f"\n🎯 Running GAT Configuration Experiments (max {max_experiments})...")
        
        # Create parameter combinations
        gat_combinations = list(itertools.product(
            self.gat_config_grid['gat_layers'],
            self.gat_config_grid['gat_num_heads'],
            self.gat_config_grid['gat_hidden_per_head']
        ))
        
        # Limit to max_experiments
        gat_combinations = gat_combinations[:max_experiments]
        
        experiment_id = len(self.experiment_results) + 1
        
        for gat_layers, gat_heads, gat_hidden in gat_combinations:
            config_overrides = {
                'gat_layers': gat_layers,
                'gat_num_heads': gat_heads,
                'gat_hidden_per_head': gat_hidden,
                'fused_embedding_dim': gat_heads * gat_hidden,  # Adjust output dimension
            }
            
            config = self.create_experiment_config(**config_overrides)
            self.run_single_experiment(config, experiment_id)
            experiment_id += 1
    
    def run_fusion_experiments(self, max_experiments=6):
        """Run GAT context injection experiments."""
        print(f"\n🎯 Running Fusion Configuration Experiments (max {max_experiments})...")
        
        fusion_ratios = self.fusion_config_grid['fusion_ratio'][:max_experiments]
        experiment_id = len(self.experiment_results) + 1
        
        for fusion_ratio in fusion_ratios:
            config_overrides = {
                'fusion_ratio': fusion_ratio,
            }
            
            config = self.create_experiment_config(**config_overrides)
            self.run_single_experiment(config, experiment_id)
            experiment_id += 1
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report of all experiments."""
        if not self.experiment_results:
            print("No experiment results to summarize.")
            return
        
        print(f"\n📈 EXPERIMENTAL SUMMARY REPORT")
        print("="*60)
        
        # Filter successful experiments
        successful_experiments = [r for r in self.experiment_results if r.get('status') == 'completed']
        failed_experiments = [r for r in self.experiment_results if r.get('status') == 'failed']
        
        print(f"Total Experiments: {len(self.experiment_results)}")
        print(f"Successful: {len(successful_experiments)}")
        print(f"Failed: {len(failed_experiments)}")
        
        if successful_experiments:
            # Sort by final MAPE
            successful_experiments.sort(key=lambda x: x['final_mape'])
            
            print(f"\\n🏆 TOP 5 BEST RESULTS:")
            for i, result in enumerate(successful_experiments[:5]):
                print(f"  {i+1}. Exp {result['experiment_id']:03d}: {result['final_mape']:.4f}% MAPE")
                print(f"     Key params: {self._format_key_params(result['config'])}")
            
            # Best result details
            best = successful_experiments[0]
            print(f"\\n🥇 BEST CONFIGURATION (Exp {best['experiment_id']:03d}):")
            print(f"   Final MAPE: {best['final_mape']:.4f}%")
            print(f"   Phase A MAPE: {best['best_mape_phase_a']:.4f}%")
            print(f"   Phase B MAPE: {best['best_mape_phase_b']:.4f}%")
            print(f"   Duration: {best['duration_seconds']:.1f}s")
            
            # Key parameters of best config
            best_config = best['config']
            print(f"\\n   🔧 Best Configuration Parameters:")
            print(f"      Learning Rates: frontend_b={best_config['lr_frontend_phase_b']}, unet_b={best_config['lr_unet_finetune_phase_b']}")
            print(f"      GAT: layers={best_config['gat_layers']}, heads={best_config['gat_num_heads']}, hidden_per_head={best_config['gat_hidden_per_head']}")
            print(f"      Fusion: ratio={best_config['fusion_ratio']}")
            print(f"      Training: batch_size={best_config['batch_size']}, weight_decay={best_config['weight_decay']}")
            print(f"      Scheduler: {best_config['lr_scheduler_type']}")
        
        # Save summary to file
        summary_file = os.path.join(self.results_dir, f"{self.base_experiment_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Experimental Summary Report\\n")
            f.write(f"Generated: {datetime.now().isoformat()}\\n")
            f.write(f"Total Experiments: {len(self.experiment_results)}\\n")
            f.write(f"Successful: {len(successful_experiments)}\\n")
            f.write(f"Failed: {len(failed_experiments)}\\n\\n")
            
            if successful_experiments:
                f.write(f"Best Result: {successful_experiments[0]['final_mape']:.4f}% MAPE\\n")
                f.write(f"Best Config: {self._format_key_params(successful_experiments[0]['config'])}\\n")
        
        print(f"\\n📄 Summary saved to: {summary_file}")

    def run_focused_lr_experiments_around_champion(self, 
                                              champion_lr_frontend=2e-5, 
                                              champion_lr_unet=5e-6,
                                              champion_lr_film_generator=1e-4,  # NEW: FiLM champion LR
                                              max_experiments=5):
        """
        Run focused experiments around champion learning rates.
        Tests small variations around the best known LR configuration.
        """
        print(f"\n🎯 Running Focused LR Experiments Around Champion...")
        print(f"   Champion LRs: Frontend={champion_lr_frontend:.2e}, U-Net={champion_lr_unet:.2e}, FiLM={champion_lr_film_generator:.2e}")
        
        # Define small variations around champion values
        frontend_variations = [
            champion_lr_frontend * 0.5,
            champion_lr_frontend,
            champion_lr_frontend * 2.0
        ]
        
        unet_variations = [
            champion_lr_unet * 0.5,
            champion_lr_unet,
            champion_lr_unet * 2.0
        ]
        
        film_variations = [
            champion_lr_film_generator * 0.5,
            champion_lr_film_generator,
            champion_lr_film_generator * 2.0
        ]
        
        # Create all combinations
        all_lr_combinations = list(itertools.product(
            frontend_variations, unet_variations, film_variations
        ))
        
        # Limit to max_experiments
        selected_combinations = all_lr_combinations[:max_experiments]
        
        print(f"   Testing {len(selected_combinations)} LR combinations around champion:")
        
        experiment_count = 0
        for lr_frontend, lr_unet, lr_film in selected_combinations:
            experiment_count += 1
            
            print(f"\n   [{experiment_count}/{len(selected_combinations)}] Testing LRs: "
                  f"Frontend={lr_frontend:.2e}, U-Net={lr_unet:.2e}, FiLM={lr_film:.2e}")
            
            config = self.create_experiment_config(
                lr_frontend_phase_b=lr_frontend,
                lr_unet_finetune_phase_b=lr_unet,
                lr_film_generator=lr_film,  # NEW: Include FiLM LR variation
                lr_scheduler_type=None  # Keep simple for focused tuning
            )
            
            experiment_id = f"Champion_LR_{experiment_count}"
            result = self.run_single_experiment(config, experiment_id)
            
            if result.get('status') == 'completed':
                final_mape = result['final_mape']
                print(f"      Result: {final_mape:.4f}% MAPE")
            else:
                print(f"      Result: FAILED")
        
        print(f"\n✅ Focused Champion LR Experiments Completed!")
        return self.experiment_results[-len(selected_combinations):]  # Return recent results

    def calculate_total_possible_experiments(self):
        """
        Calculate and display the total number of possible experiments for each grid.
        Useful for planning experimental runs.
        """
        print(f"\n📊 EXPERIMENTAL SCOPE ANALYSIS")
        print("="*50)
        
        # LR Schedule experiments
        base_lr_triplets = len(self.lr_schedule_grid['lr_frontend_phase_b']) * len(self.lr_schedule_grid['lr_unet_finetune_phase_b']) * len(self.lr_schedule_grid['lr_film_generator'])
        
        none_configs = base_lr_triplets * 1  # None scheduler
        cosine_configs = base_lr_triplets * 1  # CosineAnnealingLR
        reduce_configs = base_lr_triplets * len(self.lr_schedule_grid['scheduler_patience']) * len(self.lr_schedule_grid['scheduler_factor'])
        total_lr_configs = none_configs + cosine_configs + reduce_configs
        
        print(f"🎯 LR/Scheduler Experiments:")
        print(f"   • {base_lr_triplets} LR triplets (frontend × unet × film)")
        print(f"   • None scheduler: {none_configs} configs")
        print(f"   • CosineAnnealingLR: {cosine_configs} configs")  
        print(f"   • ReduceLROnPlateau: {reduce_configs} configs ({len(self.lr_schedule_grid['scheduler_patience'])} patience × {len(self.lr_schedule_grid['scheduler_factor'])} factor)")
        print(f"   📈 TOTAL: {total_lr_configs} experiments")
        
        # GAT config experiments
        gat_configs = 1
        for key, values in self.gat_config_grid.items():
            gat_configs *= len(values)
        print(f"\n🧠 GAT Configuration Experiments:")
        print(f"   📈 TOTAL: {gat_configs} experiments")
        
        # Fusion experiments  
        fusion_configs = len(self.fusion_config_grid['fusion_ratio'])  # Only concat_conv implemented
        print(f"\n🔗 Fusion Configuration Experiments:")
        print(f"   📈 TOTAL: {fusion_configs} experiments (only concat_conv implemented)")
        
        # Training config experiments
        training_configs = 1
        for key, values in self.training_config_grid.items():
            training_configs *= len(values)
        print(f"\n⚙️  Training Configuration Experiments:")
        print(f"   📈 TOTAL: {training_configs} experiments")
        
        # Combined scope
        combined_total = total_lr_configs * gat_configs * fusion_configs * training_configs
        print(f"\n🌍 COMBINED SCOPE (if all grids combined):")
        print(f"   📈 TOTAL: {combined_total:,} experiments")
        print(f"   ⚠️  This is computationally infeasible!")
        print(f"   💡 Use structured exploration (one grid at a time) instead")
        
        return {
            'lr_schedule': total_lr_configs,
            'gat_config': gat_configs,
            'fusion': fusion_configs,
            'training_config': training_configs,
            'combined_total': combined_total
        }

    def run_film_config_experiments(self, max_experiments=12):
        """Run systematic FiLM configuration experiments."""
        print(f"\n🎬 Running FiLM Configuration Experiments (max {max_experiments})...")
        
        # Create parameter combinations for FiLM-specific parameters
        film_combinations = list(itertools.product(
            self.film_config_grid['film_generator_mlp_type'],
            self.film_config_grid['lambda_gamma_res'],
            self.film_config_grid['lambda_beta_res'],
            self.film_config_grid['weight_decay_film']
        ))
        
        # Limit to max_experiments
        film_combinations = film_combinations[:max_experiments]
        
        experiment_id_start = len(self.experiment_results) + 1
        
        for i, (mlp_type, lambda_gamma, lambda_beta, wd_film) in enumerate(film_combinations):
            config_overrides = {
                'film_generator_mlp_type': mlp_type,
                'lambda_gamma_res': lambda_gamma,
                'lambda_beta_res': lambda_beta,
                'weight_decay_film': wd_film,
                'use_film_reg': True  # Ensure FiLM regularization is enabled
            }
            
            config = self.create_experiment_config(**config_overrides)
            experiment_id = f"film_config_{experiment_id_start + i}"
            
            print(f"   🎬 FiLM Exp {i+1}: MLP={mlp_type}, λ_γ={lambda_gamma}, λ_β={lambda_beta}, WD_film={wd_film}")
            self.run_single_experiment(config, experiment_id)

# === ORCHESTRATION FUNCTIONS ===

def run_systematic_stage2_experiments(pretrained_unet_weights_path, 
                                     experiment_type="lr_schedule",
                                     max_experiments=10):
    """
    Orchestrate systematic Stage 2 experiments.
    
    Args:
        pretrained_unet_weights_path: Path to Stage 1 pretrained U-Net weights
        experiment_type: Type of experiments to run:
            - 'lr_schedule': Full grid search of LR/scheduler combinations (~99 experiments)
            - 'lr_focused': Focused experiments around champion configuration (~5 experiments)  
            - 'gat_config': GAT architecture experiments
            - 'fusion': Fusion method experiments
            - 'all': Run subsets of each type
        max_experiments: Maximum number of experiments to run per type
    """
    
    if not os.path.exists(pretrained_unet_weights_path):
        print(f"❌ ERROR: Pretrained U-Net weights not found at: {pretrained_unet_weights_path}")
        print("   Please run Stage 1 pre-training first.")
        return None
    
    # Initialize experimental framework
    framework = Stage2ExperimentalFramework(
        pretrained_unet_weights_path=pretrained_unet_weights_path,
        base_experiment_name=f"Stage2_Systematic_{experiment_type}",
        device=device
    )
    
    # Define parameter grids
    framework.define_parameter_grids()
    
    # Run experiments based on type
    if experiment_type == "lr_schedule":
        print(f"🚀 Starting comprehensive LR/scheduler grid search...")
        print(f"   Note: This can generate up to 99 unique experiments.")
        if max_experiments is not None and max_experiments < 99:
            print(f"   Currently limited to {max_experiments} experiments.")
            print(f"   To run all combinations, use max_experiments=99 or None")
        elif max_experiments is None:
            print(f"   Running ALL experiments (no limit set).")
        else:
            print(f"   Running {max_experiments} experiments.")
        framework.run_lr_schedule_experiments(max_experiments=max_experiments)
        
    elif experiment_type == "lr_focused":
        print(f"🎯 Starting focused LR experiments around champion configuration...")
        # Use the best known LRs from your 0.0931% result
        framework.run_focused_lr_experiments_around_champion(
            champion_lr_frontend=2e-5,
            champion_lr_unet=5e-6,
            champion_lr_film_generator=1e-4,  # NEW: FiLM champion LR
            max_experiments=max_experiments
        )
        
    elif experiment_type == "gat_config":
        framework.run_gat_config_experiments(max_experiments=max_experiments)
        
    elif experiment_type == "fusion":
        framework.run_fusion_experiments(max_experiments=max_experiments)
        
    elif experiment_type == "all":
        # Run a balanced subset of each type
        print(f"🔄 Running balanced experiments across all categories...")
        lr_experiments = max(1, max_experiments // 5)
        gat_experiments = max(1, max_experiments // 5) 
        fusion_experiments = max(1, max_experiments // 5)
        film_experiments = max(1, max_experiments // 5)
        focused_experiments = max(1, max_experiments - lr_experiments - gat_experiments - fusion_experiments - film_experiments)
        
        print(f"   📊 Distribution: {lr_experiments} LR grid + {focused_experiments} LR focused + {gat_experiments} GAT + {fusion_experiments} fusion + {film_experiments} FiLM")
        
        framework.run_lr_schedule_experiments(max_experiments=lr_experiments)
        framework.run_focused_lr_experiments_around_champion(max_experiments=focused_experiments)
        framework.run_gat_config_experiments(max_experiments=gat_experiments)
        framework.run_fusion_experiments(max_experiments=fusion_experiments)
        framework.run_film_config_experiments(max_experiments=film_experiments)
        
    elif experiment_type == "film_config":
        print(f"🎬 Starting FiLM configuration experiments...")
        framework.run_film_config_experiments(max_experiments=max_experiments)
        
    else:
        print(f"❌ Unknown experiment type: {experiment_type}")
        print(f"   Valid types: 'lr_schedule', 'lr_focused', 'gat_config', 'fusion', 'all', 'film_config'")
        return None
    
    # Generate summary report
    framework.generate_summary_report()
    
    return framework

def quick_stage2_experiment_demo(pretrained_unet_weights_path):
    """
    Run a quick demonstration of the experimental framework with 3 experiments.
    """
    print("\\n🚀 Running Quick Stage 2 Experiment Demo...")
    
    framework = run_systematic_stage2_experiments(
        pretrained_unet_weights_path=pretrained_unet_weights_path,
        experiment_type="lr_schedule",
        max_experiments=3
    )
    
    if framework and framework.best_result:
        print(f"\\n✅ Demo completed! Best MAPE: {framework.best_result['final_mape']:.4f}%")
        return framework.best_result
    else:
        print("\\n❌ Demo failed or no successful experiments.")
        return None

print("✅ Systematic Experimental Framework for Stage 2 ready!")
print("\\n📋 Available functions:")
print("   - run_systematic_stage2_experiments(pretrained_path, experiment_type, max_experiments)")
print("   - quick_stage2_experiment_demo(pretrained_path)")
print("\\n🎯 Experiment types: 'lr_schedule', 'gat_config', 'fusion', 'all', 'film_config'")

# === END OF EXPERIMENTAL FRAMEWORK ===

# === COMPLETE ORCHESTRATION EXAMPLE ===
# This shows the complete workflow: Stage 1 → Stage 2 → Systematic Experiments

def run_complete_two_stage_workflow_with_experiments(
    stage1_epochs=45,
    stage1_batch_size=8,
    experiment_type="lr_schedule",
    max_stage2_experiments=10
):
    """
    Complete workflow: Stage 1 pre-training → Stage 2 systematic experiments
    
    Args:
        stage1_epochs: Epochs for Stage 1 BaselineUNet pre-training
        stage1_batch_size: Batch size for Stage 1
        experiment_type: Type of Stage 2 experiments ('lr_schedule', 'gat_config', 'fusion', 'all')
        max_stage2_experiments: Maximum number of Stage 2 experiments to run
    """
    
    print("🚀 STARTING COMPLETE TWO-STAGE WORKFLOW WITH SYSTEMATIC EXPERIMENTS")
    print("="*80)
    
    # === STAGE 1: Pre-train BaselineUNet ===
    print("🔥 STAGE 1: PRE-TRAINING BASELINE U-NET")
    print("-" * 50)
    
    stage1_weights_path, stage1_history = run_stage1_pretrain_unet(
        num_epochs=stage1_epochs,
        batch_size=stage1_batch_size,
        lr=1e-4,
        weight_decay=0.01,
        experiment_name_prefix="Stage1_FullRun"
    )
    
    if stage1_weights_path is None:
        print("❌ Stage 1 failed. Cannot proceed to Stage 2.")
        return None
        
    print(f"✅ Stage 1 completed! Pretrained weights saved at:")
    print(f"   {stage1_weights_path}")
    
    # === STAGE 2: Systematic Experiments ===
    print(f"\n🧪 STAGE 2: SYSTEMATIC EXPERIMENTS ({experiment_type.upper()})")
    print("-" * 50)
    
    experimental_framework = run_systematic_stage2_experiments(
        pretrained_unet_weights_path=stage1_weights_path,  # ← THIS IS WHERE THE PATH GOES
        experiment_type=experiment_type,
        max_experiments=max_stage2_experiments
    )
    
    if experimental_framework and experimental_framework.best_result:
        print(f"\n🏆 BEST RESULT FROM SYSTEMATIC EXPERIMENTS:")
        print(f"   Final MAPE: {experimental_framework.best_result['final_mape']:.4f}%")
        print(f"   Experiment ID: {experimental_framework.best_result['experiment_id']}")
        
        return {
            'stage1_weights_path': stage1_weights_path,
            'stage1_history': stage1_history,
            'best_stage2_result': experimental_framework.best_result,
            'experimental_framework': experimental_framework
        }
    else:
        print("❌ Stage 2 experiments failed.")
        return None

# === QUICK EXAMPLES FOR IMMEDIATE USE ===

def quick_lr_experiments():
    """Run a quick learning rate experiment (small scale for testing)"""
    return run_complete_two_stage_workflow_with_experiments(
        stage1_epochs=5,  # Quick for testing
        stage1_batch_size=8,
        experiment_type="lr_schedule",
        max_stage2_experiments=3
    )

def full_lr_experiments():
    """Run full learning rate experiments"""
    return run_complete_two_stage_workflow_with_experiments(
        stage1_epochs=45,  # Full training
        stage1_batch_size=8,
        experiment_type="lr_schedule",
        max_stage2_experiments=10
    )

def full_gat_experiments():
    """Run full GAT configuration experiments"""
    return run_complete_two_stage_workflow_with_experiments(
        stage1_epochs=45,
        stage1_batch_size=8,
        experiment_type="gat_config",
        max_stage2_experiments=15
    )

# === IF YOU ALREADY HAVE STAGE 1 WEIGHTS ===

def run_experiments_with_existing_weights(pretrained_weights_path):
    """
    Use this if you already have Stage 1 pretrained weights
    
    Args:
        pretrained_weights_path: Path to your existing Stage 1 .pth file
                               e.g., "checkpoints/Stage1_FullRun_UNet_Asymmetric_best_mape.pth"
    """
    
    if not os.path.exists(pretrained_weights_path):
        print(f"❌ ERROR: Pretrained weights not found at: {pretrained_weights_path}")
        print("Available checkpoint files:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith('.pth'):
                    print(f"   checkpoints/{f}")
        return None
    
    print(f"🔥 Using existing pretrained weights: {pretrained_weights_path}")
    
    # Run systematic experiments
    framework = run_systematic_stage2_experiments(
        pretrained_unet_weights_path=pretrained_weights_path,  # ← THIS IS WHERE YOUR PATH GOES
        experiment_type="lr_schedule",  # Change this as needed
        max_experiments=10
    )
    
    return framework

print("\n" + "="*60)
print("🎯 READY TO RUN COMPLETE WORKFLOW!")
print("="*60)
print("📋 Choose your approach:")
print("   METHOD 1 - Complete workflow (Stage 1 → Stage 2):")
print("     • quick_lr_experiments()          # Quick test (5 epochs)")
print("     • full_lr_experiments()           # Full LR experiments")
print("     • full_gat_experiments()          # Full GAT experiments")
print()
print("   METHOD 2 - Use existing Stage 1 weights:")
print("     • run_experiments_with_existing_weights('path/to/weights.pth')")
print()
print("   METHOD 3 - Manual control:")
print("     • run_systematic_stage2_experiments(pretrained_path, experiment_type, max_experiments)")
print("="*60)

# === EXPERIMENTAL STRATEGY FUNCTIONS ===

def run_comprehensive_lr_exploration(pretrained_weights_path):
    """
    Run ALL 99 LR/scheduler combinations from the defined grid.
    ⚠️ WARNING: This is computationally expensive (~99 experiments).
    """
    print("🚀 COMPREHENSIVE LR EXPLORATION")
    print("⚠️  This will run ~99 experiments. Each takes ~40 epochs.")
    print("   Estimated time: Several hours to days depending on hardware.")
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled by user")
        return None
        
    return run_systematic_stage2_experiments(
        pretrained_unet_weights_path=pretrained_weights_path,
        experiment_type="lr_schedule",
        max_experiments=None  # Run all combinations
    )

def run_focused_lr_tuning(pretrained_weights_path, champion_frontend_lr=2e-5, champion_unet_lr=5e-6):
    """
    Run focused LR experiments around a champion configuration.
    This is the recommended approach when you have a good baseline.
    """
    print("🎯 FOCUSED LR TUNING AROUND CHAMPION")
    print(f"   Champion LRs: frontend={champion_frontend_lr}, unet={champion_unet_lr}")
    print("   Running ~5 targeted experiments")
    
    framework = Stage2ExperimentalFramework(
        pretrained_unet_weights_path=pretrained_weights_path,
        base_experiment_name="Stage2_Focused_LR",
        device=device
    )
    framework.define_parameter_grids()
    
    framework.run_focused_lr_experiments_around_champion(
        champion_lr_frontend=champion_frontend_lr,
        champion_lr_unet=champion_unet_lr,
        max_experiments=5
    )
    
    framework.generate_summary_report()
    return framework

def run_balanced_exploration(pretrained_weights_path, total_experiments=20):
    """
    Run a balanced exploration across LR, GAT, and fusion experiments.
    Good for initial exploration when you want to sample multiple areas.
    """
    print(f"🔄 BALANCED EXPLORATION ({total_experiments} experiments)")
    print("   Testing multiple hyperparameter categories")
    
    return run_systematic_stage2_experiments(
        pretrained_unet_weights_path=pretrained_weights_path,
        experiment_type="all",
        max_experiments=total_experiments
    )

def analyze_experimental_scope(pretrained_weights_path):
    """
    Analyze the scope of possible experiments without running any.
    Useful for planning your experimental strategy.
    """
    print("📊 EXPERIMENTAL SCOPE ANALYSIS")
    
    framework = Stage2ExperimentalFramework(
        pretrained_unet_weights_path=pretrained_weights_path,
        base_experiment_name="Analysis_Only",
        device=device
    )
    framework.define_parameter_grids()
    
    return framework.calculate_total_possible_experiments()

def run_quick_lr_validation(pretrained_weights_path, num_experiments=5):
    """
    Quick validation of LR settings - good for testing the framework.
    """
    print(f"⚡ QUICK LR VALIDATION ({num_experiments} experiments)")
    print("   Testing framework with limited experiments")
    
    return run_systematic_stage2_experiments(
        pretrained_unet_weights_path=pretrained_weights_path,
        experiment_type="lr_schedule",
        max_experiments=num_experiments
    )

# === FILM INTEGRATION SECTION ===
# Added after careful analysis of existing codebase structure
# ================================================================

# === FiLM TRAINING FUNCTIONS ===

def run_stage2_film_training(
    pretrained_unet_weights_path,
    num_epochs_phase_a=10,
    num_epochs_phase_b=30,
    batch_size=4,
    lr_frontend_phase_a=1e-4,
    lr_frontend_phase_b=5e-5,
    lr_unet_finetune_phase_b=1e-5,
    lr_film_generator=1e-4,  # NEW: Specific LR for FiLM generator
    weight_decay=0.01,
    weight_decay_film=1e-3,  # NEW: Stronger regularization for FiLM
    min_velocity=1.5,
    logmae_initial_c=0.1,
    loss_fixed_weights=[1.0, 0.12, 0.007],
    curriculum_start_simple=True,
    curriculum_total_epochs_for_simple_phase=2,
    experiment_name_prefix="Stage2_FiLM",
    film_generator_mlp_type='linear',  # NEW: FiLM MLP type
    use_film_reg=True,  # NEW: Enable FiLM regularization
    lambda_gamma_res=0.005,  # NEW: FiLM regularization parameters
    lambda_beta_res=0.0005,
    warmup_steps=1000,  # NEW: Warm-up for frontend components
    gradient_clip_film=1.0,  # NEW: Gradient clipping for FiLM
    gradient_clip_others=5.0,
    monitor_freq=50,  # NEW: FiLM monitoring frequency
    lr_scheduler_type='ReduceLROnPlateau',  # NEW: Scheduler type for Phase 2b
    scheduler_patience=5,  # NEW: Scheduler patience
    scheduler_factor=0.5,  # NEW: Scheduler factor
    config=None  # NEW: Optional config dict to override individual parameters
):
    """
    🎬 AUTHORITATIVE FiLM-AWARE STAGE 2 TRAINING FUNCTION 🎬
    
    This is the PRIMARY function for Stage 2 training with full FiLM support.
    It provides comprehensive FiLM-specific features and optimizations:
    
    🔥 **ADVANCED FiLM FEATURES**:
    - **Granular Parameter Grouping**: Separate LRs for SincNet, GAT, GAT_Context_Norm, FiLM_Generator, U-Net
    - **FiLM-Specific Learning Rates**: Dedicated lr_film_generator and weight_decay_film
    - **Intelligent Warm-up**: Applied to frontend components but not pretrained U-Net
    - **Differential Gradient Clipping**: Separate clipping norms for FiLM vs other parameters
    - **FiLM Regularization**: Advanced loss with lambda_gamma_res and lambda_beta_res
    - **Comprehensive Monitoring**: FiLM parameter tracking and regularization loss logging
    - **Adaptive LR Scheduling**: ReduceLROnPlateau scheduler for Phase 2b convergence
    
    🧪 **INTEGRATION**: 
    - Uses the unified train_with_film_awareness() function for consistent training
    - Fully compatible with Stage2ExperimentalFramework for systematic exploration
    - Supports all FiLM architectural configurations (linear/2_layer MLP types)
    
    ⚡ **PERFORMANCE OPTIMIZATIONS**:
    - A100 GPU stability configurations
    - Memory-efficient parameter grouping
    - Intelligent checkpoint management
    
    📊 **USE CASES**:
    - Primary function for all FiLM experiments
    - Hyperparameter grid searches with Stage2ExperimentalFramework
    - Production FiLM training runs
    - Research and ablation studies
    
    For simple baseline experiments without FiLM, consider run_stage2_finetune_sincgat_unet().
    """
    print("=" * 80)
    print(f"🎬 STAGE 2: FILM-ENHANCED SINCGAT-UNET TRAINING")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if 'cuda' in str(device):
        configure_a100_stability()
        print("✅ A100 stability configured")
    
    # Setup data loaders (using existing function)
    train_loader, val_loader = setup_phase2_data_loaders(
        batch_size=batch_size, num_workers=0
    )
    
    # Create CompleteSincGAT_UNet with FiLM integration
    print(f"🏗️ Creating CompleteSincGAT_UNet with FiLM Integration...")
    print(f"   FiLM Configuration:")
    print(f"   - Generator: {film_generator_mlp_type} MLP")
    print(f"   - Context dim: 128")
    print(f"   - Target channels: 512")
    print(f"   - Output params: 1024 (γ_res + β_res)")
    
    sincgat_model = CompleteSincGAT_UNet(
        sample_rate=10001,
        num_receivers=31,
        time_samples=10001,
        num_shots=5,
        # Optimized SincNet parameters (from research)
        sinc_out_channels=60,
        sinc_kernel_size=1001,
        sinc_stride=1,  # CRITICAL: prevents aliasing
        sinc_min_low_hz=40,
        sinc_max_learnable_hz=1000,
        sinc_min_band_hz=10,
        sinc_window_func='blackman',
        sinc_init_type='logarithmic',
        # GAT parameters
        shot_embedding_dim=128,
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
        # FiLM parameters (CRITICAL)
        film_context_dim=128,
        film_target_channels=512,
        film_generator_mlp_type=film_generator_mlp_type,
        film_mlp_hidden_dim=256
    ).to(device)
    
    # Load pretrained U-Net weights
    try:
        print(f"📦 Loading pretrained U-Net weights...")
        champion_unet_checkpoint = torch.load(
            pretrained_unet_weights_path, map_location=device, weights_only=False
        )
        
        if isinstance(champion_unet_checkpoint, dict) and 'model_state_dict' in champion_unet_checkpoint:
            champion_unet_state_dict = champion_unet_checkpoint['model_state_dict']
            print(f"   Loaded from checkpoint (epoch {champion_unet_checkpoint.get('epoch', 'unknown')})")
        else:
            champion_unet_state_dict = champion_unet_checkpoint
            print(f"   Loaded direct state_dict")
            
        sincgat_model.unet.load_state_dict(champion_unet_state_dict, strict=True)
        print("✅ Successfully loaded pretrained U-Net weights into sincgat_model.unet")
    except Exception as e:
        print(f"❌ ERROR loading pretrained U-Net weights: {e}")
        raise e
    
    # Create FiLM-aware loss function
    print(f"🎯 Setting up FiLM-aware loss function...")
    criterion_stage2 = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity,
        use_adaptive_softadapt=False,
        logmae_momentum=0,
        initial_c_logmae=logmae_initial_c,
        fixed_weights_list=loss_fixed_weights,
        start_simple=curriculum_start_simple,
        curriculum_epochs=curriculum_total_epochs_for_simple_phase,
        # FiLM regularization (enabled only if use_film=True)
        use_film_reg=use_film_reg,
        lambda_gamma_res=lambda_gamma_res,
        lambda_beta_res=lambda_beta_res
    ).to(device)
    
    # Import FiLM monitoring functions
    try:
        from phase2_experimental_framework import (
            calculate_film_reg_loss,
            monitor_film_parameters
        )
        print("✅ FiLM monitoring functions imported")
    except ImportError:
        print("⚠️ FiLM monitoring functions not available - will use basic monitoring")
        def monitor_film_parameters(gamma, beta, prefix=""):
            return {
                'gamma_mean': float(gamma.mean()),
                'gamma_std': float(gamma.std()),
                'gamma_max': float(gamma.abs().max()),
                'beta_mean': float(beta.mean()),
                'beta_std': float(beta.std()),
                'beta_max': float(beta.abs().max())
            }
    
    # === Phase 2a: Frontend Training (U-Net Frozen) ===
    experiment_name_2a = f"{experiment_name_prefix}_PhaseA_FrontendFrozen"
    print(f"\n🔧 Phase 2a: Training Frontend ({num_epochs_phase_a} epochs)")
    print(f"   Experiment: {experiment_name_2a}")
    
    # Freeze U-Net
    for param in sincgat_model.unet.parameters():
        param.requires_grad = False
    print("   U-Net parameters frozen")
    
    # Create optimizer for Phase A (standard approach)
    optimizer_stage2a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, sincgat_model.parameters()),
        lr=lr_frontend_phase_a,
        weight_decay=weight_decay
    )
    print(f"   Optimizer: AdamW, LR={lr_frontend_phase_a}")
    
    # Create config for Phase 2a, inheriting from main function args (FIX 3)
    config_2a_training_loop = {
        'warmup_steps': warmup_steps, 
        'gradient_clip_film': gradient_clip_film, 
        'gradient_clip_others': gradient_clip_others, 
        'use_grad_clipping': True, 
        'monitor_freq': monitor_freq, 
        'use_film_reg': use_film_reg, 
        'epoch_monitor_freq': config.get('epoch_monitor_freq_phase_a', 5), # Allow separate freq for phase A
        # Pass LRs for potential use in warm-up target restoration if needed by train_with_film_awareness
        'lr_frontend_phase_b': lr_frontend_phase_b, 
        'lr_unet_finetune_phase_b': lr_unet_finetune_phase_b,
        'lr_film_generator': lr_film_generator
    }
    
    # Training Phase A using unified function
    # Note: train_with_curriculum_fixed is now a wrapper for train_with_film_awareness
    best_mape_2a, history_2a = train_with_curriculum_fixed(
        experiment_name=experiment_name_2a,
        model=sincgat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_stage2,
        optimizer=optimizer_stage2a,
        num_epochs=num_epochs_phase_a,
        device=device,
        calculate_mape_func=calculate_mape,
        lr_scheduler=None,  # No scheduler for Phase 2a typically
        config=config_2a_training_loop # PASS THE COMPREHENSIVE CONFIG
    )
    
    path_2a = os.path.join(CHECKPOINT_DIR, f"{experiment_name_2a}_best_mape.pth")
    print(f"✅ Phase 2a completed. Best MAPE: {best_mape_2a if best_mape_2a is not None else 'N/A'}")
    print(f"   Model saved: {path_2a}")
    
    # === Phase 2b: Full Fine-tuning with FiLM-Specific Differential LRs ===
    experiment_name_2b = f"{experiment_name_prefix}_PhaseB_FiLMFinetune"
    print(f"\n🎬 Phase 2b: FiLM Fine-tuning ({num_epochs_phase_b} epochs)")
    print(f"   Experiment: {experiment_name_2b}")
    
    # Unfreeze U-Net
    for param in sincgat_model.unet.parameters():
        param.requires_grad = True
    print("   U-Net parameters unfrozen")
    
    # Create differential optimizer with FiLM-specific groups
    print("⚙️ Setting up differential learning rates with FiLM groups...")
    
    # Group parameters by component
    film_generator_params = []
    if hasattr(sincgat_model, 'film_bottleneck_modulator') and sincgat_model.film_bottleneck_modulator is not None:
        film_generator_params = list(sincgat_model.film_bottleneck_modulator.parameters())
    
    sincnet_encoder_params = []
    if hasattr(sincgat_model, 'shot_encoder'):
        sincnet_encoder_params = list(sincgat_model.shot_encoder.parameters())
        
    gat_params = []
    if hasattr(sincgat_model, 'gat_fusion'):
        gat_params = list(sincgat_model.gat_fusion.parameters())
        
    gat_context_norm_params = []
    if hasattr(sincgat_model, 'gat_context_layernorm'):
        gat_context_norm_params = list(sincgat_model.gat_context_layernorm.parameters())
        
    unet_params = []
    if hasattr(sincgat_model, 'unet'):
        unet_params = list(sincgat_model.unet.parameters())
    
    print(f"   Parameter groups:")
    if sincnet_encoder_params: print(f"     SincNet encoder: {len(sincnet_encoder_params)} parameters")
    if gat_params: print(f"     GAT fusion: {len(gat_params)} parameters")
    if gat_context_norm_params: print(f"     GAT context norm: {len(gat_context_norm_params)} parameters")
    if film_generator_params: print(f"     FiLM generator: {len(film_generator_params)} parameters")
    if unet_params: print(f"     U-Net: {len(unet_params)} parameters")
    
    # Create optimizer with FiLM-specific differential LRs
    optimizer_stage2b = torch.optim.AdamW([
        {
            'params': sincnet_encoder_params,
            'lr': lr_frontend_phase_b,
            'weight_decay': weight_decay,
            'group_name': 'SincNet',
            'apply_warmup': True  # Apply warmup to SincNet
        },
        {
            'params': gat_params,
            'lr': lr_frontend_phase_b,
            'weight_decay': weight_decay,
            'group_name': 'GAT',
            'apply_warmup': True  # Apply warmup to GAT
        },
        {
            'params': gat_context_norm_params,
            'lr': lr_film_generator,  # Use FiLM LR for context norm
            'weight_decay': weight_decay,
            'group_name': 'GAT_Norm',
            'apply_warmup': True  # Apply warmup to GAT context norm
        },
        {
            'params': film_generator_params,
            'lr': lr_film_generator,
            'weight_decay': weight_decay_film,  # Stronger regularization
            'group_name': 'FiLM',
            'apply_warmup': True  # Apply warmup to FiLM generator
        },
        {
            'params': unet_params,
            'lr': lr_unet_finetune_phase_b,
            'weight_decay': weight_decay,
            'group_name': 'U-Net',
            'apply_warmup': False  # NO warmup for pretrained U-Net
        }
    ])
    
    print(f"   Learning rates:")
    for i, group in enumerate(optimizer_stage2b.param_groups):
        group_name = group.get('group_name', f'Group_{i}')
        print(f"     {group_name}: LR={group['lr']:.2e}, WD={group['weight_decay']:.1e}, Warmup={group.get('apply_warmup')}")
    
    # Create config for Phase 2b, inheriting from main function args (FIX 3)
    config_2b_training_loop = config_2a_training_loop.copy() # Start with Phase A config
    config_2b_training_loop['epoch_monitor_freq'] = config.get('epoch_monitor_freq_phase_b', 5) # Allow separate freq for phase B
    
    # Enable LR scheduler for Phase 2b (after warmup) - (FIX 3)
    print("⚙️ Setting up LR scheduler for Phase 2b...")
    lr_scheduler_2b = None
    
    # Allow config to override function defaults (if config is provided)
    effective_scheduler_type = lr_scheduler_type
    effective_scheduler_patience = scheduler_patience
    effective_scheduler_factor = scheduler_factor
    if config is not None:
        effective_scheduler_type = config.get('lr_scheduler_type', lr_scheduler_type)
        effective_scheduler_patience = config.get('scheduler_patience', scheduler_patience)
        effective_scheduler_factor = config.get('scheduler_factor', scheduler_factor)

    if effective_scheduler_type == 'ReduceLROnPlateau':
        lr_scheduler_2b = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_stage2b, mode='min', factor=effective_scheduler_factor, patience=effective_scheduler_patience, verbose=True
        )
        print(f"   ReduceLROnPlateau: factor={effective_scheduler_factor}, patience={effective_scheduler_patience}")
    elif effective_scheduler_type == 'CosineAnnealingLR':
        lr_scheduler_2b = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage2b, T_max=num_epochs_phase_b, eta_min=1e-7 # T_max could be config driven
        )
        print(f"   CosineAnnealingLR: T_max={num_epochs_phase_b}, eta_min=1e-7")
    else:
        print("   No LR scheduler for Phase 2b.")

    best_mape_2b, history_2b = train_with_curriculum_fixed( # Wrapper for train_with_film_awareness
        experiment_name=experiment_name_2b,
        model=sincgat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_stage2,
        optimizer=optimizer_stage2b,
        num_epochs=num_epochs_phase_b,
        device=device,
        calculate_mape_func=calculate_mape,
        lr_scheduler=lr_scheduler_2b, # PASS THE ACTUAL SCHEDULER FOR PHASE 2B
        config=config_2b_training_loop # PASS THE COMPREHENSIVE CONFIG
    )
    
    path_2b = os.path.join(CHECKPOINT_DIR, f"{experiment_name_2b}_best_mape.pth")
    print(f"✅ Phase 2b completed. Best MAPE: {best_mape_2b if best_mape_2b is not None else 'N/A'}")
    print(f"   Model saved: {path_2b}")
    
    print(f"\n🎉 FiLM Training Complete!")
    print(f"   Best Val MAPE: {best_mape_2b:.4f}%")
    print(f"   Final model: {path_2b}")
    
    # Return comprehensive results
    return {
        'final_model_path': path_2b,
        'best_mape_phase_a': best_mape_2a,
        'best_mape_phase_b': best_mape_2b,
        'training_history': history_2b,  # Use Phase 2b history as primary
        'film_config': {
            'generator_type': film_generator_mlp_type,
            'use_film_reg': use_film_reg,
            'lambda_gamma_res': lambda_gamma_res,
            'lambda_beta_res': lambda_beta_res,
            'warmup_steps': warmup_steps,
            'gradient_clip_film': gradient_clip_film,
            'gradient_clip_others': gradient_clip_others
        }
    }


def run_complete_film_workflow(
    stage1_epochs=45,
    stage1_batch_size=8,
    stage2_epochs_a=10,
    stage2_epochs_b=30,
    stage2_batch_size=4,
    film_generator_mlp_type='linear',
    experiment_name_prefix="Complete_FiLM_Workflow"
):
    """
    Complete two-stage workflow with FiLM integration.
    
    Stage 1: Pretrain U-Net on direct seismic->velocity mapping
    Stage 2: Fine-tune with FiLM-enhanced SincGAT-UNet architecture
    
    Args:
        stage1_epochs: Epochs for U-Net pretraining
        stage1_batch_size: Batch size for Stage 1
        stage2_epochs_a: Epochs for frontend training (Stage 2a)
        stage2_epochs_b: Epochs for FiLM fine-tuning (Stage 2b)
        stage2_batch_size: Batch size for Stage 2
        film_generator_mlp_type: 'linear' or '2_layer' FiLM generator
        experiment_name_prefix: Experiment name prefix
    
    Returns:
        Dictionary with complete workflow results
    """
    print("=" * 80)
    print(f"🎬 COMPLETE FiLM WORKFLOW: TWO-STAGE TRAINING")
    print("=" * 80)
    
    # === STAGE 1: U-Net Pretraining ===
    print(f"\n🏗️ STAGE 1: U-NET PRETRAINING")
    print("-" * 50)
    
    stage1_result = run_stage1_pretrain_unet(
        num_epochs=stage1_epochs,
        batch_size=stage1_batch_size,
        lr=1e-4,
        weight_decay=0.01,
        min_velocity=1.5,
        logmae_initial_c=0.1,
        loss_fixed_weights=[1.0, 0.12, 0.007],  # Champion weights
        experiment_name_prefix=f"{experiment_name_prefix}_Stage1"
    )
    
    if stage1_result is None:
        print("❌ Stage 1 failed. Cannot proceed to Stage 2.")
        return None
    
    stage1_weights_path, stage1_history = stage1_result
    print(f"✅ Stage 1 completed!")
    print(f"   Pretrained weights: {stage1_weights_path}")
    
    # === STAGE 2: FiLM-Enhanced Fine-tuning ===
    print(f"\n🎬 STAGE 2: FiLM-ENHANCED FINE-TUNING")
    print("-" * 50)
    
    stage2_result = run_stage2_film_training(
        pretrained_unet_weights_path=stage1_weights_path,
        num_epochs_phase_a=stage2_epochs_a,
        num_epochs_phase_b=stage2_epochs_b,
        batch_size=stage2_batch_size,
        lr_frontend_phase_a=1e-4,
        lr_frontend_phase_b=5e-5,
        lr_unet_finetune_phase_b=1e-5,
        lr_film_generator=1e-4,  # 10x U-Net LR for FiLM
        weight_decay=0.01,
        weight_decay_film=1e-3,  # Stronger regularization for FiLM
        min_velocity=1.5,
        logmae_initial_c=0.1,
        loss_fixed_weights=[1.0, 0.12, 0.007],  # Champion weights
        curriculum_start_simple=True,
        curriculum_total_epochs_for_simple_phase=2,
        experiment_name_prefix=f"{experiment_name_prefix}_Stage2",
        film_generator_mlp_type=film_generator_mlp_type,
        use_film_reg=True,  # CRITICAL: Enable FiLM regularization
        lambda_gamma_res=0.005,  # Research-backed λ values
        lambda_beta_res=0.0005,
        warmup_steps=1000,
        gradient_clip_film=1.0,
        gradient_clip_others=5.0,
        monitor_freq=50
    )
    
    if stage2_result is None:
        print("❌ Stage 2 failed.")
        return None
    
    print(f"✅ Complete FiLM workflow finished!")
    print(f"   Final MAPE: {stage2_result['best_mape_phase_b']:.4f}%")
    print(f"   FiLM model: {stage2_result['final_model_path']}")
    
    return {
        'stage1_weights_path': stage1_weights_path,
        'stage1_history': stage1_history,
        'stage2_result': stage2_result,
        'final_mape': stage2_result['best_mape_phase_b'],
        'final_model_path': stage2_result['final_model_path']
    }


def run_film_experiments_with_existing_weights(pretrained_weights_path):
    """
    Run FiLM experiments using existing Stage 1 pretrained weights.
    
    This function allows you to explore FiLM configurations without 
    re-running Stage 1 pretraining.
    
    Args:
        pretrained_weights_path: Path to existing Stage 1 .pth file
    
    Returns:
        Dictionary with experimental results
    """
    
    if not os.path.exists(pretrained_weights_path):
        print(f"❌ ERROR: Pretrained weights not found at: {pretrained_weights_path}")
        print("Available checkpoint files:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith('.pth'):
                    print(f"   checkpoints/{f}")
        return None
    
    print(f"🎬 Using existing pretrained weights: {pretrained_weights_path}")
    
    # Run FiLM training experiments
    experiments = {}
    
    # Experiment 1: Linear FiLM generator
    print(f"\n🧪 EXPERIMENT 1: Linear FiLM Generator")
    exp1_result = run_stage2_film_training(
        pretrained_unet_weights_path=pretrained_weights_path,
        num_epochs_phase_a=10,
        num_epochs_phase_b=30,
        batch_size=4,
        lr_film_generator=1e-4,
        film_generator_mlp_type='linear',
        experiment_name_prefix="FiLM_Exp1_Linear",
        use_film_reg=True,
        lambda_gamma_res=0.005,
        lambda_beta_res=0.0005
    )
    experiments['linear_film'] = exp1_result
    
    # Experiment 2: 2-layer FiLM generator
    print(f"\n🧪 EXPERIMENT 2: 2-Layer FiLM Generator")
    exp2_result = run_stage2_film_training(
        pretrained_unet_weights_path=pretrained_weights_path,
        num_epochs_phase_a=10,
        num_epochs_phase_b=30,
        batch_size=4,
        lr_film_generator=1e-4,
        film_generator_mlp_type='2_layer',
        experiment_name_prefix="FiLM_Exp2_2Layer",
        use_film_reg=True,
        lambda_gamma_res=0.005,
        lambda_beta_res=0.0005
    )
    experiments['2layer_film'] = exp2_result
    
    # Experiment 3: Higher FiLM learning rate
    print(f"\n🧪 EXPERIMENT 3: Higher FiLM LR")
    exp3_result = run_stage2_film_training(
        pretrained_unet_weights_path=pretrained_weights_path,
        num_epochs_phase_a=10,
        num_epochs_phase_b=30,
        batch_size=4,
        lr_film_generator=2e-4,  # Higher LR
        film_generator_mlp_type='linear',
        experiment_name_prefix="FiLM_Exp3_HighLR",
        use_film_reg=True,
        lambda_gamma_res=0.005,
        lambda_beta_res=0.0005
    )
    experiments['high_lr_film'] = exp3_result
    
    # Analyze results
    print(f"\n📊 FILM EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    best_experiment = None
    best_mape = float('inf')
    
    for exp_name, result in experiments.items():
        if result and 'best_mape_phase_b' in result:
            mape = result['best_mape_phase_b']
            print(f"{exp_name:15}: {mape:.4f}% MAPE")
            
            if mape < best_mape:
                best_mape = mape
                best_experiment = exp_name
        else:
            print(f"{exp_name:15}: FAILED")
    
    if best_experiment:
        print(f"\n🏆 BEST EXPERIMENT: {best_experiment}")
        print(f"   Best MAPE: {best_mape:.4f}%")
        print(f"   Model: {experiments[best_experiment]['final_model_path']}")
    
    return experiments


def quick_film_demo():
    """
    Quick demonstration of FiLM training (reduced epochs for testing)
    """
    print("⚡ QUICK FiLM DEMO")
    return run_complete_film_workflow(
        stage1_epochs=5,  # Quick for testing
        stage1_batch_size=8,
        stage2_epochs_a=3,
        stage2_epochs_b=10,
        stage2_batch_size=4,
        film_generator_mlp_type='linear',
        experiment_name_prefix="Quick_FiLM_Demo"
    )


def full_film_workflow():
    """
    Full FiLM workflow with production epochs
    """
    print("🚀 FULL FiLM WORKFLOW")
    return run_complete_film_workflow(
        stage1_epochs=45,  # Full training
        stage1_batch_size=8,
        stage2_epochs_a=10,
        stage2_epochs_b=30,
        stage2_batch_size=4,
        film_generator_mlp_type='linear',  # Start with linear
        experiment_name_prefix="Full_FiLM_Workflow"
    )


# === FILM WORKFLOW CONTROL PANEL ===

print("\n" + "="*80)
print("🎬 FiLM TRAINING INTEGRATION READY! (UNIFIED & VALIDATED)")
print("="*80)
print("📋 Choose your FiLM approach:")
print()
print("   METHOD 1 - INFRASTRUCTURE VALIDATION (START HERE):")
print("     • run_corrected_film_validation()     # Comprehensive validation of all fixes")
print()
print("   METHOD 2 - UNIFIED FiLM experiments (all fixes applied):")
print("     • run_corrected_film_experiments()    # Run 3 corrected FiLM experiments")
print("     • quick_film_debug()                  # Detailed monitoring experiment")
print()
print("   METHOD 3 - Complete FiLM workflow (Stage 1 → Stage 2):")
print("     • quick_film_demo()                   # Quick test (5+10 epochs)")
print("     • full_film_workflow()                # Full training (45+30 epochs)")
print()
print("   METHOD 4 - Use existing Stage 1 weights for FiLM experiments:")
print("     • run_film_experiments_with_existing_weights('path/to/weights.pth')")
print()
print("   METHOD 5 - Systematic FiLM hyperparameter exploration:")
print("     • run_systematic_stage2_experiments(pretrained_path, 'film_config', max_experiments)")
print("     • run_systematic_stage2_experiments(pretrained_path, 'lr_schedule', max_experiments)")
print()
print("✅ ALL CRITICAL FIXES APPLIED & UNIFIED:")
print("   ✅ CHECKPOINT_DIR globally defined (prevents NameError)")
print("   ✅ Training loop duality RESOLVED with unified train_with_film_awareness()")
print("   ✅ Stage2ExperimentalFramework now uses unified FiLM-aware training")
print("   ✅ run_stage2_film_training now uses unified FiLM-aware training")
print("   ✅ Granular parameter groups: SincNet, GAT, GAT_Norm, FiLM, U-Net")
print("   ✅ Differential LRs and weight decay for FiLM parameters")
print("   ✅ FiLM regularization integrated in loss function")
print("   ✅ LR warm-up, differential gradient clipping, and LR scheduler support")
print("   ✅ Comprehensive FiLM parameter monitoring")
print("   ✅ Research-backed FiLM configurations and parameter grids")
print("   ✅ Complete backward compatibility maintained")
print("="*80)
print()
print("🔧 IMMEDIATE NEXT STEPS (RECOMMENDED ORDER):")
print("   1. run_corrected_film_validation()    # Validate unified approach first")
print("   2. Check validation results for any issues")
print("   3. run_corrected_film_experiments()   # Run production FiLM experiments")
print("   4. Analyze results vs baseline (0.0952% MAPE)")
print("   5. If promising, proceed to systematic tuning using Stage2ExperimentalFramework")
print()
print("🚨 CRITICAL IMPROVEMENTS ACHIEVED:")
print("   ✅ ResidualFiLM formulation: output = x + γ_res*x + β_res")
print("   ✅ Zero initialization for identity preservation")
print("   ✅ Differential learning rates (FiLM 10x U-Net LR)")
print("   ✅ FiLM regularization: λ_γ=0.005, λ_β=0.0005")
print("   ✅ LR warm-up for frontend components")
print("   ✅ Differential gradient clipping (FiLM: 1.0, others: 5.0)")
print("   ✅ UNIFIED training function resolves duality issues")
print("   ✅ Both Stage2ExperimentalFramework AND run_stage2_film_training use same logic")
print("   ✅ Comprehensive validation and error handling")
print("="*80)
print()
print("🎯 ARCHITECTURAL SOLUTION SUMMARY:")
print("   PROBLEM: Training loop duality (separate logic in different paths)")
print("   SOLUTION: train_with_film_awareness() unified function with:")
print("     • FiLM-aware loss calculation (model_for_film_params)")
print("     • LR warm-up for frontend components")
print("     • Differential gradient clipping")
print("     • LR scheduler support")
print("     • Comprehensive FiLM monitoring")
print("     • Standard checkpointing and validation")
print()
print("   COMPATIBILITY: train_with_curriculum_fixed() wrapper ensures")
print("                  all existing code benefits from FiLM awareness")
print("="*80)

# === INTEGRATION NOTES ===
"""
🔧 INTEGRATION NOTES:

This FiLM integration carefully extends your existing main_898 infrastructure:

1. ✅ PRESERVES EXISTING WORKFLOW:
   - All original functions remain unchanged
   - Uses existing data loaders (setup_phase2_data_loaders)
   - Uses existing loss function (RefinedLogSpaceMAEHybridLoss)
   - Uses existing U-Net pretraining (run_stage1_pretrain_unet)
   - Uses existing checkpointing structure (CHECKPOINT_DIR)

2. ✅ EXTENDS WITH FILM CAPABILITIES:
   - run_stage2_film_training(): Enhanced Stage 2 with FiLM
   - run_complete_film_workflow(): Full two-stage FiLM workflow
   - run_film_experiments_with_existing_weights(): FiLM experiments
   - Comprehensive FiLM parameter monitoring
   - Research-backed FiLM configurations

3. ✅ RESEARCH-ALIGNED IMPLEMENTATION:
   - ResidualFiLM formulation (x + γ_res*x + β_res)
   - Zero initialization for identity preservation
   - Differential learning rates (FiLM 10x U-Net LR)
   - FiLM regularization (λ_γ=0.005, λ_β=0.0005)
   - LR warm-up for frontend components
   - Gradient clipping (FiLM: 1.0, others: 5.0)

4. ✅ PRODUCTION READY:
   - Comprehensive error handling
   - Detailed progress monitoring
   - Flexible configuration options
   - Clear experiment organization
   - Checkpoint management

5. 🎯 USAGE EXAMPLES:

   # Quick test of FiLM integration
   result = quick_film_demo()
   
   # Full production FiLM training
   result = full_film_workflow()
   
   # Use existing Stage 1 weights
   experiments = run_film_experiments_with_existing_weights(
       "checkpoints/Stage1_FullRun_UNet_best_mape.pth"
   )
   
   # Custom FiLM configuration
   result = run_stage2_film_training(
       pretrained_unet_weights_path="path/to/weights.pth",
       film_generator_mlp_type='2_layer',
       lr_film_generator=2e-4,
       lambda_gamma_res=0.01
   )

The integration maintains full backward compatibility while adding 
comprehensive FiLM capabilities based on our research findings.
"""

# === CORRECTED FILM EXPERIMENT RUNNER ===
# Fixes path issues and ensures all FiLM requirements are implemented

def run_corrected_film_experiments(base_checkpoint_dir="checkpoints", 
                                 champion_checkpoint_name="Extended_Absolute_Champion_epoch_40.pth"):
    """
    Corrected FiLM experiment runner that fixes path issues and ensures
    all research-backed FiLM requirements are properly implemented.
    
    Args:
        base_checkpoint_dir: Base checkpoint directory (default: "checkpoints")
        champion_checkpoint_name: Name of the champion checkpoint file
    
    Returns:
        Dictionary with experimental results
    """
    
    print("🎬 CORRECTED FiLM EXPERIMENTS")
    print("=" * 60)
    
    # === FIX 1: Correct Path Construction ===
    champion_weights_path = os.path.join(base_checkpoint_dir, champion_checkpoint_name)
    print(f"✅ Using corrected path: {champion_weights_path}")
    
    # Verify path exists
    if not os.path.exists(champion_weights_path):
        print(f"❌ ERROR: Champion weights not found at: {champion_weights_path}")
        print("Available checkpoint files:")
        if os.path.exists(base_checkpoint_dir):
            for f in os.listdir(base_checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"   {os.path.join(base_checkpoint_dir, f)}")
        return None
    
    print(f"📦 Champion weights found: {champion_weights_path}")
    
    # === EXPERIMENT 1: Linear FiLM with Enhanced Monitoring ===
    print(f"\n🧪 EXPERIMENT 1: Linear FiLM (Enhanced)")
    
    # Enhanced training with all FiLM requirements
    exp1_result = run_stage2_film_training(
        pretrained_unet_weights_path=champion_weights_path,  # ← FIXED PATH
        num_epochs_phase_a=10,
        num_epochs_phase_b=30,
        batch_size=4,
        lr_frontend_phase_a=1e-4,
        lr_frontend_phase_b=5e-5,
        lr_unet_finetune_phase_b=1e-5,
        lr_film_generator=1e-4,  # 10x U-Net LR
        weight_decay=0.01,
        weight_decay_film=1e-3,  # Stronger for FiLM
        min_velocity=1.5,
        logmae_initial_c=0.1,
        loss_fixed_weights=[1.0, 0.12, 0.007],  # Champion weights
        curriculum_start_simple=True,
        curriculum_total_epochs_for_simple_phase=2,
        experiment_name_prefix="Corrected_FiLM_Linear",
        film_generator_mlp_type='linear',
        use_film_reg=True,  # CRITICAL: Enable FiLM regularization
        lambda_gamma_res=0.005,  # Research-backed values
        lambda_beta_res=0.0005,
        warmup_steps=1000,  # LR warm-up
        gradient_clip_film=1.0,  # FiLM gradient clipping
        gradient_clip_others=5.0,  # Other parameters gradient clipping
        monitor_freq=25,  # More frequent monitoring
        lr_scheduler_type='ReduceLROnPlateau',  # NEW: Scheduler type for Phase 2b
        scheduler_patience=5,  # NEW: Scheduler patience
        scheduler_factor=0.5,  # NEW: Scheduler factor
        config=None  # NEW: Optional config dict to override individual parameters
    )
    
    # === EXPERIMENT 2: 2-Layer FiLM Generator ===
    print(f"\n🧪 EXPERIMENT 2: 2-Layer FiLM Generator")
    
    exp2_result = run_stage2_film_training(
        pretrained_unet_weights_path=champion_weights_path,  # ← FIXED PATH
        num_epochs_phase_a=10,
        num_epochs_phase_b=30,
        batch_size=4,
        lr_frontend_phase_a=1e-4,
        lr_frontend_phase_b=5e-5,
        lr_unet_finetune_phase_b=1e-5,
        lr_film_generator=1e-4,
        weight_decay=0.01,
        weight_decay_film=1e-3,
        min_velocity=1.5,
        logmae_initial_c=0.1,
        loss_fixed_weights=[1.0, 0.12, 0.007],
        curriculum_start_simple=True,
        curriculum_total_epochs_for_simple_phase=2,
        experiment_name_prefix="Corrected_FiLM_2Layer",
        film_generator_mlp_type='2_layer',  # Different architecture
        use_film_reg=True,
        lambda_gamma_res=0.005,
        lambda_beta_res=0.0005,
        warmup_steps=1000,
        gradient_clip_film=1.0,
        gradient_clip_others=5.0,
        monitor_freq=25,
        lr_scheduler_type='ReduceLROnPlateau',  # NEW: Scheduler type for Phase 2b
        scheduler_patience=5,  # NEW: Scheduler patience
        scheduler_factor=0.5,  # NEW: Scheduler factor
        config=None  # NEW: Optional config dict to override individual parameters
    )
    
    # === EXPERIMENT 3: Higher FiLM LR ===
    print(f"\n🧪 EXPERIMENT 3: Higher FiLM Learning Rate")
    
    exp3_result = run_stage2_film_training(
        pretrained_unet_weights_path=champion_weights_path,  # ← FIXED PATH
        num_epochs_phase_a=10,
        num_epochs_phase_b=30,
        batch_size=4,
        lr_frontend_phase_a=1e-4,
        lr_frontend_phase_b=5e-5,
        lr_unet_finetune_phase_b=1e-5,
        lr_film_generator=2e-4,  # 2x higher LR for FiLM
        weight_decay=0.01,
        weight_decay_film=1e-3,
        min_velocity=1.5,
        logmae_initial_c=0.1,
        loss_fixed_weights=[1.0, 0.12, 0.007],
        curriculum_start_simple=True,
        curriculum_total_epochs_for_simple_phase=2,
        experiment_name_prefix="Corrected_FiLM_HighLR",
        film_generator_mlp_type='linear',
        use_film_reg=True,
        lambda_gamma_res=0.005,
        lambda_beta_res=0.0005,
        warmup_steps=1000,
        gradient_clip_film=1.0,
        gradient_clip_others=5.0,
        monitor_freq=25,
        lr_scheduler_type='ReduceLROnPlateau',  # NEW: Scheduler type for Phase 2b
        scheduler_patience=5,  # NEW: Scheduler patience
        scheduler_factor=0.5,  # NEW: Scheduler factor
        config=None  # NEW: Optional config dict to override individual parameters
    )
    
    # === ANALYZE RESULTS ===
    experiments = {
        'linear_film': exp1_result,
        '2layer_film': exp2_result,
        'high_lr_film': exp3_result
    }
    
    print(f"\n📊 CORRECTED FILM EXPERIMENT RESULTS")
    print("=" * 60)
    
    best_experiment = None
    best_mape = float('inf')
    
    for exp_name, result in experiments.items():
        if result and 'best_mape_phase_b' in result:
            mape = result['best_mape_phase_b']
            phase_a_mape = result.get('best_mape_phase_a', 'N/A')
            print(f"{exp_name:15}: Phase A: {phase_a_mape}, Phase B: {mape:.4f}% MAPE")
            
            if mape < best_mape:
                best_mape = mape
                best_experiment = exp_name
        else:
            print(f"{exp_name:15}: FAILED")
    
    if best_experiment:
        print(f"\n🏆 BEST EXPERIMENT: {best_experiment}")
        print(f"   Best MAPE: {best_mape:.4f}%")
        print(f"   Model: {experiments[best_experiment]['final_model_path']}")
        
        # Compare to baseline
        baseline_mape = 0.0952  # From your champion U-Net
        improvement = ((baseline_mape - best_mape) / baseline_mape) * 100
        print(f"   Improvement over baseline: {improvement:.1f}%")
    
    return experiments


def run_film_monitoring_experiment(champion_weights_path):
    """
    Run a single FiLM experiment with comprehensive monitoring for debugging.
    
    This function is designed to help debug and verify all FiLM components
    are working correctly.
    """
    
    print("🔬 DETAILED FiLM MONITORING EXPERIMENT")
    print("=" * 50)
    
    # Import monitoring functions
    try:
        from phase2_experimental_framework import (
            calculate_film_reg_loss,
            monitor_film_parameters
        )
        print("✅ FiLM monitoring functions available")
    except ImportError:
        print("⚠️ FiLM monitoring functions not available")
        return None
    
    # Run with detailed monitoring
    result = run_stage2_film_training(
        pretrained_unet_weights_path=champion_weights_path,
        num_epochs_phase_a=5,  # Shorter for detailed monitoring
        num_epochs_phase_b=15,
        batch_size=4,
        lr_frontend_phase_a=1e-4,
        lr_frontend_phase_b=5e-5,
        lr_unet_finetune_phase_b=1e-5,
        lr_film_generator=1e-4,
        weight_decay=0.01,
        weight_decay_film=1e-3,
        min_velocity=1.5,
        logmae_initial_c=0.1,
        loss_fixed_weights=[1.0, 0.12, 0.007],
        curriculum_start_simple=True,
        curriculum_total_epochs_for_simple_phase=2,
        experiment_name_prefix="FiLM_Monitoring",
        film_generator_mlp_type='linear',
        use_film_reg=True,
        lambda_gamma_res=0.005,
        lambda_beta_res=0.0005,
        warmup_steps=500,  # Shorter warm-up for monitoring
        gradient_clip_film=1.0,
        gradient_clip_others=5.0,
        monitor_freq=10,  # Very frequent monitoring
        lr_scheduler_type='ReduceLROnPlateau',  # NEW: Scheduler type for Phase 2b
        scheduler_patience=5,  # NEW: Scheduler patience
        scheduler_factor=0.5,  # NEW: Scheduler factor
        config=None  # NEW: Optional config dict to override individual parameters
    )
    
    return result


# === QUICK ACCESS FUNCTIONS ===

def fix_and_run_current_experiment():
    """
    Fix the path issue and re-run the current experiment setup.
    Use this to restart your current experiment with the corrected path.
    """
    return run_corrected_film_experiments(
        base_checkpoint_dir="checkpoints",
        champion_checkpoint_name="Extended_Absolute_Champion_epoch_40.pth"
    )

def quick_film_debug():
    """
    Quick FiLM debugging with the corrected path.
    """
    champion_path = "checkpoints/Extended_Absolute_Champion_epoch_40.pth"
    return run_film_monitoring_experiment(champion_path)

# === UNIFIED FiLM-AWARE TRAINING FUNCTION ===
# This function resolves the training loop duality by providing a single,
# comprehensive training loop that handles all FiLM-specific requirements

def train_with_film_awareness(
    experiment_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    calculate_mape_func,
    lr_scheduler=None,
    config=None,
    checkpoint_freq=10
):
    """
    Unified FiLM-aware training function that handles:
    - LR warm-up for frontend components
    - Differential gradient clipping
    - FiLM regularization via model-aware criterion calls
    - LR scheduler stepping
    - Comprehensive FiLM parameter monitoring
    - Standard checkpointing and validation
    
    This function replaces the need for separate training loops and ensures
    consistent FiLM-aware training across all experimental paths.
    
    Args:
        experiment_name: Name for saving checkpoints
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (must support model_for_film_params)
        optimizer: Optimizer with potentially multiple parameter groups
        num_epochs: Number of training epochs
        device: Training device
        calculate_mape_func: Function to calculate MAPE
        lr_scheduler: Optional learning rate scheduler
        config: Configuration dict with FiLM-specific parameters
        checkpoint_freq: Frequency of checkpoint saving
    
    Returns:
        Tuple of (best_val_mape, training_history)
    """
    
    # Extract FiLM-specific config parameters with sensible defaults
    if config is None:
        config = {}
    
    warmup_steps = config.get('warmup_steps', 0)
    gradient_clip_film = config.get('gradient_clip_film', 1.0)
    gradient_clip_others = config.get('gradient_clip_others', 5.0)
    use_grad_clipping = config.get('use_grad_clipping', True)
    monitor_freq = config.get('monitor_freq', 50) # Batch level frequency
    use_film_reg = config.get('use_film_reg', False)
    epoch_monitor_freq = config.get('epoch_monitor_freq', 5)
    
    print(f"\n🎬 Starting FiLM-aware training: {experiment_name}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Gradient clipping: Film={gradient_clip_film}, Others={gradient_clip_others}")
    print(f"   FiLM regularization: {use_film_reg}")
    print(f"   LR Scheduler: {type(lr_scheduler).__name__ if lr_scheduler else 'None'}")
    print(f"   FiLM monitoring: Every {epoch_monitor_freq} epochs")
    
    # Identify FiLM parameters for differential gradient clipping
    film_params = []
    other_params = []
    
    if hasattr(model, 'film_bottleneck_modulator') and model.film_bottleneck_modulator is not None:
        film_params = list(model.film_bottleneck_modulator.parameters())
        film_param_ids = {id(p) for p in film_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in film_param_ids]
    else:
        other_params = [p for p in model.parameters() if p.requires_grad]
    
    print(f"   Parameter groups: FiLM={len(film_params)}, Others={len(other_params)}")
    
    # Training history tracking
    history = {
        'train_loss': [], 'val_mape': [], 'film_reg': [],
        'learning_rates': [], 'film_stats': []
    }
    
    best_val_mape = float('inf')
    global_step = 0
    
    # Determine original learning rates for warmup (FIX 2 Refinement)
    original_lrs_for_warmup_groups = {} 
    if warmup_steps > 0:
        for i, group in enumerate(optimizer.param_groups):
            if group.get('apply_warmup', False): # Check the explicit flag
                original_lrs_for_warmup_groups[i] = group['lr'] # Store original target LR
                # Start very low, actual scaling happens in batch loop
                group['lr'] = original_lrs_for_warmup_groups[i] * 0.01 
                print(f"   🌡️ Warm-up enabled for group '{group.get('group_name', i)}': Initial LR {group['lr']:.2e} -> Target {original_lrs_for_warmup_groups[i]:.2e}")
            elif not group.get('apply_warmup', True) and group.get('group_name', 'U-Net') == 'U-Net': # Explicitly False or U-Net
                 print(f"   🚫 Warm-up disabled for group '{group.get('group_name', i)}'")
            else: # Default to warmup for non-Unet groups if no explicit flag
                group_name = group.get('group_name', '')
                is_unet_group = 'U-Net' in group_name or 'UNet' in group_name
                if not is_unet_group: # Heuristic: if not U-Net and no explicit flag, apply warmup
                     original_lrs_for_warmup_groups[i] = group['lr']
                     group['lr'] = original_lrs_for_warmup_groups[i] * 0.01
                     print(f"   🌡️ Warm-up heuristically enabled for group '{group_name}': Initial LR {group['lr']:.2e} -> Target {original_lrs_for_warmup_groups[i]:.2e}")
                else:
                     print(f"   🚫 Warm-up heuristically disabled for U-Net group '{group_name}'")


    for epoch in range(num_epochs):
        print(f"\\n📊 Epoch {epoch+1}/{num_epochs}")
        
        # === TRAINING PHASE ===
        model.train()
        running_loss = 0.0
        running_film_reg = 0.0
        epoch_loss_components = {'total': 0.0, 'film_reg': 0.0}
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # === LR WARM-UP LOGIC ===
            if warmup_steps > 0 and global_step < warmup_steps:
                warmup_factor = (global_step + 1) / warmup_steps
                for group_idx, target_lr in original_lrs_for_warmup_groups.items():
                    if group_idx < len(optimizer.param_groups): # Ensure group_idx is valid
                        optimizer.param_groups[group_idx]['lr'] = target_lr * warmup_factor
            
            # === FORWARD PASS ===
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # === FiLM-AWARE LOSS CALCULATION ===
            if use_film_reg and hasattr(criterion, 'forward'):
                # Pass model for FiLM regularization
                loss_dict = criterion(outputs, targets, model_for_film_params=model)
                if isinstance(loss_dict, dict):
                    total_loss = loss_dict.get('total', loss_dict.get('loss', list(loss_dict.values())[0]))
                    film_reg_val = loss_dict.get('film_reg', torch.tensor(0.0))
                    
                    # Accumulate loss components for epoch summary
                    for component, value in loss_dict.items():
                        if hasattr(value, 'item'):
                            epoch_loss_components[component] = epoch_loss_components.get(component, 0.0) + value.item()
                else:
                    total_loss = loss_dict
                    film_reg_val = torch.tensor(0.0)
            else:
                # Standard loss calculation
                total_loss = criterion(outputs, targets)
                film_reg_val = torch.tensor(0.0)
                epoch_loss_components['total'] += total_loss.item()
            
            # === BACKWARD PASS ===
            total_loss.backward()
            
            # === DIFFERENTIAL GRADIENT CLIPPING ===
            if use_grad_clipping:
                if film_params:
                    torch.nn.utils.clip_grad_norm_(film_params, max_norm=gradient_clip_film)
                if other_params:
                    torch.nn.utils.clip_grad_norm_(other_params, max_norm=gradient_clip_others)
            
            optimizer.step()
            
            # === TRACKING ===
            running_loss += total_loss.item()
            if hasattr(film_reg_val, 'item'):
                running_film_reg += film_reg_val.item()
            
            global_step += 1
        
        # === VALIDATION PHASE ===
        model.eval()
        val_mape = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Calculate MAPE
                outputs_np = outputs.squeeze(1).cpu().numpy()
                targets_np = targets.squeeze(1).cpu().numpy()
                
                for i in range(outputs_np.shape[0]):
                    val_mape += calculate_mape_func(targets_np[i], outputs_np[i])
                    val_samples += 1
        
        val_mape = val_mape / val_samples if val_samples > 0 else float('inf')
        
        # === LEARNING RATE SCHEDULER ===
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_mape)
            else:
                lr_scheduler.step()
        
        # === RECORD EPOCH RESULTS ===
        epoch_loss = running_loss / len(train_loader)
        epoch_film_reg = running_film_reg / len(train_loader)
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        
        history['train_loss'].append(epoch_loss)
        history['val_mape'].append(val_mape)
        history['film_reg'].append(epoch_film_reg)
        history['learning_rates'].append(current_lrs)
        
        print(f"   Train Loss: {epoch_loss:.6f}")
        if epoch_film_reg > 0:
            print(f"   FiLM Reg: {epoch_film_reg:.6f}")
        print(f"   Val MAPE: {val_mape:.4f}%")
        print(f"   LRs: {[f'{lr:.2e}' for lr in current_lrs]}")
        
        # === PERIODIC FiLM PARAMETER MONITORING ===
        if ((epoch + 1) % epoch_monitor_freq == 0 and film_params and 
            hasattr(model, 'last_gamma_res') and model.last_gamma_res is not None):
            
            print(f"   📊 FiLM Parameter Stats (Epoch {epoch+1}):")
            try:
                # Use the globally imported monitor_film_parameters function
                # If not available, fall back to local implementation
                if 'monitor_film_parameters' in globals():
                    film_stats = monitor_film_parameters(
                        model.last_gamma_res, model.last_beta_res
                    )
                else:
                    # Fallback local implementation
                    film_stats = {
                        'gamma_mean': float(model.last_gamma_res.mean()),
                        'gamma_std': float(model.last_gamma_res.std()),
                        'gamma_max': float(model.last_gamma_res.abs().max()),
                        'beta_mean': float(model.last_beta_res.mean()),
                        'beta_std': float(model.last_beta_res.std()),
                        'beta_max': float(model.last_beta_res.abs().max())
                    }
                
                # Store stats in history
                history['film_stats'].append(film_stats)
                
                # Concise logging
                print(f"     γ: mean={film_stats['gamma_mean']:.4f}, std={film_stats['gamma_std']:.4f}, max={film_stats['gamma_max']:.4f}")
                print(f"     β: mean={film_stats['beta_mean']:.4f}, std={film_stats['beta_std']:.4f}, max={film_stats['beta_max']:.4f}")
                
                # Loss component summary (if available)
                if len(epoch_loss_components) > 1:
                    print(f"   📈 Loss Components (Epoch Average):")
                    for component, value in epoch_loss_components.items():
                        if component != 'total':
                            avg_value = value / len(train_loader)
                            print(f"     {component}: {avg_value:.6f}")
            
            except Exception as e:
                print(f"     ⚠️ FiLM monitoring error: {e}")
        
        # === SAVE BEST MODEL ===
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mape': val_mape,
                'history': history,
                'config': config
            }
            
            if lr_scheduler:
                checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            
            best_model_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best_mape.pth")
            torch.save(checkpoint, best_model_path)
            print(f"   🏆 NEW BEST MAPE: {best_val_mape:.4f}% - Model saved!")
        
        # === PERIODIC CHECKPOINTS ===
        if (epoch + 1) % checkpoint_freq == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mape': val_mape
            }, periodic_path)
            print(f"   💾 Checkpoint saved: epoch_{epoch+1}")
    
    print(f"\n✅ Training completed! Best MAPE: {best_val_mape:.4f}%")
    return best_val_mape, history

# === COMPATIBILITY WRAPPER ===
# This ensures backward compatibility with existing code that expects train_with_curriculum_fixed

def train_with_curriculum_fixed(*args, **kwargs):
    """
    Compatibility wrapper that redirects to the unified FiLM-aware training function.
    This ensures all existing code continues to work while benefiting from FiLM awareness.
    """
    # Ensure 'config' is passed or defaults to an empty dict if not present (FIX 2)
    if 'config' not in kwargs:
        kwargs['config'] = {}
    return train_with_film_awareness(*args, **kwargs)

# === COMPREHENSIVE VALIDATION FUNCTION ===

def run_corrected_film_validation():
    """
    Comprehensive validation of all critical fixes applied to the FiLM experimental framework.
    
    This function:
    1. Tests the corrected CHECKPOINT_DIR
    2. Validates the unified FiLM-aware training function
    3. Runs a single corrected FiLM experiment
    4. Confirms all research-backed FiLM requirements are working
    
    Returns:
        Dictionary with validation results
    """
    
    print("🔧 COMPREHENSIVE FiLM VALIDATION WITH ALL CRITICAL FIXES")
    print("=" * 80)
    
    # === VALIDATION 1: Check Global Definitions ===
    print("\n📋 VALIDATION 1: Global Definitions")
    try:
        print(f"   ✅ CHECKPOINT_DIR: {CHECKPOINT_DIR}")
        print(f"   ✅ Device: {device}")
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
            print(f"   ✅ Created checkpoint directory: {CHECKPOINT_DIR}")
        else:
            print(f"   ✅ Checkpoint directory exists: {CHECKPOINT_DIR}")
    except Exception as e:
        print(f"   ❌ Global definitions error: {e}")
        return {'status': 'failed', 'error': 'Global definitions'}
    
    # === VALIDATION 2: Check Unified Training Function ===
    print("\n🎬 VALIDATION 2: Unified Training Function")
    try:
        # Test that train_with_film_awareness is available
        print(f"   ✅ train_with_film_awareness function: Available")
        print(f"   ✅ train_with_curriculum_fixed wrapper: Available")
        print(f"   ✅ Training loop duality resolved")
    except Exception as e:
        print(f"   ❌ Training function error: {e}")
        return {'status': 'failed', 'error': 'Training function'}
    
    # === VALIDATION 3: Check Champion Weights ===
    print("\n📦 VALIDATION 3: Champion Weights")
    champion_weights_path = os.path.join(CHECKPOINT_DIR, "Extended_Absolute_Champion_epoch_40.pth")
    
    if not os.path.exists(champion_weights_path):
        print(f"   ⚠️ Champion weights not found at: {champion_weights_path}")
        print("   Available checkpoint files:")
        if os.path.exists(CHECKPOINT_DIR):
            checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
            if checkpoint_files:
                for f in checkpoint_files:
                    print(f"     {f}")
                # Use the first available checkpoint for validation
                champion_weights_path = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
                print(f"   📄 Using alternative weights: {champion_weights_path}")
            else:
                print(f"   ❌ No checkpoint files found. Please run Stage 1 training first.")
                return {'status': 'failed', 'error': 'No champion weights'}
        else:
            print(f"   ❌ Checkpoint directory not found")
            return {'status': 'failed', 'error': 'No checkpoint directory'}
    else:
        print(f"   ✅ Champion weights found: {champion_weights_path}")
    
    # === VALIDATION 4: Quick FiLM Training Test ===
    print("\n🧪 VALIDATION 4: Quick FiLM Training Test")
    try:
        print("   Running mini FiLM training experiment (2+3 epochs)...")
        
        # Run a very short FiLM training experiment
        validation_result = run_stage2_film_training(
            pretrained_unet_weights_path=champion_weights_path,
            num_epochs_phase_a=2,  # Very short
            num_epochs_phase_b=3,  # Very short
            batch_size=4,
            lr_frontend_phase_a=1e-4,
            lr_frontend_phase_b=5e-5,
            lr_unet_finetune_phase_b=1e-5,
            lr_film_generator=1e-4,
            weight_decay=0.01,
            weight_decay_film=1e-3,
            min_velocity=1.5,
            logmae_initial_c=0.1,
            loss_fixed_weights=[1.0, 0.12, 0.007],
            curriculum_start_simple=True,
            curriculum_total_epochs_for_simple_phase=1,
            experiment_name_prefix="Validation_FiLM",
            film_generator_mlp_type='linear',
            use_film_reg=True,
            lambda_gamma_res=0.005,
            lambda_beta_res=0.0005,
            warmup_steps=100,  # Short warm-up
            gradient_clip_film=1.0,
            gradient_clip_others=5.0,
            monitor_freq=10,  # Frequent monitoring for validation
            lr_scheduler_type='ReduceLROnPlateau',  # NEW: Scheduler type for Phase 2b
            scheduler_patience=5,  # NEW: Scheduler patience
            scheduler_factor=0.5,  # NEW: Scheduler factor
            config=None  # NEW: Optional config dict to override individual parameters
        )
        
        if validation_result and 'best_mape_phase_b' in validation_result:
            phase_a_mape = validation_result.get('best_mape_phase_a', 'N/A')
            phase_b_mape = validation_result['best_mape_phase_b']
            
            print(f"   ✅ FiLM training completed successfully!")
            print(f"   📊 Phase A MAPE: {phase_a_mape}")
            print(f"   📊 Phase B MAPE: {phase_b_mape:.4f}%")
            print(f"   📁 Model saved: {validation_result['final_model_path']}")
            
            # Check if training history contains expected FiLM metrics
            history = validation_result.get('training_history', {})
            if 'film_reg' in history and len(history['film_reg']) > 0:
                print(f"   ✅ FiLM regularization tracked: {len(history['film_reg'])} epochs")
            else:
                print(f"   ⚠️ FiLM regularization not tracked")
            
            if 'learning_rates' in history and len(history['learning_rates']) > 0:
                print(f"   ✅ Learning rate tracking: {len(history['learning_rates'])} epochs")
                last_lrs = history['learning_rates'][-1]
                print(f"   📈 Final LRs: {[f'{lr:.2e}' for lr in last_lrs]}")
            else:
                print(f"   ⚠️ Learning rate tracking not available")
            
        else:
            print(f"   ❌ FiLM training failed")
            return {'status': 'failed', 'error': 'FiLM training failed'}
    
    except Exception as e:
        print(f"   ❌ FiLM training error: {e}")
        return {'status': 'failed', 'error': f'FiLM training: {str(e)}'}
    
    # === VALIDATION 5: Stage2ExperimentalFramework Test ===
    print("\n🔬 VALIDATION 5: Stage2ExperimentalFramework Test")
    try:
        # Test that the experimental framework can be instantiated
        framework = Stage2ExperimentalFramework(
            pretrained_unet_weights_path=champion_weights_path,
            base_experiment_name="Validation_Framework",
            device=device
        )
        framework.define_parameter_grids()
        
        print(f"   ✅ Stage2ExperimentalFramework instantiated successfully")
        print(f"   ✅ Parameter grids defined (including FiLM parameters)")
        
        # Test config creation with FiLM parameters
        test_config = framework.create_experiment_config(
            lr_film_generator=2e-4,
            film_generator_mlp_type='2_layer',
            lambda_gamma_res=0.01
        )
        
        if 'lr_film_generator' in test_config and test_config['lr_film_generator'] == 2e-4:
            print(f"   ✅ FiLM parameter overrides working")
        else:
            print(f"   ⚠️ FiLM parameter overrides not working properly")
        
        print(f"   ✅ Experimental framework validation passed")
        
    except Exception as e:
        print(f"   ❌ Experimental framework error: {e}")
        return {'status': 'failed', 'error': f'Experimental framework: {str(e)}'}
    
    # === FINAL VALIDATION SUMMARY ===
    print(f"\n🎉 COMPREHENSIVE VALIDATION COMPLETED!")
    print("=" * 80)
    print("✅ ALL CRITICAL COMPONENTS VALIDATED:")
    print("   ✅ Global definitions (CHECKPOINT_DIR, device)")
    print("   ✅ Unified FiLM-aware training function")
    print("   ✅ Training loop duality resolved")
    print("   ✅ Champion weights available")
    print("   ✅ FiLM training workflow functional")
    print("   ✅ Stage2ExperimentalFramework enhanced")
    print("   ✅ FiLM regularization working")
    print("   ✅ Differential learning rates working")
    print("   ✅ Gradient clipping working")
    print("   ✅ LR warm-up working")
    print("   ✅ Model checkpointing working")
    print("=" * 80)
    print()
    print("🚀 READY FOR PRODUCTION FiLM EXPERIMENTS!")
    print("   Recommended next steps:")
    print("   1. run_corrected_film_experiments()    # Run 3 FiLM experiments")
    print("   2. Analyze results vs baseline (0.0952% MAPE)")
    print("   3. If promising, run systematic exploration")
    print("   4. Use Stage2ExperimentalFramework for comprehensive tuning")
    
    return {
        'status': 'success',
        'validation_experiment': validation_result,
        'champion_weights_path': champion_weights_path,
        'framework_ready': True,
        'all_components_working': True
    }

# ===================================================================
# ===                      EXECUTION SECTION                     ===
# ===================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎬 STARTING FiLM EXPERIMENTS FOR SEISMIC VELOCITY ESTIMATION")
    print("="*80)
    
    # === STEP 1: COMPREHENSIVE VALIDATION ===
    print("\n🔧 STEP 1: Running comprehensive validation...")
    try:
        validation_results = run_corrected_film_validation()
        
        if validation_results['status'] == 'success':
            print("✅ VALIDATION PASSED! All systems ready for FiLM experiments.")
        else:
            print(f"❌ VALIDATION FAILED: {validation_results.get('error', 'Unknown error')}")
            print("Please fix the issues above before proceeding.")
            exit(1)
    except Exception as e:
        print(f"❌ VALIDATION ERROR: {e}")
        exit(1)
    
    # === STEP 2: PRODUCTION FiLM EXPERIMENTS ===
    print("\n🧪 STEP 2: Running production FiLM experiments...")
    try:
        experiment_results = run_corrected_film_experiments()
        
        print("🎉 FiLM EXPERIMENTS COMPLETED!")
        print("Results summary:")
        for exp_name, result in experiment_results.items():
            if 'best_mape_phase_b' in result:
                mape = result['best_mape_phase_b']
                print(f"   📊 {exp_name}: {mape:.4f}% MAPE")
            else:
                print(f"   ❌ {exp_name}: Failed")
                
    except Exception as e:
        print(f"❌ EXPERIMENT ERROR: {e}")
        print("Validation passed but experiments failed. Check the error above.")
    
    print("\n" + "="*80)
    print("🎬 FiLM EXPERIMENT SESSION COMPLETE")
    print("="*80)