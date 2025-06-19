# ===================================================================
# ENHANCED COLAB EXECUTION CELL WITH COMPREHENSIVE CHECKPOINTING
# ===================================================================
import os
import sys
import torch
import numpy as np
from datetime import datetime
import shutil
import json

# 1. --- Setup and Environment ---
print("--- 1. Setting up Environment ---")
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Define Project Paths
DRIVE_ROOT = '/content/drive/MyDrive'
PROJECT_NAME = 'colab' # Assuming your project is in a folder named 'colab'
PROJECT_PATH = os.path.join(DRIVE_ROOT, PROJECT_NAME)

# Define key subdirectories
# Global CHECKPOINT_DIR is needed by the training function for saving models
CHECKPOINT_DIR = os.path.join(PROJECT_PATH, 'checkpoints')
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Add project path to system path to allow imports
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

print(f"Project Path: {PROJECT_PATH}")

# Load Project Code
print("\n--- 2. Importing Project Modules ---")
%load_ext autoreload
%autoreload 2

# This assumes your main script is named this way. Adjust if necessary.
from copy_of_main_898of_0_898model_speed_and_structure_starter_notebook import *

SAM_REPO_PATH = '/content/sam'
if not os.path.exists(SAM_REPO_PATH):
    print("⏳ Cloning the SAM repository...")
    !git clone https://github.com/davda54/sam.git {SAM_REPO_PATH}
else:
    print("✅ SAM repository already cloned.")

# Add the repository root to Python path
if SAM_REPO_PATH not in sys.path:
    sys.path.append(SAM_REPO_PATH)
    print(f"✅ Added {SAM_REPO_PATH} to system path.")

# --- Import SAM components (CORRECTED) ---
try:
    # Import directly from sam.py (not from sam.sam)
    from sam import SAM
    print("✅ SAM optimizer imported successfully.")

    # Manual implementation of BatchNorm utilities (not in repository)
    def enable_running_stats(model):
        """Enable BatchNorm running statistics tracking"""
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.track_running_stats = True

    def disable_running_stats(model):
        """Disable BatchNorm running statistics tracking"""
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.track_running_stats = False

    print("✅ BatchNorm utilities implemented successfully.")
    SAM_AVAILABLE = True

except ImportError as e:
    print(f"❌ SAM import failed with error: {e}")
    SAM_AVAILABLE = False

    # Dummy functions to prevent errors
    def enable_running_stats(model):
        pass

    def disable_running_stats(model):
        pass

if SAM_AVAILABLE:
    # Make SAM_AVAILABLE globally accessible to the imported module
    import copy_of_main_898of_0_898model_speed_and_structure_starter_notebook as main_module
    main_module.SAM_AVAILABLE = True
    main_module.SAM = SAM
    main_module.enable_running_stats = enable_running_stats
    main_module.disable_running_stats = disable_running_stats
    print("✅ SAM variables passed to training module")


# Add this after: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n--- TF32 Precision Check ---")
if torch.cuda.is_available():
    # Check current TF32 status
    tf32_disabled = verify_tf32_status()
    
    if not tf32_disabled:
        print("🔧 Forcing TF32 disable...")
        force_disable_tf32()
    
    # Double-check after any changes
    print("\n🔍 FINAL TF32 STATUS:")
    verify_tf32_status()
    
    # Also check your GPU type
    gpu_name = torch.cuda.get_device_name(0)
    print(f"   GPU: {gpu_name}")
    if "A100" in gpu_name:
        print("   📊 A100 detected - TF32 disable is critical for precision")
else:
    print("   CPU mode - TF32 not applicable")


# Set Random Seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Random seed set to {SEED}")

print("\n--- 3. Defining Experimental Configurations ---")
base_checkpoint_name = '/content/Phase2a_Only_PhaseA_FrontendFrozen_best_mape (1).pth'
# Correctly construct the full path
base_checkpoint_full_path = '/content/Phase2a_Only_PhaseA_FrontendFrozen_best_mape (1).pth'


if not os.path.exists(base_checkpoint_full_path):
    # More robust check for the file path
    raise FileNotFoundError(f"CRITICAL: Base checkpoint not found. Expected at: {base_checkpoint_full_path}")
else:
    print(f"✅ Base checkpoint found: {base_checkpoint_full_path}")

# --- FIX APPLIED HERE: Define BASE_CONFIG ---
# This dictionary holds all parameters shared across experiments.
BASE_CONFIG = {
    'num_epochs_phase_b': 50,      # <-- Total epochs remains 15
    'curriculum_simple_epochs': 0, # <-- ADD THIS LINE: 5 epochs for stabilization
    'batch_size': 5,
    'film_generator_mlp_type': '2_layer',
    'lr_frontend_phase_b': 5e-5,
    'weight_decay': 0.01,
    'weight_decay_film': 1e-3,
}

# Now, define each experiment by inheriting from BASE_CONFIG and overriding specific keys.
configurations = [
    # Experiment 1: Baseline
          {
            **BASE_CONFIG,
            'config_id': 'cfg_06_plateau_to_cosine',
            'lr_unet': 1e-5, 
            'lr_film': 1e-4,
            'lambda_gamma': 0.003, 
            'lambda_beta': 0.0008, 
            'scheduler': 'PlateauToCosine',          # 🔑 HYBRID SCHEDULER
            'plateau_patience': 4,                   # Patience for plateau phase
            'plateau_factor': 0.85,                  # Factor for LR reduction (0.8 = 20% reduction)
            'plateau_threshold': 0.001,              # Threshold for improvement detection
            'lr_switch_threshold': 2.5e-05,          # LR below which we switch to cosine
            'cosine_T_max': 15,                       # Epochs for final cosine polishing
            'cosine_eta_min': 1.14e-05,                  # Final minimum LR
            'threshold_group_index': 0,
            'warmup_steps': 800,                    
            'gradient_clip_film': 0.8,
            'gradient_clip_others': 4.5, 
            'use_grad_clipping': True,
            'use_sam': False,                        # 🔑 Fast AdamW training
            'use_gc':False,                          # 🔑 Gradient Centralization for stability
            'SAM_AVAILABLE': SAM_AVAILABLE,          # 🔑 REQUIRED: SAM integration
            'SAM_CLASS': SAM,                        # 🔑 REQUIRED: SAM class
            'enable_running_stats': enable_running_stats,   # 🔑 REQUIRED: BatchNorm utils
            'disable_running_stats': disable_running_stats, # 🔑 REQUIRED: BatchNorm utils
            'notes': 'Hybrid Plateau→Cosine scheduler: ReduceLROnPlateau exploration + CosineAnnealingLR exploitation for smooth convergence.'
        },
]

print(f"✅ Defined {len(configurations)} experiments. Each will run for {BASE_CONFIG['num_epochs_phase_b']} epochs.")

# 4. --- The Experiment Execution Loop ---
print("\n--- 4. Starting Experimental Suite ---")
all_results = []

for i, config in enumerate(configurations):
    print(f"\n▶️  Running Experiment {i+1}/{len(configurations)}: {config['config_id']}")
    print(f"   Notes: {config['notes']}")

    try:
        result = run_stage2_film_training(
            pretrained_unet_weights_path=base_checkpoint_full_path,
            config=config
        )

        final_mape = result.get('best_mape_phase_b', float('inf'))
        final_checkpoint = result.get('final_model_path')

        print(f"✅ Experiment {config['config_id']} finished. Final MAPE: {final_mape:.4f}%")

        # 🔄 ENHANCED MODEL SAVING & CHECKPOINTING
        if 'error' not in result:
            config_id = config['config_id']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create dedicated experiment directory
            experiment_dir = os.path.join(RESULTS_PATH, f"{config_id}_{timestamp}")
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Save the best model with metadata
            best_model_path = os.path.join(experiment_dir, f"{config_id}_best_model.pth")
            
            # If the training returned a model path, copy it
            if result.get('final_model_path') and os.path.exists(result['final_model_path']):
                shutil.copy2(result['final_model_path'], best_model_path)
                print(f"   💾 Best model saved: {best_model_path}")
            
            # Save experiment metadata
            metadata = {
                'config': config,
                'results': {
                    'best_mape_phase_b': result.get('best_mape_phase_b'),
                    'best_mape_phase_a': result.get('best_mape_phase_a'),
                    'training_history': result.get('training_history', {}),
                    'final_model_path': best_model_path
                },
                'timestamp': timestamp,
                'experiment_duration': result.get('training_time', 'N/A')
            }
            
            # Save metadata as JSON
            metadata_path = os.path.join(experiment_dir, f"{config_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                # Convert any non-serializable objects to strings
                serializable_metadata = {}
                for key, value in metadata.items():
                    try:
                        json.dumps(value)  # Test if serializable
                        serializable_metadata[key] = value
                    except:
                        serializable_metadata[key] = str(value)
                json.dump(serializable_metadata, f, indent=2)
            
            print(f"   📄 Metadata saved: {metadata_path}")
            
            # Update result with full paths
            result['experiment_dir'] = experiment_dir
            result['best_model_path'] = best_model_path
            result['metadata_path'] = metadata_path

        all_results.append({'config': config, 'result': result})

    except Exception as e:
        print(f"❌ Experiment {config['config_id']} FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_dir = os.path.join(RESULTS_PATH, f"FAILED_{config['config_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(error_dir, exist_ok=True)
        
        error_info = {
            'config': config,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(error_dir, 'error_log.json'), 'w') as f:
            json.dump(error_info, f, indent=2, default=str)
            
        print(f"   📁 Error log saved: {error_dir}")
        all_results.append({'config': config, 'result': {'error': str(e), 'error_dir': error_dir}})

# 5. --- Final Summary ---
print("\n🏁 Experimental Suite Finished! 🏁")
print("="*70)
print("📊 Final Results Summary:")
for res in all_results:
    cfg_id = res['config']['config_id']
    if 'error' in res['result']:
        print(f"  - {cfg_id:25}: FAILED")
    else:
        mape = res['result'].get('best_mape_phase_b', 'N/A')
        print(f"  - {cfg_id:25}: {mape:.4f}% MAPE")

best_run = min([r for r in all_results if 'error' not in r['result']],
               key=lambda x: x['result'].get('best_mape_phase_b', float('inf')),
               default=None)

if best_run:
    print("\n🏆 Best Performing Configuration:")
    print(f"   ID: {best_run['config']['config_id']}")
    print(f"   MAPE: {best_run['result']['best_mape_phase_b']:.4f}%")

# 6. --- COMPREHENSIVE FINAL CHECKPOINTING ---
print("\n💾 FINAL COMPREHENSIVE CHECKPOINTING")
print("="*70)

# Create a master results directory
master_results_dir = os.path.join(RESULTS_PATH, f"experimental_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(master_results_dir, exist_ok=True)

# Save complete experimental suite results
suite_summary = {
    'total_experiments': len(configurations),
    'successful_experiments': len([r for r in all_results if 'error' not in r['result']]),
    'failed_experiments': len([r for r in all_results if 'error' in r['result']]),
    'best_overall_mape': best_run['result']['best_mape_phase_b'] if best_run else None,
    'best_config_id': best_run['config']['config_id'] if best_run else None,
    'experiment_start_time': datetime.now().isoformat(),
    'base_checkpoint': base_checkpoint_full_path,
    'all_results': []
}

# Add detailed results for each experiment
for res in all_results:
    cfg_id = res['config']['config_id']
    if 'error' in res['result']:
        suite_summary['all_results'].append({
            'config_id': cfg_id,
            'status': 'FAILED',
            'error': res['result']['error'],
            'error_dir': res['result'].get('error_dir')
        })
    else:
        suite_summary['all_results'].append({
            'config_id': cfg_id,
            'status': 'SUCCESS',
            'final_mape': res['result'].get('best_mape_phase_b'),
            'phase_a_mape': res['result'].get('best_mape_phase_a'),
            'experiment_dir': res['result'].get('experiment_dir'),
            'best_model_path': res['result'].get('best_model_path')
        })

# Save master summary
master_summary_path = os.path.join(master_results_dir, 'experimental_suite_summary.json')
with open(master_summary_path, 'w') as f:
    json.dump(suite_summary, f, indent=2, default=str)

print(f"📊 Master summary saved: {master_summary_path}")

# Copy the best model to an easily accessible location
if best_run and best_run['result'].get('best_model_path'):
    best_model_final_path = os.path.join(CHECKPOINT_DIR, f"BEST_MODEL_{best_run['config']['config_id']}.pth")
    shutil.copy2(best_run['result']['best_model_path'], best_model_final_path)
    print(f"🏆 Best model copied to: {best_model_final_path}")

# Create a quick access summary text file
quick_summary_path = os.path.join(master_results_dir, 'QUICK_SUMMARY.txt')
with open(quick_summary_path, 'w') as f:
    f.write("EXPERIMENTAL SUITE QUICK SUMMARY\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Total Experiments: {suite_summary['total_experiments']}\n")
    f.write(f"Successful: {suite_summary['successful_experiments']}\n")
    f.write(f"Failed: {suite_summary['failed_experiments']}\n\n")
    
    if best_run:
        f.write(f"BEST RESULT:\n")
        f.write(f"Config ID: {best_run['config']['config_id']}\n")
        f.write(f"MAPE: {best_run['result']['best_mape_phase_b']:.4f}%\n")
        f.write(f"Model Path: {best_run['result'].get('best_model_path', 'N/A')}\n\n")
    
    f.write("ALL RESULTS:\n")
    for res in all_results:
        cfg_id = res['config']['config_id']
        if 'error' in res['result']:
            f.write(f"  {cfg_id}: FAILED\n")
        else:
            mape = res['result'].get('best_mape_phase_b', 'N/A')
            f.write(f"  {cfg_id}: {mape:.4f}% MAPE\n")

print(f"📄 Quick summary saved: {quick_summary_path}")
print(f"\n📁 All results archived in: {master_results_dir}")
print("💾 Checkpointing complete!")

# Print final paths for easy access
print("\n🔗 QUICK ACCESS PATHS:")
print(f"📊 Master Summary: {master_summary_path}")
print(f"📄 Quick Summary: {quick_summary_path}")
if best_run:
    print(f"🏆 Best Model: {CHECKPOINT_DIR}/BEST_MODEL_{best_run['config']['config_id']}.pth")
print(f"📁 All Results: {master_results_dir}") 