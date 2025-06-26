#!/usr/bin/env python3
"""
CORRECTED INFERENCE PIPELINE FOR SPEED AND STRUCTURE CHALLENGE
==============================================================

This pipeline creates predictions using the exact champion model parameters 
found in another_copy_of_main_898of_0_898model_speed_and_structure_starter_notebook.py

CRITICAL: All parameters are based on actual file analysis, with citations.

CITATIONS SUMMARY:
- Champion model: cfg_06_plateau_to_cosine (lines 12105-12115)
- FiLM MLP type: '2_layer' from BASE_CONFIG (line 12094)
- Model architecture: CompleteSincGAT_UNet (lines 3515-3575)
- Data preprocessing: SeismicDataset (lines 398-440)
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import math

# ============================================================================
# STEP 1: ENVIRONMENT SETUP
# ============================================================================

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Operating on device: {device}")

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ============================================================================
# STEP 2: EXACT PARAMETER DEFINITIONS (WITH CITATIONS)
# ============================================================================

# CITATION: Lines 12091-12098 in another_copy_of_main_898of_0_898model_speed_and_structure_starter_notebook.py
# BASE_CONFIG shows: 'film_generator_mlp_type': '2_layer'
CHAMPION_FILM_MLP_TYPE = '2_layer'  # ⚠️ CRITICAL: Champion uses 2_layer, NOT linear

# CITATION: Lines 3515-3575 in another_copy_of_main_898of_0_898model_speed_and_structure_starter_notebook.py
# CompleteSincGAT_UNet.__init__ default parameters
CHAMPION_MODEL_PARAMS = {
    # Dataset-specific parameters - CITATION: Line 3520-3524
    'sample_rate': 10001,     # Hz - CRITICAL: Must match actual data sampling rate
    'num_receivers': 31,      # Number of receivers per shot
    'time_samples': 10001,    # Time samples per receiver
    'num_shots': 5,           # Number of shots per sample
    
    # SincNet parameters (OPTIMIZED SETTINGS) - CITATION: Line 3525-3533
    'sinc_out_channels': 60,        # Increased from 40 to 60 (optimal for log spacing)
    'sinc_kernel_size': 1001,       # Increased from 251 to 1001 (better low-freq resolution)
    'sinc_stride': 1,               # CRITICAL: Use 1 to eliminate aliasing (was 10)
    'sinc_min_low_hz': 40,          # Lowered from 80 to 40 (captures more low frequencies)
    'sinc_max_learnable_hz': 1000,  # Upper limit at 1000Hz (where coherent signal ends)
    'sinc_min_band_hz': 10,         # Minimum bandwidth for a filter
    'sinc_window_func': 'blackman', # Changed from hamming to blackman (better side-lobe suppression)
    'sinc_init_type': 'logarithmic',# Added logarithmic spacing (better allocation across spectrum)
    'shot_embedding_dim': 128,      # Embedding dimension for each shot
    
    # GAT parameters - CITATION: Line 3535-3541
    'gat_hidden_per_head': 32,
    'gat_num_heads': 4,
    'gat_layers': 1,
    'gat_dropout_feat': 0.3,
    'gat_dropout_attn': 0.2,
    'fused_embedding_dim': 128,
    
    # U-Net parameters - CITATION: Line 3542-3545
    'n_unet_output_channels': 1,
    'unet_bilinear': True,
    'unet_bottleneck_channels': 512,
    
    # FiLM parameters (CRITICAL) - CITATION: Line 3546-3550
    'film_context_dim': 128,         # Should match fused_embedding_dim
    'film_target_channels': 512,     # Should match unet_bottleneck_channels
    'film_generator_mlp_type': CHAMPION_FILM_MLP_TYPE,  # ⚠️ CRITICAL: '2_layer' from BASE_CONFIG
    'film_mlp_hidden_dim': 256,      # For '2_layer' type
}

# CITATION: Lines 398-440 in another_copy_of_main_898of_0_898model_speed_and_structure_starter_notebook.py
# SeismicDataset.__init__ and __getitem__ methods
DATA_PREPROCESSING_PARAMS = {
    'source_coords': [1, 75, 150, 225, 300],    # Line 408: self.source_coords = [1, 75, 150, 225, 300]
    'epsilon': 1e-8,                            # Line 412: self.epsilon = 1e-8 # For safe division in normalization
    'input_dtype': torch.float32,               # Line 399: input_dtype=torch.float32
    'target_dtype': torch.float32,              # Line 399: target_dtype=torch.float32
}

# Path configuration
CHAMPION_MODEL_PATH = "/home/adi235/colab/checkpoints/cfg_06_plateau_to_cosine_PhaseB_FiLMFinetune_best_mape.pth"
TEST_DATA_DIR = "/home/adi235/colab/data/test"  # Updated to your local path
SUBMISSION_FILENAME = "speed_and_structure_submission.npz"

print("✅ Parameters loaded with citations:")
print(f"   Champion FiLM MLP Type: {CHAMPION_FILM_MLP_TYPE} (from BASE_CONFIG line 12094)")
print(f"   SincNet kernel size: {CHAMPION_MODEL_PARAMS['sinc_kernel_size']} (line 3527)")
print(f"   SincNet stride: {CHAMPION_MODEL_PARAMS['sinc_stride']} (line 3528 - CRITICAL anti-aliasing fix)")

# ============================================================================
# STEP 3: IMPORT EXISTING MODEL COMPONENTS
# ============================================================================

# Try to import from your existing module files
try:
    # CITATION: These should match your existing module files
    from complete_sincgat_unet_integration import CompleteSincGAT_UNet
    from utils import create_submission
    print("✅ Successfully imported CompleteSincGAT_UNet and create_submission from existing modules")
    
    # Test model instantiation with correct parameters
    print("\n🔧 Testing model instantiation with exact champion parameters...")
    test_model = CompleteSincGAT_UNet(
        **CHAMPION_MODEL_PARAMS
    )
    total_params = sum(p.numel() for p in test_model.parameters())
    print(f"   ✅ Model instantiated successfully with {total_params:,} parameters")
    del test_model  # Free memory
    
except ImportError as e:
    print(f"❌ Could not import from existing modules: {e}")
    print("   Make sure your module files are in the same directory or in the Python path")
    sys.exit(1)

# ============================================================================
# STEP 4: DATA PREPROCESSING FUNCTION (EXACT MATCH TO TRAINING)
# ============================================================================

def preprocess_seismic_sample(sample_folder_path, source_coords, epsilon=1e-8):
    """
    Preprocess seismic data exactly as done in training.
    
    CITATION: Lines 413-440 in another_copy_of_main_898of_0_898model_speed_and_structure_starter_notebook.py
    SeismicDataset.__getitem__ method
    """
    
    # 1. Load and preprocess 5 input seismic shot records
    stacked_seismic_data = []
    for s_coord in source_coords:
        file_path = os.path.join(sample_folder_path, f"receiver_data_src_{s_coord}.npy")
        shot_data = np.load(file_path)  # Loads as float32 by default
        
        # Per-shot normalization (standardization: zero mean, unit variance)
        # CITATION: Lines 420-426 in the notebook
        mean = np.mean(shot_data)
        std = np.std(shot_data)
        normalized_shot_data = (shot_data - mean) / (std + epsilon)
        
        stacked_seismic_data.append(normalized_shot_data)
    
    # Stack along a new "channel" dimension (first dimension for PyTorch Conv2D)
    # Resulting shape: (num_shots, time_steps, num_receivers) -> (5, 10001, 31)
    # CITATION: Line 429 in the notebook
    stacked_seismic_data_np = np.stack(stacked_seismic_data, axis=0)
    
    # Convert to PyTorch tensor with specified dtype
    # CITATION: Line 438 in the notebook
    seismic_tensor = torch.tensor(stacked_seismic_data_np, dtype=torch.float32)
    
    return seismic_tensor

# ============================================================================
# STEP 5: INFERENCE FUNCTION
# ============================================================================

def generate_predictions_for_test_set(model, test_paths, device, submission_path):
    """
    Generate predictions for the test set using the exact preprocessing pipeline.
    
    CITATION: Based on SeismicDataset preprocessing (lines 398-440) and 
    submission requirements (lines 13130-13170)
    """
    
    model.eval()  # Ensure model is in evaluation mode
    print(f"🚀 Starting inference on {len(test_paths)} test samples...")
    
    # Clear any existing submission file
    if os.path.exists(submission_path):
        os.remove(submission_path)
    
    for i, sample_folder_path in enumerate(tqdm(test_paths, desc="Processing Test Samples")):
        sample_id = os.path.basename(sample_folder_path)
        
        try:
            # 1. Preprocess the sample (exact match to training)
            seismic_tensor = preprocess_seismic_sample(
                sample_folder_path, 
                DATA_PREPROCESSING_PARAMS['source_coords'],
                DATA_PREPROCESSING_PARAMS['epsilon']
            )
            
            # Add batch dimension and move to device
            # Input shape: (1, 5, 10001, 31) - matches CompleteSincGAT_UNet expected input
            input_tensor = seismic_tensor.unsqueeze(0).to(device)
            
            # 2. Generate prediction
            with torch.no_grad():
                output_tensor = model(input_tensor)  # Expected output: (1, 1, 300, 1259)
            
            # 3. Postprocess prediction
            # Squeeze batch and channel dimensions, move to CPU, convert to NumPy
            prediction_np = output_tensor.squeeze().cpu().numpy()  # Shape: (300, 1259)
            
            # 4. CRITICAL: Cast to numpy.float64 for submission
            # CITATION: Line 13134 in notebook - "must be of type numpy.float64"
            prediction_for_submission = prediction_np.astype(np.float64)
            
            # 5. Validate prediction shape
            if prediction_for_submission.shape != (300, 1259):
                print(f"❌ ERROR: Prediction shape {prediction_for_submission.shape} != (300, 1259) for sample {sample_id}")
                continue
            
            # 6. Save to .npz file using create_submission
            create_submission(sample_id, prediction_for_submission, submission_path)
            
            # Progress update every 10 samples
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(test_paths)} samples")
                
        except Exception as e:
            print(f"❌ ERROR processing sample {sample_id}: {e}")
            continue
    
    print(f"✅ Inference complete. Submission file saved to '{submission_path}'")

# ============================================================================
# STEP 6: MAIN INFERENCE EXECUTION
# ============================================================================

def main():
    """
    Main inference function that coordinates the entire process.
    """
    
    print("🎯 STARTING CORRECTED INFERENCE PIPELINE")
    print("=" * 60)
    
    # 1. Verify champion model path
    if not os.path.exists(CHAMPION_MODEL_PATH):
        print(f"❌ ERROR: Champion model not found at {CHAMPION_MODEL_PATH}")
        # List available checkpoints
        checkpoint_dir = os.path.dirname(CHAMPION_MODEL_PATH)
        if os.path.exists(checkpoint_dir):
            print(f"Available checkpoints in {checkpoint_dir}:")
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"   - {f}")
        return False
    
    # 2. Load champion model
    print(f"📥 Loading champion model from: {CHAMPION_MODEL_PATH}")
    try:
        # Instantiate model with EXACT parameters
        champion_model = CompleteSincGAT_UNet(**CHAMPION_MODEL_PARAMS).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(CHAMPION_MODEL_PATH, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            champion_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"   ✅ Model loaded from checkpoint dict")
            if 'best_mape' in checkpoint:
                print(f"   📊 Checkpoint MAPE: {checkpoint['best_mape']:.4f}%")
        else:
            # Direct state_dict
            champion_model.load_state_dict(checkpoint, strict=True)
            print(f"   ✅ Model loaded from direct state_dict")
        
        champion_model.eval()
        print(f"   ✅ Model in evaluation mode")
        
    except Exception as e:
        print(f"❌ ERROR loading champion model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Get test sample paths
    print(f"📂 Loading test samples from: {TEST_DATA_DIR}")
    test_sample_paths = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*")))
    
    if not test_sample_paths:
        print(f"❌ ERROR: No test samples found in {TEST_DATA_DIR}")
        return False
    
    print(f"   ✅ Found {len(test_sample_paths)} test samples")
    if len(test_sample_paths) != 150:
        print(f"   ⚠️  WARNING: Expected 150 samples, found {len(test_sample_paths)}")
    
    # 4. Generate predictions
    submission_path = os.path.join(os.path.dirname(CHAMPION_MODEL_PATH), SUBMISSION_FILENAME)
    print(f"💾 Submission will be saved to: {submission_path}")
    
    generate_predictions_for_test_set(
        model=champion_model,
        test_paths=test_sample_paths,
        device=device,
        submission_path=submission_path
    )
    
    # 5. Validate generated submission
    print(f"\n🔍 Validating generated submission file...")
    try:
        submission_data = np.load(submission_path)
        file_keys = list(submission_data.files)
        
        print(f"   ✅ Submission file contains {len(file_keys)} arrays")
        
        if len(file_keys) > 0:
            # Check first array
            sample_array = submission_data[file_keys[0]]
            print(f"   ✅ Sample array shape: {sample_array.shape}")
            print(f"   ✅ Sample array dtype: {sample_array.dtype}")
            
            # Validation checks
            if len(file_keys) == 150:
                print(f"   ✅ Correct number of arrays (150)")
            else:
                print(f"   ❌ Wrong number of arrays: {len(file_keys)} (expected 150)")
            
            if sample_array.shape == (300, 1259):
                print(f"   ✅ Correct array shape (300, 1259)")
            else:
                print(f"   ❌ Wrong array shape: {sample_array.shape} (expected (300, 1259))")
            
            if sample_array.dtype == np.float64:
                print(f"   ✅ Correct dtype (float64)")
            else:
                print(f"   ❌ Wrong dtype: {sample_array.dtype} (expected float64)")
        
        submission_data.close()
        
    except Exception as e:
        print(f"❌ ERROR validating submission: {e}")
        return False
    
    print(f"\n🎉 INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📁 Submission file: {submission_path}")
    print(f"   Ready for upload to competition platform")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
