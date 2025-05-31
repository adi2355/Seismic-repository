#!/usr/bin/env python3
"""
Simple wrapper to run SincGAT OPTIMIZED curriculum training

USAGE:
    python run_sincgat_training.py

This will run the latest SincGAT model with OPTIMIZED parameters:
- stride=1 (eliminates all aliasing)
- kernel_size=1001 (better low-frequency resolution)
- logarithmic filter spacing with 60 filters
- blackman window with superior side-lobe suppression
- hierarchical anti-aliased downsampling
"""

import os
import sys

def main():
    print("="*80)
    print("🎯 SINCGAT OPTIMIZED CURRICULUM TRAINING LAUNCHER")
    print("="*80)
    print("🚨 OPTIMIZED ARCHITECTURE IMPLEMENTED:")
    print("   ❌ OLD: stride=50 → severe aliasing, hamming window, 40 filters")
    print("   ✅ NEW: stride=1 + anti-aliased downsampling, 1001-point kernel, blackman window, 60 filters")
    print("   🎯 TARGET: Beat champion 0.0862% MAPE")
    print("="*80)
    
    try:
        # Import the function from the 898MODEL file
        # Note: The filename has a leading '0' which makes it a bit tricky to import
        import importlib.util
        
        # Load the module
        spec = importlib.util.spec_from_file_location(
            "model_898", "0_898model_speed_and_structure_starter_notebook.py"
        )
        model_898 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_898)
        
        # Check if the function exists
        if hasattr(model_898, 'run_sincgat_FIXED_curriculum_training'):
            print("✅ Found run_sincgat_FIXED_curriculum_training function")
            
            # Get the function
            run_function = getattr(model_898, 'run_sincgat_FIXED_curriculum_training')
            
            print("🚀 Starting SincGAT OPTIMIZED curriculum training...")
            print("   Using recommended parameters:")
            print("   - num_epochs: 50")
            print("   - curriculum_epochs: 10") 
            print("   - batch_size: 4")
            print()
            
            # Run the training with recommended parameters
            results = run_function(
                num_epochs=50,
                curriculum_epochs=10, 
                batch_size=4
            )
            
            print("="*80)
            print("✅ TRAINING COMPLETED!")
            if results:
                print(f"🏆 Results: {results}")
            print("="*80)
            
        else:
            print("❌ run_sincgat_FIXED_curriculum_training function not found!")
            print("Available functions:")
            for attr in dir(model_898):
                if 'sincgat' in attr.lower():
                    print(f"   - {attr}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the following files exist:")
        print("   - 0_898model_speed_and_structure_starter_notebook.py")
        print("   - phase2_experimental_framework.py")
        print("   - complete_sincgat_unet_integration.py")
        print("   - sincnet_seismic_encoder.py")
        print("   - seismic_gat_fusion.py")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("This could be due to:")
        print("   - Missing data files")
        print("   - CUDA/GPU issues")
        print("   - Memory constraints")
        print("   - Missing dependencies")

if __name__ == "__main__":
    main() 