#!/usr/bin/env python3
"""
Phase 2b Diagnostic Experiments - Execution Script

This script demonstrates how to run the systematic Phase 2b diagnostic experiments
to address the stagnation issue where Phase 2b (full fine-tuning) underperforms
Phase 2a (frontend frozen) training.

Usage:
    python run_diagnostic_experiments.py [--checkpoint_path PATH] [--mode MODE] [--epochs N]

Examples:
    # Run quick test (reduced epochs)
    python run_diagnostic_experiments.py --mode quick

    # Run focused ultra-low LR experiments
    python run_diagnostic_experiments.py --mode focused_lr --epochs 15

    # Run complete diagnostic suite
    python run_diagnostic_experiments.py --mode complete --epochs 30

    # Use specific checkpoint
    python run_diagnostic_experiments.py --checkpoint_path checkpoints/best_model.pth --mode complete
"""

import os
import sys
import argparse
from datetime import datetime

# Import our diagnostic framework
from phase2b_diagnostic_experiments import (
    run_phase2b_diagnostics,
    quick_phase2b_diagnostic_test,
    focused_ultra_low_lr_test,
    Phase2bDiagnosticFramework
)


def find_best_checkpoint():
    """
    Find the best available Phase 2a checkpoint to use as baseline.
    
    Returns:
        str: Path to the best checkpoint found
    """
    checkpoint_dir = "checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for FiLM checkpoints first (these are our Phase 2a targets)
    checkpoint_files = os.listdir(checkpoint_dir)
    
    # Priority order for checkpoint selection
    priority_patterns = [
        "Corrected_FiLM_2Layer_PhaseA_FrontendFrozen_best_mape.pth",  # Best from analysis
        "Corrected_FiLM_Linear_PhaseA_FrontendFrozen_best_mape.pth",  # Available
        "Validation_FiLM_PhaseA_FrontendFrozen_best_mape.pth",       # Backup
    ]
    
    for pattern in priority_patterns:
        if pattern in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, pattern)
            print(f"✅ Found checkpoint: {checkpoint_path}")
            return checkpoint_path
    
    # If no FiLM checkpoints, look for any .pth files
    pth_files = [f for f in checkpoint_files if f.endswith('.pth')]
    
    if pth_files:
        checkpoint_path = os.path.join(checkpoint_dir, pth_files[0])
        print(f"⚠️ Using fallback checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    raise FileNotFoundError(f"No suitable checkpoints found in {checkpoint_dir}")


def run_quick_diagnostic_test(checkpoint_path: str):
    """Run quick diagnostic test with reduced epochs."""
    print("🧪 RUNNING QUICK DIAGNOSTIC TEST")
    print("="*50)
    print("This is a validation run with reduced epochs to test the framework.")
    
    try:
        results = run_phase2b_diagnostics(
            base_checkpoint_path=checkpoint_path,
            experiment_name="Quick_Phase2b_Diagnostic_Test",
            num_epochs_phase2b=5,   # Reduced for quick testing
            num_epochs_extended_2a=3
        )
        
        print("\n✅ Quick diagnostic test completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Quick diagnostic test failed: {e}")
        return None


def run_focused_lr_experiments(checkpoint_path: str, num_epochs: int = 15):
    """Run focused ultra-low learning rate experiments."""
    print("🎯 RUNNING FOCUSED ULTRA-LOW LR EXPERIMENTS")
    print("="*50)
    print("Testing hypothesis: U-Net LR too high for FiLM-conditioned pre-trained model")
    
    try:
        framework = Phase2bDiagnosticFramework(
            base_checkpoint_path=checkpoint_path,
            experiment_base_name="Focused_UltraLow_LR_Diagnostic"
        )
        
        results = framework.run_ultra_low_unet_lr_experiments(num_epochs=num_epochs)
        
        print(f"\n✅ Focused LR experiments completed!")
        print(f"   Experiments run: {len(results)}")
        
        # Print summary of results
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            best_mape = min([r['best_mape'] for r in successful_results])
            print(f"   Best MAPE achieved: {best_mape:.6f}%")
            if best_mape < 0.0893:
                print(f"   🎉 BREAKTHROUGH! Improved beyond Phase 2a baseline!")
        
        return results
        
    except Exception as e:
        print(f"❌ Focused LR experiments failed: {e}")
        return None


def run_complete_diagnostic_suite(checkpoint_path: str, 
                                num_epochs_phase2b: int = 30,
                                num_epochs_extended_2a: int = 20):
    """Run the complete diagnostic experiment suite."""
    print("🚀 RUNNING COMPLETE DIAGNOSTIC SUITE")
    print("="*80)
    print("This will run all diagnostic experiments to address Phase 2b stagnation:")
    print("  1. Ultra-low U-Net learning rates (3 experiments)")
    print("  2. FiLM regularization tuning (3 experiments)")  
    print("  3. LR scheduler optimization (3 experiments)")
    print("  4. Extended Phase 2a training (1 experiment)")
    print(f"  Total: ~10 experiments × {num_epochs_phase2b} epochs each")
    print(f"  Estimated time: Several hours on GPU")
    
    confirm = input("\nProceed with complete diagnostic suite? (y/N): ")
    if confirm.lower() != 'y':
        print("Diagnostic suite cancelled.")
        return None
    
    try:
        results = run_phase2b_diagnostics(
            base_checkpoint_path=checkpoint_path,
            experiment_name="Complete_Phase2b_Diagnostic_Suite",
            num_epochs_phase2b=num_epochs_phase2b,
            num_epochs_extended_2a=num_epochs_extended_2a
        )
        
        print("\n🎉 COMPLETE DIAGNOSTIC SUITE FINISHED!")
        return results
        
    except Exception as e:
        print(f"❌ Complete diagnostic suite failed: {e}")
        return None


def run_custom_single_experiment(checkpoint_path: str, config: dict, num_epochs: int = 20):
    """Run a single custom diagnostic experiment."""
    print("🔬 RUNNING CUSTOM SINGLE EXPERIMENT")
    print("="*50)
    print(f"Configuration: {config}")
    
    try:
        framework = Phase2bDiagnosticFramework(
            base_checkpoint_path=checkpoint_path,
            experiment_base_name="Custom_Diagnostic_Experiment"
        )
        
        result = framework.run_single_diagnostic_experiment(config, num_epochs)
        
        if result.get('success', False):
            print(f"\n✅ Custom experiment completed!")
            print(f"   Best MAPE: {result['best_mape']:.6f}%")
        else:
            print(f"\n❌ Custom experiment failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Custom experiment failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 2b diagnostic experiments to address training stagnation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to Phase 2a checkpoint (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'focused_lr', 'complete', 'custom'],
        default='quick',
        help='Diagnostic mode to run'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (defaults vary by mode)'
    )
    
    parser.add_argument(
        '--lr_unet',
        type=float,
        default=1e-6,
        help='U-Net learning rate for custom experiments'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiment_results',
        help='Directory for output results'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("🧪 PHASE 2B DIAGNOSTIC EXPERIMENTS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    
    # Find checkpoint
    try:
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
        else:
            checkpoint_path = find_best_checkpoint()
            
        print(f"Using checkpoint: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Error finding checkpoint: {e}")
        sys.exit(1)
    
    # Run experiments based on mode
    results = None
    
    if args.mode == 'quick':
        results = run_quick_diagnostic_test(checkpoint_path)
        
    elif args.mode == 'focused_lr':
        epochs = args.epochs or 15
        results = run_focused_lr_experiments(checkpoint_path, epochs)
        
    elif args.mode == 'complete':
        epochs_2b = args.epochs or 30
        epochs_2a = max(15, epochs_2b // 2)  # Extended 2a gets fewer epochs
        results = run_complete_diagnostic_suite(checkpoint_path, epochs_2b, epochs_2a)
        
    elif args.mode == 'custom':
        epochs = args.epochs or 20
        custom_config = {
            'experiment_id': 'custom_ultra_low_lr',
            'lr_unet': args.lr_unet,
            'lr_frontend': 5e-5,
            'lr_film_generator': 1e-4,
            'film_generator_type': '2_layer',
            'description': f'Custom ultra-low U-Net LR: {args.lr_unet}'
        }
        results = run_custom_single_experiment(checkpoint_path, custom_config, epochs)
    
    # Print final summary
    if results:
        print("\n" + "=" * 80)
        print("📊 DIAGNOSTIC EXPERIMENTS SUMMARY")
        print("=" * 80)
        
        if isinstance(results, dict) and 'diagnostic_suite_summary' in results:
            # Complete suite results
            summary = results['diagnostic_suite_summary']
            print(f"🎯 Target: 0.0893% MAPE (Phase 2a baseline)")
            print(f"🏆 Best achieved: {summary['best_overall_mape']:.6f}% MAPE")
            
            if summary['improvement_achieved']:
                print(f"✅ BREAKTHROUGH ACHIEVED!")
                print(f"🎉 Improvement: {summary['improvement_amount']:.6f}% MAPE reduction")
            else:
                deficit = summary['best_overall_mape'] - 0.0893
                print(f"⚠️  Target not reached: {deficit:.6f}% above baseline")
                
        elif isinstance(results, list):
            # Individual experiment results
            successful = [r for r in results if r.get('success', False)]
            if successful:
                best_mape = min([r['best_mape'] for r in successful])
                print(f"🏆 Best MAPE: {best_mape:.6f}%")
                if best_mape < 0.0893:
                    print(f"✅ BREAKTHROUGH! Improved beyond baseline!")
        
        print(f"\n💾 Results saved to: {args.output_dir}")
    else:
        print("\n❌ No results to report")
        sys.exit(1)
    
    print("\n🎉 Diagnostic experiments completed!")


if __name__ == "__main__":
    main() 