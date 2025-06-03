#!/usr/bin/env python3
"""
Example Usage: Comprehensive LR/Scheduler Exploration
=====================================================

This script demonstrates how to use the improved Stage2ExperimentalFramework
to run comprehensive learning rate and scheduler experiments.

Based on your log showing Stage2_Systematic_lr_schedule_1 achieving 0.0931%,
this shows both focused and comprehensive approaches.
"""

import sys
import os

# Add the main file to path if running as standalone script
sys.path.append('/home/adi235/colab')

# Import the enhanced experimental framework
from main_898of_0_898model_speed_and_structure_starter_notebook import (
    run_systematic_stage2_experiments,
    run_focused_lr_tuning,
    run_comprehensive_lr_exploration,
    run_balanced_exploration,
    analyze_experimental_scope,
    Stage2ExperimentalFramework
)

def main():
    # Your Stage 1 pretrained weights (update this path as needed)
    pretrained_weights_path = "/content/checkpoints/Stage1_FullRun_UNet_Asymmetric_best_mape.pth"
    
    # Verify weights exist
    if not os.path.exists(pretrained_weights_path):
        print(f"❌ Weights file not found: {pretrained_weights_path}")
        print("Please update the path to your Stage 1 weights file.")
        return
    
    print("🚀 ENHANCED EXPERIMENTAL FRAMEWORK DEMO")
    print("="*60)
    
    # === STEP 1: ANALYZE SCOPE ===
    print("\n📊 STEP 1: Analyzing experimental scope...")
    scope = analyze_experimental_scope(pretrained_weights_path)
    
    print(f"\nKey insights:")
    print(f"• LR/Scheduler experiments possible: {scope['lr_schedule']}")
    print(f"• GAT configuration experiments: {scope['gat_config']}")
    print(f"• Combined scope would be: {scope['combined_total']:,} (infeasible!)")
    
    # === STEP 2: CHOOSE STRATEGY ===
    print(f"\n🎯 STEP 2: Choose your experimental strategy")
    print(f"Based on your 0.0931% result, here are the options:")
    print(f"")
    print(f"1. FOCUSED (Recommended): 5 experiments around your champion config")
    print(f"2. COMPREHENSIVE: All 99 LR/scheduler combinations")  
    print(f"3. BALANCED: Sample 20 experiments across categories")
    print(f"4. QUICK: 5 random LR experiments for testing")
    
    choice = input("\nEnter choice (1-4) or 'q' to quit: ").strip()
    
    if choice == 'q':
        print("👋 Goodbye!")
        return
    
    # === STEP 3: RUN EXPERIMENTS ===
    framework = None
    
    if choice == '1':
        print(f"\n🎯 Running FOCUSED experiments around champion configuration...")
        print(f"Champion LRs from your 0.0931% result: frontend=2e-5, unet=5e-6")
        
        framework = run_focused_lr_tuning(
            pretrained_weights_path=pretrained_weights_path,
            champion_frontend_lr=2e-5,  # From your successful experiment
            champion_unet_lr=5e-6
        )
        
    elif choice == '2':
        print(f"\n🚀 Running COMPREHENSIVE LR exploration...")
        print(f"⚠️ This will run up to 99 experiments!")
        confirm = input("Are you sure? This will take 1-3 days (y/N): ")
        
        if confirm.lower() == 'y':
            framework = run_comprehensive_lr_exploration(pretrained_weights_path)
        else:
            print("❌ Cancelled comprehensive exploration")
            return
            
    elif choice == '3':
        print(f"\n🔄 Running BALANCED exploration...")
        
        framework = run_balanced_exploration(
            pretrained_weights_path=pretrained_weights_path,
            total_experiments=20
        )
        
    elif choice == '4':
        print(f"\n⚡ Running QUICK validation...")
        
        framework = run_systematic_stage2_experiments(
            pretrained_unet_weights_path=pretrained_weights_path,
            experiment_type="lr_schedule",
            max_experiments=5
        )
        
    else:
        print(f"❌ Invalid choice: {choice}")
        return
    
    # === STEP 4: ANALYZE RESULTS ===
    if framework and framework.experiment_results:
        print(f"\n📈 STEP 4: Quick Results Analysis")
        
        successful = [r for r in framework.experiment_results if r.get('status') == 'completed']
        if successful:
            best = min(successful, key=lambda x: x['final_mape'])
            print(f"🏆 Best result: {best['final_mape']:.4f}% MAPE")
            print(f"   Config: frontend_lr={best['config']['lr_frontend_phase_b']}")
            print(f"           unet_lr={best['config']['lr_unet_finetune_phase_b']}")
            print(f"           scheduler={best['config']['lr_scheduler_type']}")
            
            improvement = 0.0931 - best['final_mape']  # Compared to your champion
            if improvement > 0:
                print(f"🎉 Improvement over champion: {improvement:.4f}% MAPE reduction!")
            else:
                print(f"📊 Champion still competitive (difference: {abs(improvement):.4f}%)")
        
        print(f"\n📄 Full results saved to: experiment_results/")
    
    print(f"\n✅ Experimental run complete!")
    print(f"\n💡 Next steps:")
    print(f"   • Review detailed results in experiment_results/")
    print(f"   • Update base configuration with best LR settings")
    print(f"   • Consider implementing FiLM fusion for architectural improvements")
    print(f"   • Run GAT configuration experiments with optimized LRs")

# === ALTERNATIVE: MANUAL CONTROL EXAMPLE ===
def manual_framework_example():
    """
    Example of using the framework with manual control for custom experiments.
    """
    pretrained_weights_path = "/content/checkpoints/Stage1_FullRun_UNet_Asymmetric_best_mape.pth"
    
    print("🔧 MANUAL FRAMEWORK CONTROL EXAMPLE")
    
    # Initialize framework
    framework = Stage2ExperimentalFramework(
        pretrained_unet_weights_path=pretrained_weights_path,
        base_experiment_name="Manual_Custom_Experiments",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Define grids
    framework.define_parameter_grids()
    
    # Show scope
    framework.calculate_total_possible_experiments()
    
    # Run specific experiment types
    print("\n🎯 Running 3 focused LR experiments...")
    framework.run_focused_lr_experiments_around_champion(
        champion_lr_frontend=2e-5,
        champion_lr_unet=5e-6,
        max_experiments=3
    )
    
    print("\n📊 Running 5 LR grid experiments...")
    framework.run_lr_schedule_experiments(max_experiments=5)
    
    # Generate report
    framework.generate_summary_report()
    
    return framework

if __name__ == "__main__":
    main()
    
    # Uncomment to run manual example instead:
    # manual_framework_example() 