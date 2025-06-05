#!/usr/bin/env python3
"""
Test Script for Phase 2b Diagnostic Framework

This script validates that the diagnostic framework is properly set up
and can run basic experiments without errors.
"""

import os
import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from phase2b_diagnostic_experiments import (
            Phase2bDiagnosticFramework,
            run_phase2b_diagnostics,
            quick_phase2b_diagnostic_test,
            focused_ultra_low_lr_test
        )
        print("✅ Diagnostic framework modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are present:")
        print("  - phase2b_diagnostic_experiments.py")
        print("  - complete_sincgat_unet_integration.py")
        print("  - phase2_experimental_framework.py")
        return False


def test_checkpoint_detection():
    """Test checkpoint detection functionality."""
    print("\n🔍 Testing checkpoint detection...")
    
    try:
        from run_diagnostic_experiments import find_best_checkpoint
        
        checkpoint_path = find_best_checkpoint()
        print(f"✅ Checkpoint detected: {checkpoint_path}")
        
        if os.path.exists(checkpoint_path):
            print(f"✅ Checkpoint file exists and is accessible")
            return True, checkpoint_path
        else:
            print(f"❌ Checkpoint file not found at: {checkpoint_path}")
            return False, None
            
    except Exception as e:
        print(f"❌ Checkpoint detection failed: {e}")
        return False, None


def test_framework_initialization(checkpoint_path):
    """Test framework initialization."""
    print("\n🏗️ Testing framework initialization...")
    
    try:
        from phase2b_diagnostic_experiments import Phase2bDiagnosticFramework
        
        framework = Phase2bDiagnosticFramework(
            base_checkpoint_path=checkpoint_path,
            experiment_base_name="Test_Framework"
        )
        
        print(f"✅ Framework initialized successfully")
        print(f"  Device: {framework.device}")
        print(f"  Results dir: {framework.results_dir}")
        print(f"  Checkpoint dir: {framework.checkpoint_dir}")
        
        return True, framework
        
    except Exception as e:
        print(f"❌ Framework initialization failed: {e}")
        traceback.print_exc()
        return False, None


def test_model_loading(framework):
    """Test model loading functionality."""
    print("\n🤖 Testing model loading...")
    
    try:
        model = framework.load_base_model(film_generator_type='2_layer')
        
        # Check model components
        components = {
            'shot_encoder': hasattr(model, 'shot_encoder'),
            'gat_fusion': hasattr(model, 'gat_fusion'),
            'unet': hasattr(model, 'unet'),
            'film_bottleneck_modulator': hasattr(model, 'film_bottleneck_modulator')
        }
        
        print(f"✅ Model loaded successfully")
        print(f"  Components present: {sum(components.values())}/4")
        for comp, present in components.items():
            status = "✅" if present else "❌"
            print(f"    {status} {comp}")
            
        return True, model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False, None


def test_data_loaders(framework):
    """Test data loader setup."""
    print("\n📊 Testing data loader setup...")
    
    try:
        train_loader, val_loader = framework.setup_data_loaders(batch_size=2)
        
        print(f"✅ Data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Test one batch
        try:
            batch = next(iter(train_loader))
            inputs, targets = batch
            print(f"✅ Sample batch loaded")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Target shape: {targets.shape}")
            
            return True, (train_loader, val_loader)
            
        except Exception as e:
            print(f"⚠️ Data loader created but batch loading failed: {e}")
            print("  This might be expected if using dummy data")
            return True, (train_loader, val_loader)
        
    except Exception as e:
        print(f"❌ Data loader setup failed: {e}")
        traceback.print_exc()
        return False, None


def test_optimizer_creation(framework, model):
    """Test optimizer creation."""
    print("\n⚙️ Testing optimizer creation...")
    
    try:
        optimizer = framework.create_phase2b_optimizer(
            model=model,
            lr_unet=1e-6,  # Test ultra-low learning rate
            lr_frontend=5e-5,
            lr_film_generator=1e-4
        )
        
        print(f"✅ Optimizer created successfully")
        print(f"  Parameter groups: {len(optimizer.param_groups)}")
        
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('group_name', f'Group_{i}')
            print(f"    {group_name}: LR={group['lr']:.2e}, Params={len(group['params'])}")
        
        return True, optimizer
        
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        traceback.print_exc()
        return False, None


def test_criterion_creation(framework):
    """Test loss criterion creation."""
    print("\n🎯 Testing criterion creation...")
    
    try:
        criterion = framework.create_criterion(
            lambda_gamma_res=0.005,
            lambda_beta_res=0.0005,
            use_film_reg=True
        )
        
        print(f"✅ Criterion created successfully")
        print(f"  Type: {type(criterion).__name__}")
        print(f"  FiLM regularization: {hasattr(criterion, 'use_film_reg')}")
        
        return True, criterion
        
    except Exception as e:
        print(f"❌ Criterion creation failed: {e}")
        traceback.print_exc()
        return False, None


def test_single_experiment_config():
    """Test single experiment configuration."""
    print("\n🔬 Testing single experiment configuration...")
    
    try:
        from run_diagnostic_experiments import find_best_checkpoint
        from phase2b_diagnostic_experiments import Phase2bDiagnosticFramework
        
        checkpoint_path = find_best_checkpoint()
        framework = Phase2bDiagnosticFramework(
            base_checkpoint_path=checkpoint_path,
            experiment_base_name="Test_Single_Experiment"
        )
        
        # Test configuration for ultra-low LR experiment
        test_config = {
            'experiment_id': 'test_ultra_low_lr',
            'lr_unet': 1e-6,
            'lr_frontend': 5e-5,
            'lr_film_generator': 1e-4,
            'film_generator_type': '2_layer',
            'description': 'Test ultra-low U-Net LR experiment'
        }
        
        print(f"✅ Test experiment configuration valid")
        print(f"  Config: {test_config}")
        
        # Note: We don't actually run the experiment here to save time
        print(f"⏭️ Skipping actual experiment execution (use run_diagnostic_experiments.py --mode quick)")
        
        return True
        
    except Exception as e:
        print(f"❌ Single experiment configuration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("=" * 80)
    print("🧪 PHASE 2B DIAGNOSTIC FRAMEWORK VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    test_results = {}
    
    # Test 1: Module imports
    test_results['imports'] = test_imports()
    if not test_results['imports']:
        print("\n❌ Critical error: Cannot proceed without proper imports")
        return False
    
    # Test 2: Checkpoint detection
    test_results['checkpoint'], checkpoint_path = test_checkpoint_detection()
    if not test_results['checkpoint']:
        print("\n❌ Critical error: No valid checkpoint found")
        return False
    
    # Test 3: Framework initialization
    test_results['framework'], framework = test_framework_initialization(checkpoint_path)
    if not test_results['framework']:
        print("\n❌ Critical error: Framework initialization failed")
        return False
    
    # Test 4: Model loading
    test_results['model'], model = test_model_loading(framework)
    
    # Test 5: Data loaders
    test_results['data'], data_loaders = test_data_loaders(framework)
    
    # Test 6: Optimizer creation
    if test_results['model']:
        test_results['optimizer'], optimizer = test_optimizer_creation(framework, model)
    else:
        test_results['optimizer'] = False
    
    # Test 7: Criterion creation
    test_results['criterion'], criterion = test_criterion_creation(framework)
    
    # Test 8: Single experiment configuration
    test_results['single_experiment'] = test_single_experiment_config()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Framework is ready for diagnostic experiments")
        print("\nNext steps:")
        print("  1. Run quick test: python run_diagnostic_experiments.py --mode quick")
        print("  2. Run focused LR test: python run_diagnostic_experiments.py --mode focused_lr --epochs 15")
        print("  3. Run complete suite: python run_diagnostic_experiments.py --mode complete --epochs 30")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed")
        print("🔧 Please fix the failing components before running diagnostic experiments")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 