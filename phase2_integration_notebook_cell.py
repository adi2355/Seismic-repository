# =============================================================================
# PHASE 2 EXPERIMENTAL FRAMEWORK INTEGRATION
# =============================================================================

# 🚨 IMPORTANT: If you get "normalize_slopes" error:
# 1. RESTART YOUR KERNEL (Kernel → Restart)
# 2. RE-RUN ALL CELLS 
# The error has been fixed but requires kernel restart to clear cached code

# Copy this entire cell into your Jupyter notebook after your existing model definitions

print("="*80)
print("PHASE 2: ADVANCED LOSS FUNCTION EXPERIMENTAL FRAMEWORK")
print("Integrating with existing notebook infrastructure...")
print("="*80)

# Install required packages - run this in notebook cell
# !pip install pytorch-msssim softadapt scikit-image

# Force reload of the experimental framework to ensure latest fixes
import importlib
import sys
if 'phase2_experimental_framework' in sys.modules:
    importlib.reload(sys.modules['phase2_experimental_framework'])

# Import the experimental framework
exec(open('phase2_experimental_framework.py').read())

# =============================================================================
# INTEGRATION WITH EXISTING NOTEBOOK INFRASTRUCTURE
# =============================================================================

def integrate_phase2_with_existing_notebook():
    """Integrate Phase 2 experiments with your existing notebook infrastructure."""
    
    print("Setting up Phase 2 integration...")
    
    # Verify that required components from your notebook are available
    required_components = [
        'all_sample_folder_paths',  # Your data paths
        'BaselineUNet',             # Your model class
        'SeismicDataset',           # Your dataset class  
        'calculate_mape',           # Your MAPE function
        'device'                    # Your device setting
    ]
    
    missing_components = []
    for component in required_components:
        if component not in globals():
            missing_components.append(component)
    
    if missing_components:
        print(f"⚠️  Missing required components: {missing_components}")
        print("Please ensure these are defined in your notebook before running Phase 2 experiments.")
        return False
    
    print("✓ All required components found")
    return True

def setup_phase2_data_loaders(test_size=0.2, batch_size=4, num_workers=0, random_state=42):
    """Set up data loaders for Phase 2 experiments using your existing infrastructure."""
    
    if not all_sample_folder_paths:
        print("❌ No sample folder paths found. Please load your data first.")
        return None, None
    
    print(f"Setting up data loaders with {len(all_sample_folder_paths)} total samples...")
    
    # Split data
    train_paths, val_paths = train_test_split(
        all_sample_folder_paths, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    # Create datasets
    train_dataset = SeismicDataset(train_paths)
    val_dataset = SeismicDataset(val_paths)
    
    # Force single-process for stability in Colab/Jupyter (eliminates AssertionErrors)
    current_num_workers = 0
    pin_memory = False  # Not beneficial with num_workers=0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=current_num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=current_num_workers, 
        pin_memory=pin_memory
    )
    
    print(f"✓ Data loaders created (single-process for stability):")
    print(f"  - Training: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  - Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Device: {device}")
    
    return train_loader, val_loader

def run_phase2_experiments_integrated(num_epochs=5, min_velocity=1.5):
    """Run Phase 2 experiments integrated with your existing notebook setup.
    
    Args:
        num_epochs: Number of training epochs (start with 5 for testing, then use 30+ for real experiments)
        min_velocity: Minimum velocity for clamping (from your EDA)
    """
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"\n🚀 Starting Phase 2 experiments with {num_epochs} epochs per experiment...")
    print("📊 This will systematically test 4 different loss function configurations")
    print("⏱️  Estimated time: ~{} minutes".format(num_epochs * 4 * len(train_loader) // 10))
    
    # Run the experiments
    results = run_phase2_experiments(
        BaselineUNet=BaselineUNet,
        SeismicDataset=SeismicDataset, 
        train_loader=train_loader,
        val_loader=val_loader,
        calculate_mape=calculate_mape,
        device=device,
        num_epochs=num_epochs,
        min_velocity=min_velocity
    )
    
    return results

def quick_test_phase2_setup():
    """Quick test to verify Phase 2 setup works with minimal training."""
    print("🧪 Running quick Phase 2 setup test...")
    results = run_phase2_experiments_integrated(num_epochs=2)
    if results:
        print("✅ Phase 2 setup test successful!")
        print("💡 You can now run full experiments with higher num_epochs")
    return results

def test_only_hybrid_adaptive(num_epochs=2):
    """Test only the 4th experiment (HybridAdaptive) to verify SoftAdapt fixes."""
    print("🧪 Testing only Experiment 4: HybridAdaptive with SoftAdapt...")
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"🚀 Testing HybridAdaptive experiment with {num_epochs} epochs...")
    
    # Create model and optimizer
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create hybrid adaptive loss
    criterion = LogSpaceMAEHybridLoss(
        min_velocity=1.5, 
        use_adaptive_softadapt=True,
        softadapt_beta=0.1,
        softadapt_update_freq=10
    ).to(device)
    
    # Run training
    best_mape, history = train_validate_model(
        "Test_HybridAdaptiveOnly", model, train_loader, val_loader, 
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    
    print(f"✅ HybridAdaptive test completed! Best MAPE: {best_mape:.4f}%")
    return {'HybridAdaptive': best_mape, 'history': history}

def run_hybrid_loss_refinement_experiments(num_epochs=30):
    """Run systematic hybrid loss refinement experiments based on analysis insights."""
    print("🔬 Starting Hybrid Loss Refinement Experiments...")
    
    if not integrate_phase2_with_existing_notebook():
        return None
    
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    results = {}
    
    # Experiment R1: Manual MS-SSIM weight tuning
    print("\n[R1] Testing LogMAE + MS-SSIM weight tuning...")
    for w_msssim in [0.1, 0.3, 0.5]:
        model = BaselineUNet(5, 1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = RefinedLogSpaceMAEHybridLoss(
            min_velocity=1.5,
            use_adaptive_softadapt=False,
            logmae_momentum=0,  # Use fixed c=0.1
            initial_c_logmae=0.1,
            fixed_weights_list=[1.0, w_msssim, 0.0]  # No ATV for now
        ).to(device)
        
        best_mape, _ = train_validate_model(
            f"R1_LogMAE_MSSSIM_w{w_msssim}", model, train_loader, val_loader,
            criterion, optimizer, num_epochs, device, calculate_mape
        )
        results[f'LogMAE+MSSSIM_w{w_msssim}'] = best_mape
        print(f"✓ LogMAE + MS-SSIM (w={w_msssim}): {best_mape:.4f}% MAPE")
    
    # Experiment R2: Add ATV to best MS-SSIM combo
    best_msssim_w = min([(w, mape) for w, mape in results.items() if 'MSSSIM' in w], key=lambda x: x[1])
    best_w_val = float(best_msssim_w[0].split('_w')[1])
    
    print(f"\n[R2] Adding ATV to best combo (MS-SSIM w={best_w_val})...")
    for w_atv in [0.001, 0.005, 0.01]:
        model = BaselineUNet(5, 1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = RefinedLogSpaceMAEHybridLoss(
            min_velocity=1.5,
            use_adaptive_softadapt=False,
            logmae_momentum=0,
            initial_c_logmae=0.1,
            fixed_weights_list=[1.0, best_w_val, w_atv]
        ).to(device)
        
        best_mape, _ = train_validate_model(
            f"R2_FullHybrid_w{best_w_val}_{w_atv}", model, train_loader, val_loader,
            criterion, optimizer, num_epochs, device, calculate_mape
        )
        results[f'FullHybrid_w{best_w_val}_{w_atv}'] = best_mape
        print(f"✓ Full Hybrid (MS-SSIM={best_w_val}, ATV={w_atv}): {best_mape:.4f}% MAPE")
    
    # Experiment R3: Scaled SoftAdapt
    print("\n[R3] Testing scaled SoftAdapt...")
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=True,
        logmae_momentum=0,
        initial_c_logmae=0.1,
        scale_for_softadapt=True,
        component_scales=[20.0, 2.0, 200.0],  # Aggressive scaling
        softadapt_update_freq=5  # More frequent updates
    ).to(device)
    
    best_mape, _ = train_validate_model(
        "R3_ScaledSoftAdapt", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    results['ScaledSoftAdapt'] = best_mape
    print(f"✓ Scaled SoftAdapt: {best_mape:.4f}% MAPE")
    
    # Experiment R4: Curriculum Learning
    print("\n[R4] Testing curriculum learning...")
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=True,
        logmae_momentum=0,
        initial_c_logmae=0.1,
        start_simple=True,
        curriculum_epochs=10,  # LogMAE only for first 10 epochs
        scale_for_softadapt=True,
        component_scales=[15.0, 1.5, 150.0]
    ).to(device)
    
    # Need to modify training loop to call set_epoch
    best_mape = train_with_curriculum(
        "R4_CurriculumSoftAdapt", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    results['CurriculumSoftAdapt'] = best_mape
    print(f"✓ Curriculum SoftAdapt: {best_mape:.4f}% MAPE")
    
    # Print summary
    print("\n" + "="*80)
    print("HYBRID LOSS REFINEMENT RESULTS")
    print("="*80)
    champion_mape = 0.4435  # FixedCLogMAE benchmark
    print(f"Champion to beat: FixedCLogMAE = {champion_mape:.4f}% MAPE")
    print("-" * 50)
    
    for exp_name, mape in sorted(results.items(), key=lambda x: x[1]):
        improvement = "🏆 NEW CHAMPION!" if mape < champion_mape else f"({(mape-champion_mape)/champion_mape*100:+.1f}%)"
        print(f"{exp_name:25s}: {mape:.4f}% MAPE {improvement}")
    
    return results

def train_with_curriculum(experiment_name, model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs, device, calculate_mape_func):
    """Training function with curriculum learning support."""
    print(f"\n--- Starting Curriculum Experiment: {experiment_name} ---")
    
    best_val_mape = float('inf')
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    model_path = os.path.join(checkpoint_dir, f"{experiment_name}_best_mape.pth")
    
    for epoch in range(num_epochs):
        # Set epoch for curriculum learning
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if hasattr(criterion, 'forward') and callable(getattr(criterion, 'forward')):
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total'] if isinstance(loss_dict, dict) else loss_dict
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        running_val_mape = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets_torch = inputs.to(device), targets.to(device)
                outputs_torch = model(inputs)
                
                outputs_np = outputs_torch.squeeze(1).cpu().numpy()
                targets_np = targets_torch.squeeze(1).cpu().numpy()
                batch_mape_sum = 0.0
                for i in range(outputs_np.shape[0]):
                    batch_mape_sum += calculate_mape_func(targets_np[i], outputs_np[i])
                running_val_mape += (batch_mape_sum / outputs_np.shape[0]) * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_val_mape = running_val_mape / len(val_loader.dataset)
        
        print_msg = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val MAPE: {epoch_val_mape:.4f}%"
        
        if hasattr(criterion, 'use_adaptive_softadapt') and criterion.use_adaptive_softadapt and hasattr(criterion, 'current_weights'):
            try:
                weights_str = ", ".join([f"{w:.3f}" for w in criterion.current_weights.cpu().numpy()])
                print_msg += f" | Weights: [{weights_str}]"
            except:
                pass
        
        if epoch_val_mape < best_val_mape:
            best_val_mape = epoch_val_mape
            torch.save(model.state_dict(), model_path)
            print_msg += " <<< BEST MAPE SO FAR - MODEL SAVED"
        
        print(print_msg)
    
    return best_val_mape

# =============================================================================
# READY-TO-USE EXPERIMENTAL COMMANDS
# =============================================================================

print("\n" + "="*60)
print("PHASE 2 EXPERIMENTAL FRAMEWORK READY!")
print("="*60)
print("Available commands:")
print()
print("1. Quick Setup Test (2 epochs):")
print("   results = quick_test_phase2_setup()")
print()
print("2. Test Only HybridAdaptive (2 epochs):")
print("   results = test_only_hybrid_adaptive()")
print()
print("3. Full Phase 2 Experiments (30 epochs):")
print("   results = run_phase2_experiments_integrated(num_epochs=30)")
print()
print("4. 🔬 Hybrid Loss Refinement Experiments:")
print("   results = run_hybrid_loss_refinement_experiments(num_epochs=20)")
print()
print("5. Custom Configuration:")
print("   train_loader, val_loader = setup_phase2_data_loaders()")
print("   # Then use individual loss functions as needed")
print()
print("Advanced Loss Functions Available:")
print("- AdaptiveLogSpaceMAE: MAPE-aligned loss with adaptive parameters")
print("- SeismicMSSSIM: Geological structure-aware similarity loss")  
print("- AnisotropicTotalVariationLoss: Layer-aware smoothness regularization")
print("- RefinedLogSpaceMAEHybridLoss: Improved multi-component loss with scaling")
print("="*60)

# Example of how to use the framework
"""
# USAGE EXAMPLE:

# 1. Quick test (run this first to verify everything works)
results_test = quick_test_phase2_setup()

# 2. If test passes, run full experiments
results_full = run_phase2_experiments_integrated(num_epochs=30)

# 3. Analyze results
print("Final Results Summary:")
for exp_name, best_mape in results_full.items():
    print(f"{exp_name}: {best_mape:.4f}% MAPE")

# 4. Best models are automatically saved in ./checkpoints/ directory
# Load best model for further use:
# best_model = BaselineUNet(5, 1)
# best_model.load_state_dict(torch.load('checkpoints/Exp_HybridAdaptiveWeights_best_mape.pth'))
# best_model.to(device)
""" 

def run_refined_phase2_experiments_integrated(num_epochs=30, min_velocity=1.5):
    """Run the complete refined Phase 2 experimental suite with all fixes and improvements.
    
    This includes:
    1. All original experiments with critical bug fixes
    2. Systematic weight tuning around champion [1.0, 0.1, 0.005]  
    3. Fixed curriculum learning with proper SoftAdapt initialization
    4. Improved SoftAdapt scaling based on component magnitude analysis
    
    Args:
        num_epochs: Number of training epochs (recommend 30+ for full experiments)
        min_velocity: Minimum velocity for clamping (from your EDA)
    """
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"\n🚀 Starting REFINED Phase 2 experiments with {num_epochs} epochs per experiment...")
    print("🔧 All critical bugs fixed, systematic tuning implemented")
    print("📊 This will systematically test 6 core experiments + weight tuning")
    print("⏱️  Estimated time: ~{} minutes".format(num_epochs * 6 * len(train_loader) // 8))
    
    # Run the refined experimental suite
    results = run_refined_phase2_experiments(
        BaselineUNet=BaselineUNet,
        train_loader=train_loader,
        val_loader=val_loader,
        calculate_mape=calculate_mape,
        device=device,
        num_epochs=num_epochs,
        min_velocity=min_velocity
    )
    
    return results

def test_systematic_weight_tuning_only(num_epochs=15, champion_mape=0.3790):
    """Test only the systematic weight tuning around champion weights [1.0, 0.1, 0.005].
    
    This is useful for focused tuning experiments without running the full suite.
    """
    print("🎯 Testing systematic weight tuning around champion weights...")
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"🔬 Running systematic weight tuning with {num_epochs} epochs per test...")
    print(f"🏆 Target to beat: {champion_mape:.4f}% MAPE")
    
    # Run systematic weight tuning
    results = run_systematic_weight_tuning_experiments(
        BaselineUNet=BaselineUNet,
        train_loader=train_loader,
        val_loader=val_loader,
        calculate_mape=calculate_mape,
        device=device,
        num_epochs=num_epochs,
        min_velocity=1.5,
        champion_mape=champion_mape
    )
    
    return results

def test_fixed_curriculum_only(num_epochs=25):
    """Test only the fixed curriculum learning experiment to verify the bug fix."""
    print("🧪 Testing fixed curriculum learning + SoftAdapt...")
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"🔬 Testing curriculum learning with {num_epochs} epochs...")
    print("📚 First 10 epochs: LogMAE only, then full hybrid with SoftAdapt")
    
    # Create model and optimizer
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create fixed curriculum loss with proper initialization
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=True,
        logmae_momentum=0,  # Use fixed c=0.1
        initial_c_logmae=0.1,
        start_simple=True,
        curriculum_epochs=10,
        component_scales="adaptive",  # [15.0, 2.0, 50.0]
        softadapt_beta=0.1,
        softadapt_update_freq=5
    ).to(device)
    
    # Run training with fixed curriculum function
    best_mape = train_with_curriculum_fixed(
        "Test_FixedCurriculum", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    
    print(f"✅ Fixed curriculum test completed! Best MAPE: {best_mape:.4f}%")
    return {'FixedCurriculum': best_mape}

def validate_champion_weights(num_epochs=20):
    """Validate the current champion hybrid weights [1.0, 0.1, 0.005] with fresh training."""
    print("🏆 Validating champion hybrid weights [1.0, 0.1, 0.005]...")
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"🔬 Validating champion with {num_epochs} epochs...")
    
    # Create model and optimizer
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create champion hybrid loss
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=False,
        logmae_momentum=0,  # Use fixed c=0.1 (best single component)
        initial_c_logmae=0.1,
        fixed_weights_list=[1.0, 0.1, 0.005]
    ).to(device)
    
    # Run training
    best_mape, history = train_validate_model(
        "Validate_Champion", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    
    print(f"✅ Champion validation completed!")
    print(f"🎯 Validation MAPE: {best_mape:.4f}%")
    
    # Plot validation results
    plot_history(history, "Champion Validation [1.0, 0.1, 0.005]")
    
    return {'ChampionValidation': best_mape, 'history': history}

# =============================================================================
# ENHANCED READY-TO-USE EXPERIMENTAL COMMANDS
# =============================================================================

print("\n" + "="*60)
print("REFINED PHASE 2 EXPERIMENTAL FRAMEWORK READY!")
print("="*60)
print("🔧 Critical bugs fixed:")
print("   ✓ Curriculum learning AttributeError resolved")
print("   ✓ SoftAdapt scaling improved based on component analysis")
print("   ✓ Enhanced error handling and initialization")
print()
print("Available commands:")
print()
print("🏆 1. Validate Current Champion (Quick):")
print("   results = validate_champion_weights(num_epochs=20)")
print()
print("🎯 2. Systematic Weight Tuning Only:")
print("   results = test_systematic_weight_tuning_only(num_epochs=15)")
print()
print("🧪 3. Test Fixed Curriculum Learning:")
print("   results = test_fixed_curriculum_only(num_epochs=25)")
print()
print("🚀 4. Complete Refined Phase 2 Suite (Full):")
print("   results = run_refined_phase2_experiments_integrated(num_epochs=30)")
print()
print("⚡ 5. Quick Setup Test (Previous - Still Available):")
print("   results = quick_test_phase2_setup()")
print()
print("🔧 Enhanced Loss Functions Available:")
print("   ✓ RefinedLogSpaceMAEHybridLoss with proper curriculum support")
print("   ✓ Systematic weight tuning around champion [1.0, 0.1, 0.005]")
print("   ✓ Improved SoftAdapt scaling [15.0, 2.0, 50.0]")
print("   ✓ Multiple SoftAdapt configurations (responsive, conservative)")
print("="*60)

# Example usage with refined experiments
"""
# REFINED USAGE EXAMPLE:

# 1. Quick validation of current champion
champion_results = validate_champion_weights()

# 2. If validation successful, test systematic weight tuning
tuning_results = test_systematic_weight_tuning_only(num_epochs=15, champion_mape=champion_results['ChampionValidation'])

# 3. Test fixed curriculum learning (bug was critical)
curriculum_results = test_fixed_curriculum_only()

# 4. If individual tests pass, run complete refined suite
full_results = run_refined_phase2_experiments_integrated(num_epochs=30)

# 5. Analyze final results
print("FINAL ANALYSIS:")
print(f"Champion Validation: {champion_results['ChampionValidation']:.4f}% MAPE")
best_tuning = min(tuning_results.values()) if tuning_results else float('inf')
print(f"Best Tuning Result: {best_tuning:.4f}% MAPE") 
print(f"Curriculum Learning: {curriculum_results['FixedCurriculum']:.4f}% MAPE")
""" 