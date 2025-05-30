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

def setup_phase2_data_loaders(test_size=0.2, batch_size=8, num_workers=0, random_state=42):
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

def validate_champion_weights_a100_stable(num_epochs=20, disable_tf32=True):
    """Validate champion hybrid weights [1.0, 0.1, 0.005] with A100 stability optimizations.
    
    Addresses numerical precision issues when running the hybrid champion loss
    on A100 GPUs compared to L4 or other architectures.
    """
    print("🔧 Validating champion hybrid weights with A100 stability optimizations...")
    
    # Configure A100 stability FIRST
    configure_a100_stability(disable_tf32=disable_tf32, verbose=True)
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"🔬 Validating champion with {num_epochs} epochs (A100 optimized)...")
    
    # Create model and optimizer
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create A100-stabilized champion hybrid loss
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=False,
        logmae_momentum=0,  # Use fixed c=0.1 (best single component)
        initial_c_logmae=0.1,
        fixed_weights_list=[1.0, 0.1, 0.005]
    ).to(device)
    
    # Replace SeismicMSSSIM with stabilized version
    criterion.seismic_ms_ssim = StabilizedSeismicMSSSIM(
        apply_log=True, data_range_log=2.0, c_for_log=0.1
    ).to(device)
    
    print("✓ Using StabilizedSeismicMSSSIM for A100 compatibility")
    
    # Diagnostic check before training
    print("\n🔍 Pre-training diagnostic check:")
    diagnose_loss_components(model, criterion, val_loader, device, num_batches=3)
    
    # Run training
    best_mape, history = train_validate_model(
        "A100_Stable_Champion", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    
    print(f"\n✅ A100-stable champion validation completed!")
    print(f"🎯 Validation MAPE: {best_mape:.4f}%")
    
    # Post-training diagnostic
    print("\n🔍 Post-training diagnostic check:")
    diagnose_loss_components(model, criterion, val_loader, device, num_batches=3)
    
    # Plot validation results
    plot_history(history, "A100-Stable Champion [1.0, 0.1, 0.005]")
    
    return {'A100_StableChampion': best_mape, 'history': history}

def test_champion_weight_variants_a100(num_epochs=25):
    """Test the champion weight variants with A100 stability to find the absolute best configuration.
    
    Tests both [1.0, 0.1, 0.005] (original champion) and [1.0, 0.12, 0.007] (systematic tuning best)
    with A100 optimizations to determine the true champion.
    """
    print("🏆 Testing champion weight variants with A100 stability...")
    
    # Configure A100 stability
    configure_a100_stability(disable_tf32=True, verbose=True)
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    results = {}
    
    # Test original champion weights [1.0, 0.1, 0.005]
    print(f"\n[1/2] 🔬 Testing Original Champion [1.0, 0.1, 0.005]...")
    
    model1 = BaselineUNet(5, 1).to(device)
    optimizer1 = optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=0.01)
    criterion1 = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5, use_adaptive_softadapt=False, logmae_momentum=0,
        initial_c_logmae=0.1, fixed_weights_list=[1.0, 0.1, 0.005]
    ).to(device)
    criterion1.seismic_ms_ssim = StabilizedSeismicMSSSIM(apply_log=True, data_range_log=2.0, c_for_log=0.1).to(device)
    
    best_mape1, _ = train_validate_model(
        "A100_Champion_Original", model1, train_loader, val_loader,
        criterion1, optimizer1, num_epochs, device, calculate_mape
    )
    results['Original_Champion_1.0_0.1_0.005'] = best_mape1
    print(f"✓ Original Champion [1.0, 0.1, 0.005]: {best_mape1:.4f}% MAPE")
    
    # Test systematic tuning best weights [1.0, 0.12, 0.007]
    print(f"\n[2/2] 🔬 Testing Systematic Tuning Best [1.0, 0.12, 0.007]...")
    
    model2 = BaselineUNet(5, 1).to(device)
    optimizer2 = optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=0.01)
    criterion2 = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5, use_adaptive_softadapt=False, logmae_momentum=0,
        initial_c_logmae=0.1, fixed_weights_list=[1.0, 0.12, 0.007]
    ).to(device)
    criterion2.seismic_ms_ssim = StabilizedSeismicMSSSIM(apply_log=True, data_range_log=2.0, c_for_log=0.1).to(device)
    
    best_mape2, _ = train_validate_model(
        "A100_Champion_Tuned", model2, train_loader, val_loader,
        criterion2, optimizer2, num_epochs, device, calculate_mape
    )
    results['Tuned_Champion_1.0_0.12_0.007'] = best_mape2
    print(f"✓ Tuned Champion [1.0, 0.12, 0.007]: {best_mape2:.4f}% MAPE")
    
    # Determine absolute champion
    absolute_champion = min(results.items(), key=lambda x: x[1])
    champion_name, champion_mape = absolute_champion
    
    print("\n" + "="*60)
    print("🏆 A100-STABLE CHAMPION COMPARISON")
    print("="*60)
    print(f"Original Champion [1.0, 0.1, 0.005]: {results['Original_Champion_1.0_0.1_0.005']:.4f}% MAPE")
    print(f"Tuned Champion [1.0, 0.12, 0.007]: {results['Tuned_Champion_1.0_0.12_0.007']:.4f}% MAPE")
    print(f"\n👑 ABSOLUTE CHAMPION: {champion_name}")
    print(f"🎯 CHAMPION MAPE: {champion_mape:.4f}%")
    
    baseline_mape = 3.93
    improvement = (baseline_mape - champion_mape) / baseline_mape * 100
    print(f"📈 IMPROVEMENT vs BASELINE: {improvement:.1f}%")
    print("="*60)
    
    return results

def diagnose_a100_issues_only():
    """Quick diagnostic to check if A100 is causing issues with the hybrid loss."""
    print("🔍 Quick A100 diagnostic for hybrid loss issues...")
    
    # Configure A100 stability
    configure_a100_stability(disable_tf32=True, verbose=True)
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    # Create model and hybrid loss
    model = BaselineUNet(5, 1).to(device)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5, use_adaptive_softadapt=False, logmae_momentum=0,
        initial_c_logmae=0.1, fixed_weights_list=[1.0, 0.1, 0.005]
    ).to(device)
    
    print("🔬 Testing standard SeismicMSSSIM:")
    stats_standard = diagnose_loss_components(model, criterion, val_loader, device, num_batches=3)
    
    # Replace with stabilized version
    criterion.seismic_ms_ssim = StabilizedSeismicMSSSIM(apply_log=True, data_range_log=2.0, c_for_log=0.1).to(device)
    
    print("🔬 Testing StabilizedSeismicMSSSIM:")
    stats_stabilized = diagnose_loss_components(model, criterion, val_loader, device, num_batches=3)
    
    print("✅ A100 diagnostic complete!")
    return {'standard': stats_standard, 'stabilized': stats_stabilized}

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
print("   ✓ A100 GPU stability optimizations added")
print()
print("Available commands:")
print()
print("🏆 1. Validate Current Champion (A100 Stable):")
print("   results = validate_champion_weights_a100_stable(num_epochs=20)")
print()
print("🥇 2. Test Champion Weight Variants (A100 Optimized):")
print("   results = test_champion_weight_variants_a100(num_epochs=25)")
print()
print("🔍 3. Quick A100 Diagnostic:")
print("   results = diagnose_a100_issues_only()")
print()
print("🎯 4. Systematic Weight Tuning (A100 Compatible):")
print("   configure_a100_stability()  # Run first")
print("   results = test_systematic_weight_tuning_only(num_epochs=15)")
print()
print("🧪 5. Test Fixed Curriculum Learning:")
print("   results = test_fixed_curriculum_only(num_epochs=25)")
print()
print("🚀 6. Complete Refined Phase 2 Suite:")
print("   results = run_refined_phase2_experiments_integrated(num_epochs=30)")
print()
print("🔧 A100 GPU Optimizations Available:")
print("   ✓ TF32 disable for FP32 precision")
print("   ✓ StabilizedSeismicMSSSIM with enhanced numerical stability")  
print("   ✓ Loss component diagnostic tools")
print("   ✓ Automatic precision handling for sensitive operations")
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

def validate_absolute_champion_extended(num_epochs=45):
    """Extended validation of the absolute champion configuration [1.0, 0.12, 0.007].
    
    Confirms the 0.0997% MAPE breakthrough with longer training and 
    checks for potential further improvements.
    """
    print("👑 Extended validation of ABSOLUTE CHAMPION [1.0, 0.12, 0.007]...")
    print(f"🎯 Target: Confirm/improve upon 0.0997% MAPE breakthrough")
    
    # Configure A100 stability
    configure_a100_stability(disable_tf32=True, verbose=True)
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    print(f"🔬 Extended training with {num_epochs} epochs...")
    
    # Create model and optimizer  
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create ABSOLUTE CHAMPION hybrid loss
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=False,
        logmae_momentum=0,  # Use fixed c=0.1 (best single component)
        initial_c_logmae=0.1,
        fixed_weights_list=[1.0, 0.12, 0.007]  # CHAMPION WEIGHTS
    ).to(device)
    
    # Use StabilizedSeismicMSSSIM for A100 compatibility
    criterion.seismic_ms_ssim = StabilizedSeismicMSSSIM(
        apply_log=True, data_range_log=2.0, c_for_log=0.1
    ).to(device)
    
    print("✓ Using CHAMPION configuration: [1.0, 0.12, 0.007]")
    print("✓ Using StabilizedSeismicMSSSIM for A100 stability")
    
    # Pre-training diagnostic
    print("\n🔍 Pre-training champion diagnostic:")
    diagnose_loss_components(model, criterion, val_loader, device, num_batches=3)
    
    # Run extended training with checkpointing every 10 epochs
    best_mape, history = train_validate_model_with_checkpoints(
        "Extended_Absolute_Champion", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    
    print(f"\n🏆 EXTENDED CHAMPION VALIDATION COMPLETE!")
    print(f"🎯 Final MAPE: {best_mape:.4f}%")
    
    # Determine if we beat the 0.0997% target
    target_mape = 0.0997
    if best_mape < target_mape:
        improvement = (target_mape - best_mape) / target_mape * 100
        print(f"🎉 NEW RECORD! {improvement:.2f}% improvement over previous champion!")
    elif best_mape <= target_mape * 1.05:  # Within 5%
        print(f"✅ Confirmed champion performance (within 5% of target)")
    else:
        print(f"⚠️  Below target by {((best_mape - target_mape) / target_mape * 100):.1f}%")
    
    # Post-training diagnostic
    print("\n🔍 Post-training champion diagnostic:")
    diagnose_loss_components(model, criterion, val_loader, device, num_batches=3)
    
    # Enhanced results analysis
    print("\n" + "="*60)
    print("📈 CHAMPION PERFORMANCE ANALYSIS")
    print("="*60)
    baseline_mape = 3.93
    improvement_vs_baseline = (baseline_mape - best_mape) / baseline_mape * 100
    print(f"Baseline MAPE: {baseline_mape:.2f}%")
    print(f"Champion MAPE: {best_mape:.4f}%")
    print(f"Total Improvement: {improvement_vs_baseline:.1f}%")
    print(f"Effective Reduction: {baseline_mape / best_mape:.1f}x better")
    print("="*60)
    
    # Plot detailed results
    plot_history(history, f"ABSOLUTE CHAMPION [1.0, 0.12, 0.007] - {best_mape:.4f}% MAPE")
    
    return {
        'Extended_Champion_MAPE': best_mape, 
        'history': history,
        'improvement_vs_baseline': improvement_vs_baseline,
        'beats_target': best_mape < target_mape
    }

def train_validate_model_with_checkpoints(experiment_name, model, train_loader, val_loader, criterion, 
                                        optimizer, num_epochs, device, calculate_mape_func, 
                                        checkpoint_freq=10):
    """Enhanced training with regular checkpointing for long experiments."""
    print(f"\n--- Starting Extended Experiment: {experiment_name} ---")
    
    best_val_mape = float('inf')
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    final_model_path = os.path.join(checkpoint_dir, f"{experiment_name}_best_mape.pth")
    
    history = {
        'train_loss': [], 'val_mae': [], 'val_mape': [],
        'val_logmae_loss': [], 'val_msssim_loss': [], 'val_atv_loss': [], 
        'loss_weights': []
    }

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(criterion, RefinedLogSpaceMAEHybridLoss):
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total']
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation Phase
        model.eval()
        running_val_mae_orig_scale = 0.0
        running_val_mape = 0.0
        running_val_logmae_component = 0.0
        running_val_msssim_component = 0.0
        running_val_atv_component = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets_torch = inputs.to(device), targets.to(device)
                outputs_torch = model(inputs)

                # Calculate MAE on original scale
                mae_orig = F.l1_loss(outputs_torch, targets_torch)
                running_val_mae_orig_scale += mae_orig.item() * inputs.size(0)

                # Calculate components if hybrid loss
                if isinstance(criterion, RefinedLogSpaceMAEHybridLoss):
                    val_loss_dict = criterion(outputs_torch, targets_torch)
                    running_val_logmae_component += val_loss_dict['logmae'].item() * inputs.size(0)
                    running_val_msssim_component += val_loss_dict['msssim'].item() * inputs.size(0)
                    running_val_atv_component += val_loss_dict['atv'].item() * inputs.size(0)

                # Calculate MAPE
                outputs_np = outputs_torch.squeeze(1).cpu().numpy()
                targets_np = targets_torch.squeeze(1).cpu().numpy()
                batch_mape_sum = 0.0
                for i in range(outputs_np.shape[0]):
                    batch_mape_sum += calculate_mape_func(targets_np[i], outputs_np[i])
                running_val_mape += (batch_mape_sum / outputs_np.shape[0]) * inputs.size(0)
        
        epoch_val_mae_orig = running_val_mae_orig_scale / len(val_loader.dataset)
        epoch_val_mape = running_val_mape / len(val_loader.dataset)
        history['val_mae'].append(epoch_val_mae_orig)
        history['val_mape'].append(epoch_val_mape)

        if isinstance(criterion, RefinedLogSpaceMAEHybridLoss):
            history['val_logmae_loss'].append(running_val_logmae_component / len(val_loader.dataset))
            history['val_msssim_loss'].append(running_val_msssim_component / len(val_loader.dataset))
            history['val_atv_loss'].append(running_val_atv_component / len(val_loader.dataset))

        print_msg = (f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | "
                     f"Val MAE (Orig): {epoch_val_mae_orig:.6f} | Val MAPE: {epoch_val_mape:.4f}%")

        if epoch_val_mape < best_val_mape:
            best_val_mape = epoch_val_mape
            torch.save(model.state_dict(), final_model_path)
            print_msg += " <<< NEW BEST MAPE - MODEL SAVED"
        
        # Checkpoint every N epochs
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mape': best_val_mape,
                'history': history
            }, checkpoint_path)
            print_msg += f" | Checkpoint saved"
        
        print(print_msg)

    print(f"\nExtended training complete! Best Val MAPE: {best_val_mape:.4f}%")
    return best_val_mape, history

def document_champion_configuration():
    """Document the CHAMPION configuration that achieved 0.0862% MAPE for future reference."""
    
    champion_config = {
        "performance": {
            "validation_mape": 0.0862,
            "improvement_vs_baseline": 97.8,  # % improvement over 3.93% baseline
            "improvement_vs_previous": 46.4,  # % improvement over 0.1609% previous best
            "original_mae": 0.17,  # Approximate final original scale MAE
            "epochs_to_convergence": 41
        },
        "architecture": {
            "model_class": "BaselineUNet", 
            "input_channels": 5,
            "output_channels": 1,
            "parameters": "~1.9M"  # Approximate
        },
        "loss_function": {
            "type": "RefinedLogSpaceMAEHybridLoss",
            "components": {
                "logmae": {"weight": 1.0, "fixed_c": 0.1, "momentum": 0},
                "ms_ssim": {"weight": 0.12, "apply_log": True, "data_range": 2.0},
                "atv": {"weight": 0.007, "weight_h": 1.0, "weight_v": 0.3}
            },
            "stabilized_msssim": True,
            "adaptive_weighting": False
        },
        "training_config": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 8,
            "epochs": 45,
            "hardware": "A100",
            "tf32_disabled": True,
            "num_workers": 0
        },
        "final_loss_components": {
            "logmae_mean": 0.039319,
            "msssim_mean": 0.254417,
            "atv_mean": 0.004205,
            "total_mean": 0.069878
        },
        "checkpoint_path": "checkpoints/Extended_Absolute_Champion_best_mape.pth",
        "validation_date": "2024-current",
        "notes": "World-class performance achieved through systematic loss engineering and A100 optimization"
    }
    
    print("=" * 80)
    print("🏆 CHAMPION CONFIGURATION DOCUMENTATION")
    print("=" * 80)
    print(f"🎯 VALIDATION MAPE: {champion_config['performance']['validation_mape']:.4f}%")
    print(f"📈 IMPROVEMENT vs BASELINE: {champion_config['performance']['improvement_vs_baseline']:.1f}%")
    print(f"🚀 IMPROVEMENT vs PREVIOUS: {champion_config['performance']['improvement_vs_previous']:.1f}%")
    print()
    print("🔧 LOSS CONFIGURATION:")
    print(f"   LogMAE (Fixed c=0.1): weight = {champion_config['loss_function']['components']['logmae']['weight']}")
    print(f"   MS-SSIM (Log-space): weight = {champion_config['loss_function']['components']['ms_ssim']['weight']}")
    print(f"   AnisotropicTV: weight = {champion_config['loss_function']['components']['atv']['weight']}")
    print()
    print("⚙️  TRAINING CONFIGURATION:")
    print(f"   Batch Size: {champion_config['training_config']['batch_size']}")
    print(f"   Hardware: {champion_config['training_config']['hardware']} (TF32 disabled)")
    print(f"   Optimizer: {champion_config['training_config']['optimizer']} (lr={champion_config['training_config']['learning_rate']}, wd={champion_config['training_config']['weight_decay']})")
    print()
    print("📊 FINAL COMPONENT BALANCE:")
    print(f"   LogMAE: {champion_config['final_loss_components']['logmae_mean']:.6f}")
    print(f"   MS-SSIM: {champion_config['final_loss_components']['msssim_mean']:.6f}")
    print(f"   ATV: {champion_config['final_loss_components']['atv_mean']:.6f}")
    print("=" * 80)
    
    return champion_config

def create_champion_model():
    """Create the exact champion model configuration for architectural experiments."""
    
    print("🏆 Creating CHAMPION model configuration...")
    
    # Create champion hybrid loss
    champion_loss = RefinedLogSpaceMAEHybridLoss(
        min_velocity=1.5,
        use_adaptive_softadapt=False,
        logmae_momentum=0,  # Fixed c=0.1
        initial_c_logmae=0.1,
        fixed_weights_list=[1.0, 0.12, 0.007]  # CHAMPION WEIGHTS
    )
    
    # Use StabilizedSeismicMSSSIM for A100 compatibility
    champion_loss.seismic_ms_ssim = StabilizedSeismicMSSSIM(
        apply_log=True, 
        data_range_log=2.0, 
        c_for_log=0.1
    )
    
    print("✓ Champion loss function created")
    print("✓ Configuration: [1.0, 0.12, 0.007] with StabilizedSeismicMSSSIM")
    
    return champion_loss

def validate_champion_visual_outputs(num_samples=5):
    """Generate visual validation of champion model outputs for qualitative assessment."""
    
    print("🔍 Generating visual validation of CHAMPION outputs...")
    
    # Configure A100 stability
    configure_a100_stability(disable_tf32=True, verbose=False)
    
    # Verify integration
    if not integrate_phase2_with_existing_notebook():
        return None
    
    # Setup data loaders
    train_loader, val_loader = setup_phase2_data_loaders()
    if train_loader is None or val_loader is None:
        return None
    
    # Load champion model
    model = BaselineUNet(5, 1).to(device)
    champion_path = "checkpoints/Extended_Absolute_Champion_best_mape.pth"
    
    try:
        model.load_state_dict(torch.load(champion_path))
        print(f"✓ Loaded champion model from {champion_path}")
    except FileNotFoundError:
        print(f"⚠️  Champion model not found at {champion_path}")
        print("   Please run the extended validation first")
        return None
    
    model.eval()
    
    # Generate predictions on validation samples
    predictions = []
    targets = []
    
    with torch.no_grad():
        for i, (inputs, target_batch) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            inputs, target_batch = inputs.to(device), target_batch.to(device)
            pred_batch = model(inputs)
            
            # Convert to numpy for visualization
            pred_np = pred_batch.squeeze(1).cpu().numpy()
            target_np = target_batch.squeeze(1).cpu().numpy()
            
            predictions.extend(pred_np)
            targets.extend(target_np)
    
    # Calculate detailed metrics
    mapes = []
    maes = []
    
    for pred, target in zip(predictions, targets):
        sample_mape = calculate_mape(target, pred)
        sample_mae = np.mean(np.abs(pred - target))
        mapes.append(sample_mape)
        maes.append(sample_mae)
    
    print(f"\n📊 CHAMPION VISUAL VALIDATION RESULTS:")
    print(f"Samples analyzed: {len(predictions)}")
    print(f"Average MAPE: {np.mean(mapes):.4f}% (±{np.std(mapes):.4f}%)")
    print(f"Average MAE: {np.mean(maes):.6f} (±{np.std(maes):.6f})")
    print(f"Best sample MAPE: {np.min(mapes):.4f}%")
    print(f"Worst sample MAPE: {np.max(mapes):.4f}%")
    
    # Create visualization
    fig, axes = plt.subplots(3, min(num_samples, 3), figsize=(15, 9))
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(min(num_samples, 3)):
        pred = predictions[i]
        target = targets[i]
        diff = np.abs(pred - target)
        
        # Prediction
        axes[0, i].imshow(pred, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Champion Prediction {i+1}\nMAPE: {mapes[i]:.4f}%')
        axes[0, i].axis('off')
        
        # Ground Truth
        axes[1, i].imshow(target, cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Ground Truth {i+1}')
        axes[1, i].axis('off')
        
        # Absolute Difference
        axes[2, i].imshow(diff, cmap='hot', aspect='auto')
        axes[2, i].set_title(f'Absolute Error {i+1}\nMAE: {maes[i]:.6f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('CHAMPION MODEL (0.0862% MAPE) - Visual Validation', fontsize=16, y=1.02)
    plt.show()
    
    return {
        'predictions': predictions,
        'targets': targets,
        'mapes': mapes,
        'maes': maes,
        'summary_stats': {
            'mean_mape': np.mean(mapes),
            'std_mape': np.std(mapes),
            'mean_mae': np.mean(maes),
            'std_mae': np.std(maes)
        }
    }

# =============================================================================
# PHASE 2 PRIORITY 2: ARCHITECTURAL INNOVATIONS PREPARATION
# =============================================================================

def prepare_architectural_experiments():
    """Prepare for Phase 2 Priority 2: Architectural innovations using champion loss."""
    
    print("=" * 80)
    print("🚀 PHASE 2 PRIORITY 2: ARCHITECTURAL INNOVATIONS")
    print("=" * 80)
    print("Using CHAMPION loss function as foundation for architectural experiments")
    print(f"Champion Performance Target: 0.0862% MAPE")
    print()
    print("📋 PLANNED ARCHITECTURAL ENHANCEMENTS:")
    print("1. 🔗 LightweightGATFusion: Inter-shot modeling with Graph Attention")
    print("2. 🧱 AnisotropicConvBlock: Geological structure-aware convolutions")
    print("3. 🔍 HyPerStructureUNet: Full integration of enhanced components")
    print()
    print("🎯 GOALS:")
    print("- Push below 0.05% MAPE (98.7%+ improvement vs baseline)")
    print("- Enhance geological realism and structure preservation")
    print("- Maintain computational efficiency for submission")
    print()
    print("🔧 FOUNDATION COMPONENTS READY:")
    print("✓ Champion loss function [1.0, 0.12, 0.007]")
    print("✓ A100 stability optimizations")
    print("✓ StabilizedSeismicMSSSIM")
    print("✓ Systematic experimental framework")
    print("=" * 80)
    
    # Return champion loss for architectural experiments
    champion_loss = create_champion_model()
    champion_config = document_champion_configuration()
    
    return champion_loss, champion_config