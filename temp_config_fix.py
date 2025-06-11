# CORRECTED CONFIGURATION - fixes the threshold issue

BASE_CONFIG = {
    'num_epochs_phase_b': 40,
    'curriculum_simple_epochs': 0,
    'batch_size': 5,
    'film_generator_mlp_type': '2_layer',
    'lr_frontend_phase_b': 5e-5,
    'weight_decay': 0.01,
    'weight_decay_film': 1e-3,
}

configurations = [
    {
        **BASE_CONFIG,
        'config_id': 'cfg_06_plateau_to_cosine_FIXED',
        'lr_unet': 1e-5,                        # U-Net starts at 1e-5
        'lr_film': 1e-4,
        'lambda_gamma': 0.005, 
        'lambda_beta': 0.0005, 
        'scheduler': 'PlateauToCosine',
        'plateau_patience': 4,
        'plateau_factor': 0.85,                 # 15% reduction per plateau
        'plateau_threshold': 0.001,
        
        # ✅ CRITICAL FIX: Threshold must be LOWER than initial U-Net LR
        'lr_switch_threshold': 5e-6,            # Switch after ~2 plateau reductions
        # Logic: 1e-5 → 8.5e-6 → 7.225e-6, switch when LR ≤ 5e-6
        
        'cosine_T_max': 8,
        'cosine_eta_min': 1e-8,
        'threshold_group_index': 4,             # ✅ EXPLICIT: Monitor U-Net group
        'use_sam': False,
        'use_gc': False,
        'SAM_AVAILABLE': SAM_AVAILABLE,
        'SAM_CLASS': SAM,
        'enable_running_stats': enable_running_stats,
        'disable_running_stats': disable_running_stats,
        'notes': 'CORRECTED: Hybrid Plateau→Cosine with proper U-Net threshold (5e-6 < 1e-5 initial LR)'
    },
]

# Alternative configurations for different switching behaviors:

# Option 1: Conservative switching (after many plateau reductions)
conservative_config = {
    **BASE_CONFIG,
    'config_id': 'cfg_07_conservative_switch',
    'lr_unet': 1e-5,
    'lr_film': 1e-4,
    'lr_switch_threshold': 2e-6,                # Switch after ~3-4 plateau reductions
    'cosine_T_max': 6,                          # Shorter cosine phase
    # ... other params same
}

# Option 2: Aggressive switching (after first plateau reduction)
aggressive_config = {
    **BASE_CONFIG,
    'config_id': 'cfg_08_aggressive_switch', 
    'lr_unet': 1e-5,
    'lr_film': 1e-4,
    'lr_switch_threshold': 8e-6,                # Switch after first plateau reduction
    'cosine_T_max': 12,                         # Longer cosine phase
    # ... other params same
} 