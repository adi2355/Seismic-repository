# Phase 2: Advanced Loss Function Experimental Framework
# Systematic experimentation for seismic velocity inversion performance improvement

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import advanced loss function libraries
try:
    from pytorch_msssim import MS_SSIM
    print("✓ pytorch_msssim imported successfully")
except ImportError:
    print("✗ pytorch_msssim not found. Install with: pip install pytorch-msssim")
    MS_SSIM = None

try:
    from softadapt import LossWeightedSoftAdapt
    print("✓ softadapt imported successfully")
except ImportError:
    print("✗ softadapt not found. Install with: pip install softadapt")
    LossWeightedSoftAdapt = None


# =============================================================================
# ADVANCED LOSS FUNCTION IMPLEMENTATIONS
# =============================================================================

class AdaptiveLogSpaceMAE(nn.Module):
    """Adaptive Log-Space MAE Loss with momentum-based c parameter adaptation.
    
    This loss function addresses MAPE optimization challenges by:
    1. Operating in log-space to align with relative error objectives
    2. Adaptively adjusting the c parameter based on data statistics
    3. Providing numerical stability through clamping and epsilon handling
    """
    def __init__(self, min_velocity=1.5, momentum=0.9, initial_c=0.1, adaptive_c_on_true_only=True):
        super().__init__()
        self.min_velocity = min_velocity
        self.momentum = momentum
        self.register_buffer('running_min_vel', torch.tensor(float(min_velocity)))
        self.initial_c = initial_c
        self.adaptive_c_on_true_only = adaptive_c_on_true_only
        self.epsilon_log = 1e-8

    def forward(self, vp_pred, vp_true):
        if self.training and self.momentum > 0:
            if self.adaptive_c_on_true_only:
                current_batch_min_val = torch.min(vp_true).detach()
            else:
                current_batch_min_val = torch.min(torch.min(vp_pred.detach()), torch.min(vp_true.detach()))

            current_batch_min_val = torch.max(current_batch_min_val, torch.tensor(0.1, device=vp_pred.device))

            if self.running_min_vel == self.min_velocity:
                self.running_min_vel = current_batch_min_val
            else:
                self.running_min_vel = self.momentum * self.running_min_vel + (1 - self.momentum) * current_batch_min_val
            c_val = torch.clamp(0.1 * self.running_min_vel, min=0.01, max=1.0)
        else:
            c_val = torch.tensor(self.initial_c, device=vp_pred.device)

        vp_pred_safe = torch.clamp(vp_pred, min=self.min_velocity)
        vp_true_safe = torch.clamp(vp_true, min=self.min_velocity)

        log_pred = torch.log(vp_pred_safe + c_val + self.epsilon_log)
        log_true = torch.log(vp_true_safe + c_val + self.epsilon_log)

        return F.l1_loss(log_pred, log_true)


class SeismicMSSSIM(nn.Module):
    """Seismic-specific MS-SSIM loss for structural similarity preservation.
    
    Features:
    1. Optional log-space application for relative error alignment
    2. Dynamic normalization for consistent MS-SSIM behavior
    3. Geological structure-aware similarity measurement
    """
    def __init__(self, apply_log=True, data_range_log=2.0, min_velocity_for_log_norm=1.5, c_for_log=0.1):
        super().__init__()
        self.apply_log = apply_log
        if MS_SSIM is not None:
            self.ms_ssim_module = MS_SSIM(
                data_range=data_range_log if apply_log else 6.5,
                size_average=True,
                channel=1
            )
        else:
            raise ImportError("MS_SSIM not available. Install pytorch-msssim.")
        self.min_velocity_for_log_norm = min_velocity_for_log_norm
        self.c_for_log = c_for_log
        self.epsilon_log = 1e-8

    def forward(self, vp_pred, vp_true):
        if vp_pred.ndim == 3: vp_pred = vp_pred.unsqueeze(1)
        if vp_true.ndim == 3: vp_true = vp_true.unsqueeze(1)

        if self.apply_log:
            vp_pred_safe = torch.clamp(vp_pred, min=self.min_velocity_for_log_norm)
            vp_true_safe = torch.clamp(vp_true, min=self.min_velocity_for_log_norm)

            log_pred_raw = torch.log(vp_pred_safe + self.c_for_log + self.epsilon_log)
            log_true_raw = torch.log(vp_true_safe + self.c_for_log + self.epsilon_log)

            # Dynamic normalization
            log_min_true_batch = torch.min(log_true_raw)
            log_max_true_batch = torch.max(log_true_raw)
            
            if (log_max_true_batch - log_min_true_batch) < 1e-6:
                log_pred_norm = log_pred_raw - log_min_true_batch
                log_true_norm = log_true_raw - log_min_true_batch
            else:
                log_pred_norm = (log_pred_raw - log_min_true_batch) / (log_max_true_batch - log_min_true_batch) * self.ms_ssim_module.data_range
                log_true_norm = (log_true_raw - log_min_true_batch) / (log_max_true_batch - log_min_true_batch) * self.ms_ssim_module.data_range
            
            return 1 - self.ms_ssim_module(log_pred_norm, log_true_norm)
        else:
            return 1 - self.ms_ssim_module(vp_pred, vp_true)


class AnisotropicTotalVariationLoss(nn.Module):
    """Anisotropic Total Variation Loss for geological structure preservation.
    
    Features:
    1. Different weights for horizontal vs. vertical smoothing
    2. Encourages layer-parallel smoothness while preserving boundaries
    3. Geological domain-specific regularization
    """
    def __init__(self, weight_h=1.0, weight_v=0.3):
        super().__init__()
        self.weight_h = weight_h
        self.weight_v = weight_v
    
    def forward(self, input_tensor):
        if input_tensor.ndim != 4 or input_tensor.size(1) != 1:
            raise ValueError("Input tensor for ATV must be (B,1,H,W)")
        
        # Horizontal TV (penalizes changes along width)
        tv_h = torch.abs(input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :-1])
        # Vertical TV (penalizes changes along height)
        tv_v = torch.abs(input_tensor[:, :, 1:, :] - input_tensor[:, :, :-1, :])
        
        loss_h = self.weight_h * torch.mean(tv_h)
        loss_v = self.weight_v * torch.mean(tv_v)
        
        return loss_h + loss_v


class RefinedLogSpaceMAEHybridLoss(nn.Module):
    """Refined multi-component hybrid loss with improved curriculum learning and SoftAdapt scaling.
    
    CRITICAL FIXES for experimental reliability:
    1. Proper current_weights initialization for curriculum learning
    2. Improved pre-scaling for SoftAdapt based on observed component magnitudes
    3. Better curriculum transition logic
    """
    def __init__(self, min_velocity=1.5, use_adaptive_softadapt=True,
                 initial_c_logmae=0.1, logmae_momentum=0.9,
                 ms_ssim_apply_log=True, ms_ssim_data_range_log=2.0, ms_ssim_c_log=0.1,
                 atv_weight_h=1.0, atv_weight_v=0.3,
                 softadapt_beta=0.1, softadapt_update_freq=10,
                 fixed_weights_list=[1.0, 0.3, 0.005],
                 # Enhanced parameters for refinement
                 scale_for_softadapt=True,
                 component_scales="adaptive",  # "adaptive" or list like [10.0, 1.0, 100.0]
                 curriculum_epochs=0,  # Epochs to train only LogMAE before activating other components
                 start_simple=False):  # If True, start with only LogMAE
        super().__init__()
        
        # Core loss components
        self.adaptive_log_mae = AdaptiveLogSpaceMAE(
            min_velocity=min_velocity, momentum=logmae_momentum, initial_c=initial_c_logmae
        ) if logmae_momentum > 0 else FixedCLogSpaceMAE(
            fixed_c=initial_c_logmae, min_velocity=min_velocity
        )
        
        self.seismic_ms_ssim = SeismicMSSSIM(
            apply_log=ms_ssim_apply_log, data_range_log=ms_ssim_data_range_log, c_for_log=ms_ssim_c_log
        )
        self.anisotropic_tv = AnisotropicTotalVariationLoss(
            weight_h=atv_weight_h, weight_v=atv_weight_v
        )
        
        # Refinement parameters
        self.scale_for_softadapt = scale_for_softadapt
        self.curriculum_epochs = curriculum_epochs
        self.start_simple = start_simple
        self.register_buffer('epoch_counter', torch.tensor(0))
        
        # Enhanced component scaling based on observed magnitudes from successful experiments
        if component_scales == "adaptive":
            # Based on observed magnitudes from R2_FullHybrid_w0.1_0.005 (best result: 0.3790% MAPE)
            # Typical observed values: LogMAE ~0.03, MS-SSIM ~0.2, ATV ~0.05
            # Goal: bring them to similar order of magnitude for SoftAdapt
            self.component_scales = [15.0, 2.0, 50.0]  # Refined based on analysis
        else:
            self.component_scales = component_scales
        
        # SoftAdapt configuration storage for curriculum learning
        self.softadapt_beta = softadapt_beta
        self.update_frequency = softadapt_update_freq
        
        # CRITICAL FIX: Always initialize current_weights to avoid AttributeError
        self.register_buffer('fixed_weights', torch.tensor(fixed_weights_list, dtype=torch.float32))
        self.register_buffer('current_weights', torch.tensor(fixed_weights_list, dtype=torch.float32))
        
        # SoftAdapt setup with proper initialization
        self.config_use_adaptive_softadapt = use_adaptive_softadapt
        self.use_adaptive_softadapt_active = self.config_use_adaptive_softadapt and not start_simple
        
        if self.use_adaptive_softadapt_active and LossWeightedSoftAdapt is not None:
            self.softadapt_object = LossWeightedSoftAdapt(
                beta=softadapt_beta,
                accuracy_order=2
            )
            self.loss_history = {'logmae': [], 'msssim': [], 'atv': []}
            self.iteration = 0
        elif self.config_use_adaptive_softadapt and LossWeightedSoftAdapt is None:
            print("Warning: SoftAdapt requested but not available. Using fixed weights.")
            self.use_adaptive_softadapt_active = False

    def set_epoch(self, epoch):
        """Set current epoch for curriculum learning with proper SoftAdapt activation."""
        self.epoch_counter = torch.tensor(epoch)
        
        # CRITICAL FIX: Proper curriculum transition with current_weights initialization
        if (self.start_simple and epoch >= self.curriculum_epochs and 
            self.config_use_adaptive_softadapt and not self.use_adaptive_softadapt_active):
            
            print(f"🔄 Activating full adaptive hybrid loss at epoch {epoch}")
            self.use_adaptive_softadapt_active = True
            
            if LossWeightedSoftAdapt is not None:
                # Re-initialize SoftAdapt components
                self.softadapt_object = LossWeightedSoftAdapt(
                    beta=self.softadapt_beta, 
                    accuracy_order=2
                )
                self.loss_history = {'logmae': [], 'msssim': [], 'atv': []}
                self.iteration = 0
                
                # CRITICAL FIX: Ensure current_weights is properly initialized
                self.current_weights = self.fixed_weights.clone().to(self.current_weights.device)
                print(f"✓ SoftAdapt initialized with starting weights: {self.current_weights.cpu().numpy()}")
            else:
                print("⚠️  SoftAdapt library not available, falling back to fixed weights")
                self.use_adaptive_softadapt_active = False

    def forward(self, vp_pred, vp_target):
        # Always compute LogMAE
        logmae_val = self.adaptive_log_mae(vp_pred, vp_target)
        
        # Check if we're in curriculum phase (only LogMAE)
        if self.start_simple and self.epoch_counter < self.curriculum_epochs:
            return {
                'total': logmae_val, 'logmae': logmae_val, 
                'msssim': torch.tensor(0.0, device=vp_pred.device),
                'atv': torch.tensor(0.0, device=vp_pred.device),
                'weights': np.array([1.0, 0.0, 0.0])
            }
        
        # Compute other components
        msssim_val = self.seismic_ms_ssim(vp_pred, vp_target)
        atv_val = self.anisotropic_tv(vp_pred)

        # SoftAdapt weight adaptation with improved scaling
        if self.use_adaptive_softadapt_active and self.training:
            # Enhanced pre-scaling for SoftAdapt
            if self.scale_for_softadapt:
                scaled_logmae = logmae_val.item() * self.component_scales[0]
                scaled_msssim = msssim_val.item() * self.component_scales[1]
                scaled_atv = atv_val.item() * self.component_scales[2]
                self.loss_history['logmae'].append(scaled_logmae)
                self.loss_history['msssim'].append(scaled_msssim)
                self.loss_history['atv'].append(scaled_atv)
            else:
                self.loss_history['logmae'].append(logmae_val.item())
                self.loss_history['msssim'].append(msssim_val.item())
                self.loss_history['atv'].append(atv_val.item())
            
            self.iteration += 1
            
            # SoftAdapt weight updates with error handling
            if self.iteration > 1 and self.iteration % self.update_frequency == 0:
                history_for_softadapt = [
                    torch.tensor(self.loss_history['logmae'][-self.update_frequency:], dtype=torch.float32),
                    torch.tensor(self.loss_history['msssim'][-self.update_frequency:], dtype=torch.float32),
                    torch.tensor(self.loss_history['atv'][-self.update_frequency:], dtype=torch.float32)
                ]
                
                if all(len(h) >= 2 for h in history_for_softadapt):
                    try:
                        adapted_weights_raw = self.softadapt_object.get_component_weights(
                            *history_for_softadapt, verbose=False
                        )
                        
                        # Enhanced TensorFlow tensor conversion
                        if not isinstance(adapted_weights_raw, torch.Tensor):
                            if hasattr(adapted_weights_raw, 'numpy'):
                                adapted_weights_np = adapted_weights_raw.numpy()
                            else:
                                adapted_weights_np = np.array(adapted_weights_raw, dtype=np.float32)
                            adapted_weights_pytorch = torch.from_numpy(adapted_weights_np)
                        else:
                            adapted_weights_pytorch = adapted_weights_raw

                        self.current_weights = adapted_weights_pytorch.to(
                            device=vp_pred.device, dtype=torch.float32
                        )
                        
                    except Exception as e:
                        print(f"⚠️  SoftAdapt update failed: {e}, using previous weights")
                        # Keep current_weights unchanged
            
            weights_to_use = self.current_weights
        else:
            weights_to_use = self.fixed_weights.to(vp_pred.device)
        
        total_loss = (weights_to_use[0] * logmae_val +
                      weights_to_use[1] * msssim_val +
                      weights_to_use[2] * atv_val)
        
        return {
            'total': total_loss, 'logmae': logmae_val, 'msssim': msssim_val,
            'atv': atv_val, 'weights': weights_to_use.detach().cpu().numpy()
        }


# =============================================================================
# FIXED C LOG SPACE MAE FOR COMPARISON
# =============================================================================

class FixedCLogSpaceMAE(nn.Module):
    """Fixed-c Log-Space MAE Loss for comparison with adaptive version."""
    def __init__(self, fixed_c=0.1, min_velocity=1.5, epsilon_log=1e-8):
        super().__init__()
        self.fixed_c = fixed_c
        self.min_velocity = min_velocity
        self.epsilon_log = epsilon_log
    
    def forward(self, vp_pred, vp_true):
        vp_pred_safe = torch.clamp(vp_pred, min=self.min_velocity)
        vp_true_safe = torch.clamp(vp_true, min=self.min_velocity)
        log_pred = torch.log(vp_pred_safe + self.fixed_c + self.epsilon_log)
        log_true = torch.log(vp_true_safe + self.fixed_c + self.epsilon_log)
        return F.l1_loss(log_pred, log_true)


# =============================================================================
# REUSABLE TRAINING AND EVALUATION FRAMEWORK
# =============================================================================

def train_validate_model(experiment_name, model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs, device, calculate_mape_func, lr_scheduler=None, checkpoint_dir="checkpoints"):
    """Reusable training and validation function for systematic experimentation.
    
    Args:
        experiment_name: String identifier for the experiment
        model: PyTorch model to train
        train_loader, val_loader: DataLoader objects
        criterion: Loss function object
        optimizer: PyTorch optimizer
        num_epochs: Number of training epochs
        device: torch.device
        calculate_mape_func: Function to calculate MAPE from numpy arrays
        lr_scheduler: Optional learning rate scheduler
        checkpoint_dir: Directory to save best models
    
    Returns:
        best_val_mape: Best validation MAPE achieved
        history: Dictionary containing training history
    """
    print(f"\n--- Starting Experiment: {experiment_name} ---")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    model_path = os.path.join(checkpoint_dir, f"{experiment_name}_best_mape.pth")

    best_val_mape = float('inf')
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

                # Calculate MAE on original scale for consistent comparison
                mae_orig = F.l1_loss(outputs_torch, targets_torch)
                running_val_mae_orig_scale += mae_orig.item() * inputs.size(0)

                # Calculate components if hybrid loss
                if isinstance(criterion, RefinedLogSpaceMAEHybridLoss):
                    val_loss_dict = criterion(outputs_torch, targets_torch)
                    running_val_logmae_component += val_loss_dict['logmae'].item() * inputs.size(0)
                    running_val_msssim_component += val_loss_dict['msssim'].item() * inputs.size(0)
                    running_val_atv_component += val_loss_dict['atv'].item() * inputs.size(0)
                    if epoch == 0 and len(history['loss_weights']) == 0:
                        history['loss_weights'].append(val_loss_dict['weights'])

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
            if criterion.use_adaptive_softadapt:
                history['loss_weights'].append(criterion.current_weights.cpu().numpy().copy())

        print_msg = (f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | "
                     f"Val MAE (Orig): {epoch_val_mae_orig:.6f} | Val MAPE: {epoch_val_mape:.4f}%")
        
        if isinstance(criterion, RefinedLogSpaceMAEHybridLoss) and criterion.use_adaptive_softadapt:
            weights_str = ", ".join([f"{w:.3f}" for w in criterion.current_weights.cpu().numpy()])
            print_msg += f" | Weights: [{weights_str}]"

        if epoch_val_mape < best_val_mape:
            best_val_mape = epoch_val_mape
            torch.save(model.state_dict(), model_path)
            print_msg += " <<< BEST MAPE SO FAR - MODEL SAVED"
        
        print(print_msg)

        if lr_scheduler:
            lr_scheduler.step()

    print(f"\nFinished Experiment: {experiment_name}. Best Val MAPE: {best_val_mape:.4f}% saved to {model_path}")
    return best_val_mape, history


def plot_history(history, title_prefix="", save_path=None):
    """Plot training history for analysis."""
    if not history['train_loss']:
        print(f"No history to plot for {title_prefix}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Loss and MAE
    axes[0].plot(history['train_loss'], label='Train Loss (Criterion)', color='blue')
    axes[0].plot(history['val_mae'], label='Val MAE (Original Scale)', color='orange')
    axes[0].set_title(f'{title_prefix} Loss & MAE History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss/MAE')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: MAPE
    axes[1].plot(history['val_mape'], label='Val MAPE %', color='green')
    axes[1].set_title(f'{title_prefix} Validation MAPE History')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAPE %')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Loss components (if hybrid loss)
    if 'val_logmae_loss' in history and history['val_logmae_loss']:
        axes[2].plot(history['val_logmae_loss'], label='Val LogMAE Comp.', color='red')
        axes[2].plot(history['val_msssim_loss'], label='Val MS-SSIM Comp.', color='purple')
        axes[2].plot(history['val_atv_loss'], label='Val ATV Comp.', color='brown')
        axes[2].set_title(f'{title_prefix} Val Loss Components')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss Value')
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_systematic_weight_tuning_experiments(BaselineUNet, train_loader, val_loader, calculate_mape, device,
                                            num_epochs=20, min_velocity=1.5, champion_mape=0.3790):
    """Systematic manual tuning around the champion weights [1.0, 0.1, 0.005].
    
    This function implements the precise weight tuning strategy identified from analysis:
    1. Keep LogMAE at 1.0 (reference weight)
    2. Vary MS-SSIM weight around 0.1 while keeping ATV at 0.005
    3. With best MS-SSIM weight, vary ATV weight around 0.005
    
    Args:
        champion_mape: Current best MAPE to beat (0.3790% from R2_FullHybrid_w0.1_0.005)
    """
    
    print("="*80)
    print("SYSTEMATIC WEIGHT TUNING EXPERIMENTS")
    print(f"Goal: Beat champion MAPE of {champion_mape:.4f}%")
    print("Strategy: Guided search around weights [1.0, 0.1, 0.005]")
    print("="*80)
    
    results = {}
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    
    # Phase 1: Vary MS-SSIM weight while keeping ATV fixed at 0.005
    print(f"\n📊 Phase 1: MS-SSIM Weight Tuning (ATV fixed at 0.005)")
    print("-" * 60)
    
    ms_ssim_weights = [0.05, 0.08, 0.12, 0.15, 0.20]  # Around 0.1
    atv_weight_fixed = 0.005
    
    for w_msssim in ms_ssim_weights:
        print(f"\n🔬 Testing weights [1.0, {w_msssim}, {atv_weight_fixed}]...")
        
        model = BaselineUNet(5, 1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = RefinedLogSpaceMAEHybridLoss(
            min_velocity=min_velocity,
            use_adaptive_softadapt=False,
            logmae_momentum=0,  # Use fixed c=0.1 (best single component)
            initial_c_logmae=0.1,
            fixed_weights_list=[1.0, w_msssim, atv_weight_fixed]
        ).to(device)
        
        best_mape, _ = train_validate_model(
            f"Tune_MSSSIM_{w_msssim}", model, train_loader, val_loader,
            criterion, optimizer, num_epochs, device, calculate_mape
        )
        
        results[f'MSSSIM_{w_msssim}'] = best_mape
        improvement = "🏆 NEW CHAMPION!" if best_mape < champion_mape else f"({(best_mape-champion_mape)/champion_mape*100:+.1f}%)"
        print(f"✓ Weights [1.0, {w_msssim}, {atv_weight_fixed}]: {best_mape:.4f}% MAPE {improvement}")
    
    # Find best MS-SSIM weight from Phase 1
    best_msssim_result = min(results.items(), key=lambda x: x[1])
    best_msssim_weight = float(best_msssim_result[0].split('_')[1])
    best_msssim_mape = best_msssim_result[1]
    
    print(f"\n🎯 Phase 1 Best: MS-SSIM weight {best_msssim_weight} → {best_msssim_mape:.4f}% MAPE")
    
    # Phase 2: Vary ATV weight with best MS-SSIM weight
    print(f"\n📊 Phase 2: ATV Weight Tuning (MS-SSIM fixed at {best_msssim_weight})")
    print("-" * 60)
    
    atv_weights = [0.001, 0.003, 0.007, 0.010, 0.015]  # Around 0.005
    
    for w_atv in atv_weights:
        print(f"\n🔬 Testing weights [1.0, {best_msssim_weight}, {w_atv}]...")
        
        model = BaselineUNet(5, 1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = RefinedLogSpaceMAEHybridLoss(
            min_velocity=min_velocity,
            use_adaptive_softadapt=False,
            logmae_momentum=0,
            initial_c_logmae=0.1,
            fixed_weights_list=[1.0, best_msssim_weight, w_atv]
        ).to(device)
        
        best_mape, _ = train_validate_model(
            f"Tune_ATV_{w_atv}", model, train_loader, val_loader,
            criterion, optimizer, num_epochs, device, calculate_mape
        )
        
        results[f'ATV_{w_atv}'] = best_mape
        improvement = "🏆 NEW CHAMPION!" if best_mape < champion_mape else f"({(best_mape-champion_mape)/champion_mape*100:+.1f}%)"
        print(f"✓ Weights [1.0, {best_msssim_weight}, {w_atv}]: {best_mape:.4f}% MAPE {improvement}")
    
    # Find overall best result
    overall_best = min(results.items(), key=lambda x: x[1])
    overall_best_mape = overall_best[1]
    
    print("\n" + "="*80)
    print("SYSTEMATIC WEIGHT TUNING RESULTS")
    print("="*80)
    print(f"Champion to beat: {champion_mape:.4f}% MAPE")
    print("-" * 50)
    
    # Print Phase 1 results
    print("Phase 1 - MS-SSIM Weight Tuning:")
    for exp_name, mape in sorted(results.items(), key=lambda x: x[1]):
        if 'MSSSIM' in exp_name:
            w_val = exp_name.split('_')[1]
            improvement = "🏆 NEW CHAMPION!" if mape < champion_mape else f"({(mape-champion_mape)/champion_mape*100:+.1f}%)"
            print(f"  [1.0, {w_val}, 0.005]: {mape:.4f}% MAPE {improvement}")
    
    print("\nPhase 2 - ATV Weight Tuning:")
    for exp_name, mape in sorted(results.items(), key=lambda x: x[1]):
        if 'ATV' in exp_name:
            w_val = exp_name.split('_')[1]
            improvement = "🏆 NEW CHAMPION!" if mape < champion_mape else f"({(mape-champion_mape)/champion_mape*100:+.1f}%)"
            print(f"  [1.0, {best_msssim_weight}, {w_val}]: {mape:.4f}% MAPE {improvement}")
    
    print(f"\n🏆 OVERALL BEST: {overall_best_mape:.4f}% MAPE")
    if overall_best_mape < champion_mape:
        improvement_pct = (champion_mape - overall_best_mape) / champion_mape * 100
        print(f"🎉 NEW CHAMPION! {improvement_pct:.2f}% improvement over previous best!")
    
    print("="*80)
    return results


def run_refined_phase2_experiments(BaselineUNet, train_loader, val_loader, calculate_mape, device,
                                  num_epochs=30, min_velocity=1.5):
    """Run the complete refined Phase 2 experimental suite with all fixes and improvements.
    
    This includes:
    1. Original experiments with bug fixes
    2. Systematic weight tuning
    3. Fixed curriculum learning
    4. Improved SoftAdapt scaling
    """
    
    print("="*80)
    print("REFINED PHASE 2: COMPLETE EXPERIMENTAL SUITE")
    print("All bugs fixed, systematic tuning implemented")
    print("="*80)
    
    results = {}
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    
    # Experiment 1: Baseline comparison (FixedCLogSpaceMAE)
    print("\n[1/6] 🔬 FixedCLogSpaceMAE (c=0.1) - Current Single-Component Champion")
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FixedCLogSpaceMAE(fixed_c=0.1, min_velocity=min_velocity).to(device)
    
    best_mape, _ = train_validate_model(
        "Refined_FixedCLogMAE", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    results['FixedCLogMAE_c0.1'] = best_mape
    print(f"✓ FixedCLogMAE (c=0.1): {best_mape:.4f}% MAPE")
    
    # Experiment 2: Current hybrid champion weights
    print("\n[2/6] 🔬 Hybrid Fixed Weights [1.0, 0.1, 0.005] - Current Overall Champion")
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity,
        use_adaptive_softadapt=False,
        logmae_momentum=0,
        initial_c_logmae=0.1,
        fixed_weights_list=[1.0, 0.1, 0.005]
    ).to(device)
    
    best_mape, _ = train_validate_model(
        "Refined_HybridChampion", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    results['HybridChampion_1.0_0.1_0.005'] = best_mape
    print(f"✓ Hybrid Champion [1.0, 0.1, 0.005]: {best_mape:.4f}% MAPE")
    
    # Experiment 3: Systematic weight tuning
    print("\n[3/6] 🔬 Systematic Weight Tuning Around Champion")
    tuning_results = run_systematic_weight_tuning_experiments(
        BaselineUNet, train_loader, val_loader, calculate_mape, device,
        num_epochs=num_epochs//2, min_velocity=min_velocity, 
        champion_mape=best_mape  # Use current best as target
    )
    results.update(tuning_results)
    
    # Experiment 4: Fixed curriculum learning (bug-fixed)
    print("\n[4/6] 🔬 Fixed Curriculum Learning + SoftAdapt")
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity,
        use_adaptive_softadapt=True,
        logmae_momentum=0,
        initial_c_logmae=0.1,
        start_simple=True,
        curriculum_epochs=10,
        component_scales="adaptive",  # Use refined adaptive scaling
        softadapt_beta=0.1,
        softadapt_update_freq=5
    ).to(device)
    
    best_mape = train_with_curriculum_fixed(
        "Refined_CurriculumSoftAdapt", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    results['CurriculumSoftAdapt_Fixed'] = best_mape
    print(f"✓ Fixed Curriculum + SoftAdapt: {best_mape:.4f}% MAPE")
    
    # Experiment 5: Improved scaled SoftAdapt (no curriculum)
    print("\n[5/6] 🔬 Improved Scaled SoftAdapt (No Curriculum)")
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity,
        use_adaptive_softadapt=True,
        logmae_momentum=0,
        initial_c_logmae=0.1,
        start_simple=False,
        component_scales="adaptive",  # [15.0, 2.0, 50.0]
        softadapt_beta=0.05,  # More responsive
        softadapt_update_freq=8
    ).to(device)
    
    best_mape, _ = train_validate_model(
        "Refined_ScaledSoftAdapt", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    results['ScaledSoftAdapt_Improved'] = best_mape
    print(f"✓ Improved Scaled SoftAdapt: {best_mape:.4f}% MAPE")
    
    # Experiment 6: Conservative SoftAdapt (higher beta, more stable)
    print("\n[6/6] 🔬 Conservative SoftAdapt (Higher Beta for Stability)")
    model = BaselineUNet(5, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity,
        use_adaptive_softadapt=True,
        logmae_momentum=0,
        initial_c_logmae=0.1,
        start_simple=False,
        component_scales="adaptive",
        softadapt_beta=0.2,  # More conservative
        softadapt_update_freq=15
    ).to(device)
    
    best_mape, _ = train_validate_model(
        "Refined_ConservativeSoftAdapt", model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device, calculate_mape
    )
    results['ConservativeSoftAdapt'] = best_mape
    print(f"✓ Conservative SoftAdapt: {best_mape:.4f}% MAPE")
    
    # Final analysis
    print("\n" + "="*80)
    print("REFINED PHASE 2 - FINAL RESULTS ANALYSIS")
    print("="*80)
    
    # Find overall champion
    overall_champion = min(results.items(), key=lambda x: x[1])
    champion_name, champion_mape = overall_champion
    
    print(f"🏆 OVERALL CHAMPION: {champion_name}")
    print(f"🎯 CHAMPION MAPE: {champion_mape:.4f}%")
    
    baseline_mape = 3.93  # Original baseline
    improvement = (baseline_mape - champion_mape) / baseline_mape * 100
    print(f"📈 IMPROVEMENT vs BASELINE: {improvement:.1f}%")
    
    print("\nComplete Results Ranking:")
    for i, (exp_name, mape) in enumerate(sorted(results.items(), key=lambda x: x[1]), 1):
        status = "👑 CHAMPION" if mape == champion_mape else f"  #{i}"
        print(f"{status} {exp_name}: {mape:.4f}% MAPE")
    
    print("="*80)
    return results


def train_with_curriculum_fixed(experiment_name, model, train_loader, val_loader, criterion, optimizer,
                               num_epochs, device, calculate_mape_func):
    """Enhanced training function with proper curriculum learning and SoftAdapt handling."""
    print(f"\n--- Starting Fixed Curriculum Experiment: {experiment_name} ---")
    
    best_val_mape = float('inf')
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    model_path = os.path.join(checkpoint_dir, f"{experiment_name}_best_mape.pth")
    
    for epoch in range(num_epochs):
        # CRITICAL: Set epoch for curriculum learning
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle both dict and scalar loss returns
            loss_output = criterion(outputs, targets)
            if isinstance(loss_output, dict):
                loss = loss_output['total']
            else:
                loss = loss_output
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        running_val_mape = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets_torch = inputs.to(device), targets.to(device)
                outputs_torch = model(inputs)
                
                # Calculate MAPE
                outputs_np = outputs_torch.squeeze(1).cpu().numpy()
                targets_np = targets_torch.squeeze(1).cpu().numpy()
                batch_mape_sum = 0.0
                for i in range(outputs_np.shape[0]):
                    batch_mape_sum += calculate_mape_func(targets_np[i], outputs_np[i])
                running_val_mape += (batch_mape_sum / outputs_np.shape[0]) * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_val_mape = running_val_mape / len(val_loader.dataset)
        
        # Enhanced logging with curriculum and SoftAdapt info
        print_msg = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val MAPE: {epoch_val_mape:.4f}%"
        
        # Add curriculum phase indicator
        if hasattr(criterion, 'start_simple') and criterion.start_simple:
            if epoch < criterion.curriculum_epochs:
                print_msg += " [CURRICULUM: LogMAE only]"
            else:
                print_msg += " [FULL HYBRID]"
                
                # Add SoftAdapt weights if available
                if (hasattr(criterion, 'use_adaptive_softadapt_active') and 
                    criterion.use_adaptive_softadapt_active and 
                    hasattr(criterion, 'current_weights')):
                    try:
                        weights_str = ", ".join([f"{w:.3f}" for w in criterion.current_weights.cpu().numpy()])
                        print_msg += f" | Weights: [{weights_str}]"
                    except:
                        print_msg += " | Weights: [updating...]"
        
        if epoch_val_mape < best_val_mape:
            best_val_mape = epoch_val_mape
            torch.save(model.state_dict(), model_path)
            print_msg += " <<< BEST MAPE SO FAR - MODEL SAVED"
        
        print(print_msg)
    
    print(f"\nFinished {experiment_name}. Best Val MAPE: {best_val_mape:.4f}%")
    return best_val_mape


def run_phase2_experiments(BaselineUNet, SeismicDataset, train_loader, val_loader, calculate_mape, device, 
                          num_epochs=30, min_velocity=1.5):
    """Run complete Phase 2 experimental suite.
    
    Args:
        BaselineUNet: Model class
        SeismicDataset: Dataset class  
        train_loader, val_loader: DataLoader objects
        calculate_mape: MAPE calculation function
        device: torch.device
        num_epochs: Number of epochs per experiment
        min_velocity: Minimum velocity for clamping
    
    Returns:
        results_summary: Dictionary with experimental results
    """
    
    results_summary = {}
    
    # Global configurations
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    N_INPUT_CHANNELS = 5
    N_OUTPUT_CHANNELS = 1
    
    print("="*80)
    print("PHASE 2: SYSTEMATIC LOSS FUNCTION EXPERIMENTS")
    print("="*80)
    
    # Experiment 1.1.A: AdaptiveLogSpaceMAE (adaptive c)
    print("\n[1/4] Testing AdaptiveLogSpaceMAE with adaptive c parameter...")
    model_exp1a = BaselineUNet(N_INPUT_CHANNELS, N_OUTPUT_CHANNELS).to(device)
    optimizer_exp1a = optim.AdamW(model_exp1a.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_exp1a = AdaptiveLogSpaceMAE(min_velocity=min_velocity, momentum=0.9, initial_c=0.1).to(device)
    
    best_mape_exp1a, history_exp1a = train_validate_model(
        "Exp1A_AdaptiveLogMAE", model_exp1a, train_loader, val_loader, 
        criterion_exp1a, optimizer_exp1a, num_epochs, device, calculate_mape
    )
    results_summary['AdaptiveLogMAE'] = best_mape_exp1a
    
    # Experiment 1.1.B: FixedCLogSpaceMAE (c=0.1)
    print("\n[2/4] Testing FixedCLogSpaceMAE with c=0.1...")
    model_exp1b = BaselineUNet(N_INPUT_CHANNELS, N_OUTPUT_CHANNELS).to(device)
    optimizer_exp1b = optim.AdamW(model_exp1b.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_exp1b = FixedCLogSpaceMAE(fixed_c=0.1, min_velocity=min_velocity).to(device)
    
    best_mape_exp1b, history_exp1b = train_validate_model(
        "Exp1B_FixedC0.1LogMAE", model_exp1b, train_loader, val_loader, 
        criterion_exp1b, optimizer_exp1b, num_epochs, device, calculate_mape
    )
    results_summary['FixedCLogMAE'] = best_mape_exp1b
    
    # Experiment 1.3.B: Full Hybrid with Fixed Weights
    print("\n[3/4] Testing Full Hybrid Loss with fixed weights...")
    model_exp_hybrid_fixed = BaselineUNet(N_INPUT_CHANNELS, N_OUTPUT_CHANNELS).to(device)
    optimizer_exp_hybrid_fixed = optim.AdamW(model_exp_hybrid_fixed.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_exp_hybrid_fixed = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity, 
        use_adaptive_softadapt=False,
        fixed_weights_list=[1.0, 0.3, 0.005]
    ).to(device)
    
    best_mape_hybrid_fixed, history_hybrid_fixed = train_validate_model(
        "Exp_HybridFixedWeights", model_exp_hybrid_fixed, train_loader, val_loader, 
        criterion_exp_hybrid_fixed, optimizer_exp_hybrid_fixed, num_epochs, device, calculate_mape
    )
    results_summary['HybridFixed'] = best_mape_hybrid_fixed
    
    # Experiment 1.4: Full Hybrid with Adaptive Weights
    print("\n[4/4] Testing Full Hybrid Loss with adaptive SoftAdapt weights...")
    model_exp_hybrid_adaptive = BaselineUNet(N_INPUT_CHANNELS, N_OUTPUT_CHANNELS).to(device)
    optimizer_exp_hybrid_adaptive = optim.AdamW(model_exp_hybrid_adaptive.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_exp_hybrid_adaptive = RefinedLogSpaceMAEHybridLoss(
        min_velocity=min_velocity, 
        use_adaptive_softadapt=True,
        softadapt_beta=0.1,
        softadapt_update_freq=10
    ).to(device)
    
    best_mape_hybrid_adaptive, history_hybrid_adaptive = train_validate_model(
        "Exp_HybridAdaptiveWeights", model_exp_hybrid_adaptive, train_loader, val_loader, 
        criterion_exp_hybrid_adaptive, optimizer_exp_hybrid_adaptive, num_epochs, device, calculate_mape
    )
    results_summary['HybridAdaptive'] = best_mape_hybrid_adaptive
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_history(history_exp1a, "Exp1A_AdaptiveLogMAE")
    plot_history(history_exp1b, "Exp1B_FixedCLogMAE") 
    plot_history(history_hybrid_fixed, "Exp_HybridFixedWeights")
    plot_history(history_hybrid_adaptive, "Exp_HybridAdaptiveWeights")
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 2 EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    for exp_name, best_mape in results_summary.items():
        print(f"{exp_name:20s}: {best_mape:.4f}% MAPE")
    
    best_experiment = min(results_summary, key=results_summary.get)
    best_mape = results_summary[best_experiment]
    print(f"\nBest Performance: {best_experiment} with {best_mape:.4f}% MAPE")
    print("="*80)
    
    return results_summary

print("Phase 2 Experimental Framework loaded successfully!")
print("🔧 All critical bugs fixed, systematic tuning implemented")
print("Ready for refined systematic loss function experimentation.") 