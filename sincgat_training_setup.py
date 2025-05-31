"""
SincGAT-UNet Training Setup

This module provides comprehensive training utilities for the SincGAT-UNet model,
including champion loss functions, mixed precision training, and complete training loops.

Key Components:
1. Champion Hybrid Loss Functions
2. Mixed Precision Training Support  
3. A100 Stability Configuration
4. Complete Training and Validation Loops
5. Model Saving and Loading Utilities

Based on research findings for optimal seismic velocity model prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import math
from tqdm import tqdm
import os
import json
from datetime import datetime
import warnings

# Import the complete model
from complete_sincgat_unet_integration import CompleteSincGAT_UNet, configure_a100_stability


# =====================================
# CHAMPION LOSS FUNCTIONS
# =====================================

class StabilizedSeismicMSSSIM(nn.Module):
    """
    Stabilized MS-SSIM implementation for seismic velocity models.
    Based on champion research findings with stabilization for training.
    """
    def __init__(self, data_range=1.0, size_average=True, channel=1, 
                 spatial_dims=2, nonnegative_ssim=True):
        super().__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel
        self.nonnegative_ssim = nonnegative_ssim
        
        # MS-SSIM parameters
        self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32)
        self.levels = len(self.weights)
        
        # Gaussian kernel parameters
        self.kernel_size = 11
        self.sigma = 1.5
        self.gaussian_kernel = self._create_gaussian_kernel()
        
    def _create_gaussian_kernel(self):
        """Create Gaussian kernel for SSIM computation"""
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords -= self.kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        
        # Create 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(self.channel, 1, self.kernel_size, self.kernel_size)
        
        return kernel
    
    def _gaussian_filter(self, x, kernel):
        """Apply Gaussian filter"""
        padding = self.kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=self.channel)
    
    def _ssim_per_channel(self, x, y, kernel):
        """Compute SSIM for each channel"""
        mu_x = self._gaussian_filter(x, kernel)
        mu_y = self._gaussian_filter(y, kernel)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = self._gaussian_filter(x ** 2, kernel) - mu_x_sq
        sigma_y_sq = self._gaussian_filter(y ** 2, kernel) - mu_y_sq
        sigma_xy = self._gaussian_filter(x * y, kernel) - mu_xy
        
        # SSIM constants for stability
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        
        # SSIM computation
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean([2, 3])  # Average over spatial dimensions
    
    def forward(self, pred, target):
        """Compute MS-SSIM loss"""
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        device = pred.device
        kernel = self.gaussian_kernel.to(device)
        weights = self.weights.to(device)
        
        levels_ssim = []
        x, y = pred, target
        
        for i in range(self.levels):
            ssim_val = self._ssim_per_channel(x, y, kernel)
            levels_ssim.append(ssim_val)
            
            # Downsample for next level (except last)
            if i < self.levels - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)
        
        # Compute weighted MS-SSIM
        ms_ssim = torch.stack(levels_ssim)
        ms_ssim = torch.prod(ms_ssim ** weights.view(-1, 1))
        
        # Convert to loss (1 - MS-SSIM) and ensure non-negative if requested
        loss = 1 - ms_ssim
        if self.nonnegative_ssim:
            loss = torch.clamp(loss, min=0.0)
        
        return loss


class AdaptiveLogSpaceMAE(nn.Module):
    """
    Adaptive Log-Space MAE Loss with momentum-based c parameter adaptation.
    When momentum=0, acts as FixedCLogSpaceMAE with c=initial_c.
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
        # For fixed c (champion config), momentum=0 means use initial_c
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
            # Fixed c value for champion configuration
            c_val = torch.tensor(self.initial_c, device=vp_pred.device)

        vp_pred_safe = torch.clamp(vp_pred, min=self.min_velocity)
        vp_true_safe = torch.clamp(vp_true, min=self.min_velocity)

        log_pred = torch.log(vp_pred_safe + c_val + self.epsilon_log)
        log_true = torch.log(vp_true_safe + c_val + self.epsilon_log)

        return F.l1_loss(log_pred, log_true)


class RefinedLogSpaceMAEHybridLoss(nn.Module):
    """
    Champion hybrid loss function with fixed weights based on research findings.
    
    This implementation uses the champion configuration:
    - Fixed weights: [1.0, 0.12, 0.007] 
    - No adaptive SoftAdapt
    - FixedCLogMAE with c=0.1, momentum=0
    - StabilizedSeismicMSSSIM
    
    CRITICAL FIX: Now uses proper FixedCLogSpaceMAE instead of MSE for primary term.
    """
    
    def __init__(self, 
                 fixed_weights_list=[1.0, 0.12, 0.007],
                 use_adaptive_softadapt=False,
                 logmae_momentum=0,  # FixedCLogMAE
                 logmae_c=0.1,
                 min_velocity=1.5,
                 epsilon=1e-8):
        super().__init__()
        
        self.fixed_weights = torch.tensor(fixed_weights_list, dtype=torch.float32)
        self.use_adaptive_softadapt = use_adaptive_softadapt
        self.logmae_momentum = logmae_momentum
        self.logmae_c = logmae_c
        self.min_velocity = min_velocity
        self.epsilon = epsilon
        
        # CRITICAL FIX: Use AdaptiveLogSpaceMAE configured for fixed c=0.1
        self.fixed_c_logmae = AdaptiveLogSpaceMAE(
            min_velocity=min_velocity,
            momentum=logmae_momentum,  # 0 for fixed c
            initial_c=logmae_c,       # 0.1 for champion config
            adaptive_c_on_true_only=True
        )
        
        # Initialize other loss components
        self.ms_ssim_loss = StabilizedSeismicMSSSIM()
        
        # For tracking (not used in forward pass with fixed weights)
        self.loss_history = []
        
        print(f"🎯 Champion Hybrid Loss initialized:")
        print(f"   Fixed weights: {fixed_weights_list}")
        print(f"   FixedCLogMAE: c={logmae_c}, momentum={logmae_momentum}")
        print(f"   Min velocity: {min_velocity}")

    def forward(self, pred, target):
        """
        Compute champion hybrid loss
        
        Returns:
            dict: Contains 'total' loss and individual component losses
        """
        device = pred.device
        weights = self.fixed_weights.to(device)
        
        # Compute individual loss components using CHAMPION formulation
        logmae = self.fixed_c_logmae(pred, target)  # Primary fidelity term
        ms_ssim = self.ms_ssim_loss(pred, target)   # Structural similarity
        
        # Anisotropic Total Variation (compute directly here for simplicity)
        if pred.ndim != 4 or pred.size(1) != 1:
            # Ensure correct shape for ATV
            if pred.ndim == 3:
                pred_atv = pred.unsqueeze(1)
                target_atv = target.unsqueeze(1)
            else:
                pred_atv = pred
                target_atv = target
        else:
            pred_atv = pred
            target_atv = target
            
        # Compute ATV on predictions (structural regularization)
        tv_h = torch.abs(pred_atv[:, :, :, 1:] - pred_atv[:, :, :, :-1])
        tv_v = torch.abs(pred_atv[:, :, 1:, :] - pred_atv[:, :, :-1, :])
        atv = torch.mean(tv_h) + 0.3 * torch.mean(tv_v)  # Anisotropic weights
        
        # Weighted combination (champion fixed weights)
        # weights[0] * LogMAE + weights[1] * MS-SSIM + weights[2] * ATV
        total_loss = weights[0] * logmae + weights[1] * ms_ssim + weights[2] * atv
        
        # Store for tracking
        loss_dict = {
            'total': total_loss,
            'logmae': logmae.item(),  # FIXED: Now using actual LogMAE
            'ms_ssim': ms_ssim.item(),
            'atv': atv.item(),
            'weights': weights.detach().cpu().numpy().tolist()
        }
        
        # Update history for analysis
        self.loss_history.append(loss_dict.copy())
        
        return loss_dict


# =====================================
# EVALUATION METRICS
# =====================================

class SeismicEvaluationMetrics:
    """
    Comprehensive evaluation metrics for seismic velocity model prediction.
    Includes MAPE, MAE, RMSE, and relative metrics.
    """
    
    def __init__(self, min_velocity=1.5, epsilon=1e-8):
        self.min_velocity = min_velocity
        self.epsilon = epsilon
    
    def _safe_relative_error(self, pred, target):
        """Compute relative error with safety checks"""
        # Ensure targets are above minimum threshold
        target_safe = torch.clamp(target, min=self.min_velocity)
        return torch.abs(pred - target) / (target_safe + self.epsilon)
    
    def compute_mape(self, pred, target):
        """Mean Absolute Percentage Error"""
        relative_error = self._safe_relative_error(pred, target)
        mape = torch.mean(relative_error) * 100
        return mape.item()
    
    def compute_mae_original_scale(self, pred, target):
        """MAE in original velocity scale (km/s)"""
        mae = torch.mean(torch.abs(pred - target))
        return mae.item()
    
    def compute_rmse_original_scale(self, pred, target):
        """RMSE in original velocity scale (km/s)"""
        mse = torch.mean((pred - target) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()
    
    def compute_r2_score(self, pred, target):
        """R-squared coefficient of determination"""
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + self.epsilon))
        return r2.item()
    
    def compute_all_metrics(self, pred, target):
        """Compute all evaluation metrics"""
        metrics = {
            'mape': self.compute_mape(pred, target),
            'mae_original': self.compute_mae_original_scale(pred, target),
            'rmse_original': self.compute_rmse_original_scale(pred, target),
            'r2_score': self.compute_r2_score(pred, target)
        }
        return metrics


# =====================================
# TRAINING UTILITIES
# =====================================

class SincGATTrainer:
    """
    Complete trainer for SincGAT-UNet model with all champion optimizations.
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 use_mixed_precision=True,
                 save_dir='./checkpoints',
                 model_name='sincgat_unet',
                 lr_scheduler=None):
        
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.model_name = model_name
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize champion loss and metrics
        self.criterion = RefinedLogSpaceMAEHybridLoss()
        self.evaluator = SeismicEvaluationMetrics()
        
        # Initialize optimizer (AdamW as per champion config)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (optional)
        self.lr_scheduler = lr_scheduler
        
        # Mixed precision setup
        self.use_mixed_precision = use_mixed_precision
        if use_mixed_precision:
            self.scaler = GradScaler()
            self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"✅ Mixed precision enabled with {self.autocast_dtype}")
        
        # Training state
        self.current_epoch = 0
        self.best_val_mape = float('inf')
        self.training_history = []
        
        print(f"🚀 SincGAT Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Mixed precision: {use_mixed_precision}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress bar
        train_pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast(dtype=self.autocast_dtype):
                    outputs = self.model(inputs)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['total']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total']
                
                # Standard backward pass
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LogMAE': f'{loss_dict["logmae"]:.4f}',
                'MS-SSIM': f'{loss_dict["ms_ssim"]:.4f}',
                'ATV': f'{loss_dict["atv"]:.4f}'
            })
        
        avg_train_loss = total_loss / num_batches
        return avg_train_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        val_pbar = tqdm(self.val_loader, desc=f'Validation Epoch {self.current_epoch}')
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with autocast(dtype=self.autocast_dtype):
                        outputs = self.model(inputs)
                        loss_dict = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss_dict = self.criterion(outputs, targets)
                
                total_loss += loss_dict['total'].item()
                
                # Compute evaluation metrics
                metrics = self.evaluator.compute_all_metrics(outputs, targets)
                all_metrics.append(metrics)
                
                # Update progress bar with current batch metrics
                val_pbar.set_postfix({
                    'Val Loss': f'{loss_dict["total"].item():.4f}',
                    'MAPE': f'{metrics["mape"]:.4f}%',
                    'MAE': f'{metrics["mae_original"]:.4f}',
                    'R²': f'{metrics["r2_score"]:.4f}'
                })
        
        # Aggregate metrics
        avg_val_loss = total_loss / len(self.val_loader)
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_val_loss, avg_metrics
    
    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mape': self.best_val_mape,
            'metrics': metrics,
            'training_history': self.training_history,
            'model_config': {
                'sample_rate': self.model.sample_rate,
                'num_shots': self.model.num_shots,
                'time_samples': self.model.time_samples,
                'num_receivers': self.model.num_receivers
            }
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved (MAPE: {metrics['mape']:.4f}%)")
    
    def train(self, num_epochs, save_every=5):
        """Complete training loop"""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
        # Configure A100 stability
        if 'cuda' in str(self.device):
            configure_a100_stability()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            # Check for improvement
            is_best = val_metrics['mape'] < self.best_val_mape
            if is_best:
                self.best_val_mape = val_metrics['mape']
            
            # Log epoch results
            epoch_summary = {
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            }
            self.training_history.append(epoch_summary)
            
            print(f"\n📊 Epoch {self.current_epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val MAPE: {val_metrics['mape']:.4f}% {'🎯 NEW BEST!' if is_best else ''}")
            print(f"   Val MAE: {val_metrics['mae_original']:.4f}")
            print(f"   Val R²: {val_metrics['r2_score']:.4f}")
            
            # Save checkpoint
            if self.current_epoch % save_every == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)
            
            # Step learning rate scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation MAPE: {self.best_val_mape:.4f}%")
        print(f"   Final model saved to: {self.save_dir}")


# =====================================
# QUICK TRAINING SETUP FUNCTION
# =====================================

def setup_sincgat_training(train_loader, val_loader, 
                          sample_rate=500,
                          device='cuda',
                          learning_rate=1e-4,
                          save_dir='./sincgat_checkpoints',
                          use_lr_scheduler=True,
                          T_0=10,  # CosineAnnealingWarmRestarts period
                          T_mult=2):  # Period multiplier
    """
    Quick setup function for SincGAT-UNet training with champion configuration.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader  
        sample_rate: Sampling rate of seismic data (CRITICAL!)
        device: Training device
        learning_rate: Learning rate for AdamW
        save_dir: Directory to save checkpoints
        use_lr_scheduler: Whether to use CosineAnnealingWarmRestarts
        T_0: Initial restart period for scheduler
        T_mult: Period multiplier for scheduler
    
    Returns:
        trainer: Configured SincGATTrainer ready for training
    """
    print("🔧 Setting up SincGAT-UNet training...")
    
    # Create model with correct sample rate
    model = CompleteSincGAT_UNet(
        sample_rate=sample_rate,  # CRITICAL: Set from dataset metadata
        num_receivers=31,
        time_samples=10001,
        num_shots=5,
        sinc_out_channels=40,
        shot_embedding_dim=128,
        gat_hidden_per_head=32,
        gat_num_heads=4,
        fused_embedding_dim=128,
        n_unet_output_channels=1
    )
    
    # Create trainer with champion configuration
    trainer = SincGATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=0.01,  # Champion AdamW config
        use_mixed_precision=True,
        save_dir=save_dir,
        model_name='sincgat_unet_champion'
    )
    
    # Create learning rate scheduler after trainer (uses trainer's optimizer)
    if use_lr_scheduler:
        trainer.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate * 0.01
        )
        print(f"📊 LR Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
    
    print("✅ SincGAT-UNet training setup complete!")
    return trainer


if __name__ == "__main__":
    print("="*80)
    print("SINCGAT-UNET TRAINING SETUP")
    print("="*80)
    print("This module provides comprehensive training utilities.")
    print("Use setup_sincgat_training() to quickly configure training.")
    print("\nKey features:")
    print("✅ Champion hybrid loss (fixed weights)")
    print("✅ Mixed precision training")
    print("✅ A100 stability configuration")
    print("✅ Comprehensive evaluation metrics")
    print("✅ Automatic checkpointing")
    print("✅ Progress tracking and logging")