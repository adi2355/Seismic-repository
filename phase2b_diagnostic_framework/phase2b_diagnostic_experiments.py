"""
Phase 2b Diagnostic Experiments Framework

This module implements systematic diagnostic experiments to address the Phase 2b 
stagnation issue where full fine-tuning (unfreezing U-Net) underperforms 
Phase 2a (frontend frozen) training.

Target: Break through the 0.0893% MAPE barrier achieved in Phase 2a.

Key Experiments:
1. Ultra-low U-Net learning rates (addressing potential instability)
2. FiLM regularization strength tuning (λ_gamma_res, λ_beta_res)
3. LR scheduler optimization (patience, factor, type)
4. Extended Phase 2a training (without Phase 2b)

Starting Point: Best Phase 2a checkpoint (0.0893% MAPE with 2-layer FiLM)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import existing components
from complete_sincgat_unet_integration import CompleteSincGAT_UNet, configure_a100_stability
from phase2_experimental_framework import (
    RefinedLogSpaceMAEHybridLoss, 
    calculate_film_reg_loss,
    monitor_film_parameters
)


class Phase2bDiagnosticFramework:
    """
    Systematic framework for diagnosing and optimizing Phase 2b training.
    
    Addresses the critical issue where Phase 2b (full fine-tuning) consistently
    underperforms Phase 2a (frontend frozen) across multiple FiLM experiments.
    """
    
    def __init__(self, 
                 base_checkpoint_path: str,
                 experiment_base_name: str = "Phase2b_Diagnostic",
                 results_dir: str = "experiment_results",
                 checkpoint_dir: str = "checkpoints",
                 device: Optional[torch.device] = None):
        """
        Initialize diagnostic framework.
        
        Args:
            base_checkpoint_path: Path to best Phase 2a checkpoint (0.0893% target)
            experiment_base_name: Base name for diagnostic experiments
            results_dir: Directory for storing experimental results
            checkpoint_dir: Directory for model checkpoints
            device: Training device (auto-detected if None)
        """
        self.base_checkpoint_path = base_checkpoint_path
        self.experiment_base_name = experiment_base_name
        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize tracking
        self.diagnostic_results = []
        self.best_overall_mape = float('inf')
        self.best_experiment_config = None
        
        # Configure A100 stability
        if 'cuda' in str(self.device):
            configure_a100_stability()
            print("✅ A100 stability configured for diagnostics")
    
    def load_base_model(self, film_generator_type: str = '2_layer') -> CompleteSincGAT_UNet:
        """
        Load the base model from Phase 2a checkpoint.
        
        Args:
            film_generator_type: Type of FiLM generator ('linear' or '2_layer')
            
        Returns:
            Loaded CompleteSincGAT_UNet model
        """
        print(f"🔄 Loading base model from: {self.base_checkpoint_path}")
        
        # Create model with same configuration as Phase 2a
        model = CompleteSincGAT_UNet(
            sample_rate=10001,
            num_receivers=31,
            time_samples=10001,
            num_shots=5,
            # Optimized SincNet parameters
            sinc_out_channels=60,
            sinc_kernel_size=1001,
            sinc_stride=1,
            sinc_min_low_hz=40,
            sinc_max_learnable_hz=1000,
            sinc_min_band_hz=10,
            sinc_window_func='blackman',
            sinc_init_type='logarithmic',
            # GAT parameters
            shot_embedding_dim=128,
            gat_hidden_per_head=32,
            gat_num_heads=4,
            gat_layers=1,
            gat_dropout_feat=0.3,
            gat_dropout_attn=0.2,
            fused_embedding_dim=128,
            # U-Net parameters
            n_unet_output_channels=1,
            unet_bilinear=True,
            unet_bottleneck_channels=512,
            # FiLM parameters
            film_context_dim=128,
            film_target_channels=512,
            film_generator_mlp_type=film_generator_type,
            film_mlp_hidden_dim=256
        ).to(self.device)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.base_checkpoint_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                base_mape = checkpoint.get('best_val_mape', 'unknown')
                print(f"✅ Loaded model with Phase 2a MAPE: {base_mape}")
            else:
                model.load_state_dict(checkpoint, strict=True)
                print(f"✅ Loaded model state dict")
                
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise e
            
        return model
    
    def setup_data_loaders(self, batch_size: int = 4):
        """Setup training and validation data loaders."""
        print(f"📊 Setting up data loaders (batch_size={batch_size})")
        
        # Import data loader setup from main notebook
        try:
            from main_898of_0_898model_speed_and_structure_starter_notebook import setup_phase2_data_loaders
            train_loader, val_loader = setup_phase2_data_loaders(
                batch_size=batch_size, 
                num_workers=0
            )
            return train_loader, val_loader
        except ImportError:
            print("⚠️ Using dummy data loaders for testing")
            # Create dummy data loaders for testing
            from torch.utils.data import DataLoader, TensorDataset
            
            # Dummy data matching expected shapes
            dummy_inputs = torch.randn(32, 5, 10001, 31)  # (B, 5_shots, time, receivers)
            dummy_targets = torch.randn(32, 300, 1259)     # (B, H, W) velocity maps
            
            dataset = TensorDataset(dummy_inputs, dummy_targets)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader
    
    def create_criterion(self, 
                        lambda_gamma_res: float = 0.005,
                        lambda_beta_res: float = 0.0005,
                        use_film_reg: bool = True) -> RefinedLogSpaceMAEHybridLoss:
        """
        Create FiLM-aware loss criterion.
        
        Args:
            lambda_gamma_res: Regularization strength for gamma_res
            lambda_beta_res: Regularization strength for beta_res
            use_film_reg: Enable FiLM regularization
            
        Returns:
            Configured loss criterion
        """
        criterion = RefinedLogSpaceMAEHybridLoss(
            min_velocity=1.5,
            use_adaptive_softadapt=False,
            logmae_momentum=0,
            initial_c_logmae=0.1,
            fixed_weights_list=[1.0, 0.12, 0.007],  # Champion weights
            start_simple=True,
            curriculum_epochs=2,
            use_film_reg=use_film_reg,
            lambda_gamma_res=lambda_gamma_res,
            lambda_beta_res=lambda_beta_res
        ).to(self.device)
        
        return criterion
    
    def create_phase2b_optimizer(self, 
                                model: CompleteSincGAT_UNet,
                                lr_frontend: float = 5e-5,
                                lr_unet: float = 1e-5,
                                lr_film_generator: float = 1e-4,
                                weight_decay: float = 0.01,
                                weight_decay_film: float = 1e-3) -> torch.optim.AdamW:
        """
        Create Phase 2b optimizer with differential learning rates.
        
        Args:
            model: The model to optimize
            lr_frontend: Learning rate for frontend components (SincNet, GAT)
            lr_unet: Learning rate for U-Net (key parameter for diagnostics)
            lr_film_generator: Learning rate for FiLM generator
            weight_decay: Standard weight decay
            weight_decay_film: Weight decay for FiLM parameters
            
        Returns:
            Configured optimizer with parameter groups
        """
        # Group parameters by component
        film_generator_params = []
        if hasattr(model, 'film_bottleneck_modulator') and model.film_bottleneck_modulator is not None:
            film_generator_params = list(model.film_bottleneck_modulator.parameters())
        
        sincnet_encoder_params = []
        if hasattr(model, 'shot_encoder'):
            sincnet_encoder_params = list(model.shot_encoder.parameters())
            
        gat_params = []
        if hasattr(model, 'gat_fusion'):
            gat_params = list(model.gat_fusion.parameters())
            
        gat_context_norm_params = []
        if hasattr(model, 'gat_context_layernorm'):
            gat_context_norm_params = list(model.gat_context_layernorm.parameters())
            
        unet_params = []
        if hasattr(model, 'unet'):
            unet_params = list(model.unet.parameters())
        
        # Create optimizer with differential learning rates
        optimizer = torch.optim.AdamW([
            {
                'params': sincnet_encoder_params,
                'lr': lr_frontend,
                'weight_decay': weight_decay,
                'group_name': 'SincNet',
                'apply_warmup': True
            },
            {
                'params': gat_params,
                'lr': lr_frontend,
                'weight_decay': weight_decay,
                'group_name': 'GAT',
                'apply_warmup': True
            },
            {
                'params': gat_context_norm_params,
                'lr': lr_film_generator,
                'weight_decay': weight_decay,
                'group_name': 'GAT_Norm',
                'apply_warmup': True
            },
            {
                'params': film_generator_params,
                'lr': lr_film_generator,
                'weight_decay': weight_decay_film,
                'group_name': 'FiLM',
                'apply_warmup': True
            },
            {
                'params': unet_params,
                'lr': lr_unet,  # KEY DIAGNOSTIC PARAMETER
                'weight_decay': weight_decay,
                'group_name': 'U-Net',
                'apply_warmup': False  # No warmup for pretrained U-Net
            }
        ])
        
        return optimizer
    
    def create_lr_scheduler(self, 
                           optimizer: torch.optim.Optimizer,
                           scheduler_type: str = 'ReduceLROnPlateau',
                           patience: int = 5,
                           factor: float = 0.5,
                           **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler for Phase 2b.
        
        Args:
            optimizer: The optimizer to schedule
            scheduler_type: Type of scheduler ('ReduceLROnPlateau', 'CosineAnnealingLR')
            patience: Patience for ReduceLROnPlateau
            factor: Factor for ReduceLROnPlateau
            **kwargs: Additional scheduler arguments
            
        Returns:
            Configured scheduler or None
        """
        if scheduler_type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=factor, 
                patience=patience, 
                verbose=True
            )
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = kwargs.get('T_max', 30)
            eta_min = kwargs.get('eta_min', 1e-7)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=T_max, 
                eta_min=eta_min
            )
        else:
            return None
    
    def calculate_mape(self, pred, target, min_velocity=1.5, epsilon=1e-8):
        """Calculate MAPE for diagnostic experiments."""
        # Simple MAPE calculation for diagnostics
        pred_np = pred if isinstance(pred, np.ndarray) else pred.cpu().numpy()
        target_np = target if isinstance(target, np.ndarray) else target.cpu().numpy()
        
        # Clip predictions to avoid extreme errors
        pred_clipped = np.clip(pred_np, min_velocity, 10.0)
        target_clipped = np.clip(target_np, min_velocity, 10.0)
        
        # Calculate MAPE
        mape = np.mean(np.abs((target_clipped - pred_clipped) / (target_clipped + epsilon))) * 100
        return mape
    
    def simple_training_loop(self,
                           model,
                           train_loader,
                           val_loader,
                           criterion,
                           optimizer,
                           num_epochs,
                           lr_scheduler=None,
                           experiment_name="diagnostic"):
        """
        Simplified training loop for diagnostic experiments.
        
        This is a streamlined version focusing on the core Phase 2b diagnostic needs.
        """
        print(f"🏋️ Starting training: {experiment_name}")
        
        best_val_mape = float('inf')
        history = {'train_loss': [], 'val_mape': []}
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Calculate loss (handle both dict and tensor returns)
                if hasattr(criterion, 'forward'):
                    loss_result = criterion(outputs, targets, model_for_film_params=model)
                    if isinstance(loss_result, dict):
                        loss = loss_result.get('total', loss_result.get('loss', list(loss_result.values())[0]))
                    else:
                        loss = loss_result
                else:
                    loss = criterion(outputs, targets)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            # Validation phase
            model.eval()
            val_mape_total = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    
                    # Calculate MAPE for each sample in batch
                    outputs_np = outputs.squeeze(1).cpu().numpy() if outputs.dim() > 3 else outputs.cpu().numpy()
                    targets_np = targets.squeeze(1).cpu().numpy() if targets.dim() > 2 else targets.cpu().numpy()
                    
                    for i in range(outputs_np.shape[0]):
                        mape = self.calculate_mape(outputs_np[i], targets_np[i])
                        val_mape_total += mape
                        val_samples += 1
            
            epoch_train_loss = train_loss / num_batches
            epoch_val_mape = val_mape_total / val_samples if val_samples > 0 else float('inf')
            
            history['train_loss'].append(epoch_train_loss)
            history['val_mape'].append(epoch_val_mape)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_train_loss:.6f}, Val MAPE={epoch_val_mape:.6f}%")
            
            # Update best model
            if epoch_val_mape < best_val_mape:
                best_val_mape = epoch_val_mape
                # Save checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_best.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_mape': best_val_mape,
                    'history': history
                }, checkpoint_path)
            
            # Learning rate scheduling
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(epoch_val_mape)
                else:
                    lr_scheduler.step()
        
        return best_val_mape, history
    
    def run_single_diagnostic_experiment(self,
                                       experiment_config: Dict,
                                       num_epochs: int = 30) -> Dict:
        """
        Run a single diagnostic experiment with specified configuration.
        
        Args:
            experiment_config: Configuration dict with experimental parameters
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary with experiment results
        """
        experiment_id = experiment_config.get('experiment_id', f'exp_{len(self.diagnostic_results)}')
        experiment_name = f"{self.experiment_base_name}_{experiment_id}"
        
        print(f"\n{'='*80}")
        print(f"🧪 DIAGNOSTIC EXPERIMENT: {experiment_id}")
        print(f"{'='*80}")
        print(f"Configuration:")
        for key, value in experiment_config.items():
            if key != 'experiment_id':
                print(f"  {key}: {value}")
        
        try:
            # Load base model
            film_type = experiment_config.get('film_generator_type', '2_layer')
            model = self.load_base_model(film_type)
            
            # Ensure U-Net is unfrozen for Phase 2b (unless extended Phase 2a)
            if not experiment_config.get('keep_unet_frozen', False):
                for param in model.unet.parameters():
                    param.requires_grad = True
                print("   U-Net parameters unfrozen for Phase 2b")
            else:
                for param in model.unet.parameters():
                    param.requires_grad = False
                print("   U-Net parameters kept frozen")
            
            # Setup data loaders
            batch_size = experiment_config.get('batch_size', 4)
            train_loader, val_loader = self.setup_data_loaders(batch_size)
            
            # Create criterion
            criterion = self.create_criterion(
                lambda_gamma_res=experiment_config.get('lambda_gamma_res', 0.005),
                lambda_beta_res=experiment_config.get('lambda_beta_res', 0.0005),
                use_film_reg=experiment_config.get('use_film_reg', True)
            )
            
            # Create optimizer
            optimizer = self.create_phase2b_optimizer(
                model=model,
                lr_frontend=experiment_config.get('lr_frontend', 5e-5),
                lr_unet=experiment_config.get('lr_unet', 1e-5),  # KEY DIAGNOSTIC PARAMETER
                lr_film_generator=experiment_config.get('lr_film_generator', 1e-4),
                weight_decay=experiment_config.get('weight_decay', 0.01),
                weight_decay_film=experiment_config.get('weight_decay_film', 1e-3)
            )
            
            # Create scheduler
            scheduler = self.create_lr_scheduler(
                optimizer=optimizer,
                scheduler_type=experiment_config.get('scheduler_type', 'ReduceLROnPlateau'),
                patience=experiment_config.get('scheduler_patience', 5),
                factor=experiment_config.get('scheduler_factor', 0.5),
                T_max=num_epochs
            )
            
            # Run training
            best_mape, history = self.simple_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                lr_scheduler=scheduler,
                experiment_name=experiment_name
            )
            
            # Record results
            result = {
                'experiment_id': experiment_id,
                'config': experiment_config,
                'best_mape': best_mape,
                'training_history': history,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            print(f"\n✅ Experiment {experiment_id} completed successfully")
            print(f"   Best MAPE: {best_mape:.6f}%")
            
            # Update best overall result
            if best_mape < self.best_overall_mape:
                self.best_overall_mape = best_mape
                self.best_experiment_config = experiment_config.copy()
                print(f"🏆 NEW BEST MAPE: {best_mape:.6f}% (Experiment: {experiment_id})")
            
        except Exception as e:
            print(f"❌ Experiment {experiment_id} failed: {e}")
            result = {
                'experiment_id': experiment_id,
                'config': experiment_config,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
        
        self.diagnostic_results.append(result)
        return result
    
    def run_ultra_low_unet_lr_experiments(self, num_epochs: int = 30) -> List[Dict]:
        """
        Experiment 2b.v1: Test ultra-low U-Net learning rates.
        
        Hypothesis: The U-Net LR (1e-5) might be too high for a heavily pre-trained
        and now FiLM-conditioned U-Net, causing instability.
        
        Args:
            num_epochs: Number of epochs per experiment
            
        Returns:
            List of experiment results
        """
        print(f"\n🎯 ULTRA-LOW U-NET LR EXPERIMENTS")
        print(f"{'='*50}")
        print("Hypothesis: Standard U-Net LR (1e-5) too high for FiLM-conditioned pre-trained U-Net")
        
        ultra_low_lr_configs = [
            {
                'experiment_id': 'ultra_low_unet_lr_2e6',
                'lr_unet': 2e-6,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'Ultra-low U-Net LR: 2e-6'
            },
            {
                'experiment_id': 'ultra_low_unet_lr_1e6',
                'lr_unet': 1e-6,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'Ultra-low U-Net LR: 1e-6'
            },
            {
                'experiment_id': 'ultra_low_unet_lr_5e7',
                'lr_unet': 5e-7,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'Extremely low U-Net LR: 5e-7'
            }
        ]
        
        results = []
        for config in ultra_low_lr_configs:
            result = self.run_single_diagnostic_experiment(config, num_epochs)
            results.append(result)
        
        return results
    
    def run_film_regularization_experiments(self, num_epochs: int = 30) -> List[Dict]:
        """
        Experiment 2b.v2 & 2b.v3: Test different FiLM regularization strengths.
        
        Hypothesis: The optimal FiLM regularization strength might change when 
        the U-Net is also adapting during Phase 2b.
        
        Args:
            num_epochs: Number of epochs per experiment
            
        Returns:
            List of experiment results
        """
        print(f"\n🎯 FILM REGULARIZATION EXPERIMENTS")
        print(f"{'='*50}")
        print("Hypothesis: FiLM regularization λ should be tuned for Phase 2b dynamics")
        
        film_reg_configs = [
            {
                'experiment_id': 'weak_film_reg',
                'lambda_gamma_res': 0.001,
                'lambda_beta_res': 0.0001,
                'lr_unet': 1e-5,  # Standard U-Net LR
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'Weaker FiLM regularization'
            },
            {
                'experiment_id': 'strong_film_reg',
                'lambda_gamma_res': 0.01,
                'lambda_beta_res': 0.001,
                'lr_unet': 1e-5,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'Stronger FiLM regularization'
            },
            {
                'experiment_id': 'asymmetric_film_reg',
                'lambda_gamma_res': 0.01,   # Strong gamma regularization
                'lambda_beta_res': 0.0001,  # Weak beta regularization  
                'lr_unet': 1e-5,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'Asymmetric FiLM regularization (strong γ, weak β)'
            }
        ]
        
        results = []
        for config in film_reg_configs:
            result = self.run_single_diagnostic_experiment(config, num_epochs)
            results.append(result)
        
        return results
    
    def run_scheduler_optimization_experiments(self, num_epochs: int = 30) -> List[Dict]:
        """
        Experiment 2b.v4: Test different LR scheduler configurations.
        
        Hypothesis: The current scheduler (ReduceLROnPlateau with patience 5, factor 0.5) 
        might not be optimal for Phase 2b convergence.
        
        Args:
            num_epochs: Number of epochs per experiment
            
        Returns:
            List of experiment results
        """
        print(f"\n🎯 LR SCHEDULER OPTIMIZATION EXPERIMENTS")
        print(f"{'='*50}")
        print("Hypothesis: Current scheduler settings suboptimal for Phase 2b")
        
        scheduler_configs = [
            {
                'experiment_id': 'aggressive_scheduler',
                'scheduler_type': 'ReduceLROnPlateau',
                'scheduler_patience': 3,
                'scheduler_factor': 0.2,
                'lr_unet': 1e-5,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'More aggressive LR reduction (patience=3, factor=0.2)'
            },
            {
                'experiment_id': 'cosine_scheduler',
                'scheduler_type': 'CosineAnnealingLR',
                'lr_unet': 1e-5,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'Cosine annealing scheduler'
            },
            {
                'experiment_id': 'patient_scheduler',
                'scheduler_type': 'ReduceLROnPlateau',
                'scheduler_patience': 8,
                'scheduler_factor': 0.3,
                'lr_unet': 1e-5,
                'lr_frontend': 5e-5,
                'lr_film_generator': 1e-4,
                'film_generator_type': '2_layer',
                'description': 'More patient scheduler (patience=8, factor=0.3)'
            }
        ]
        
        results = []
        for config in scheduler_configs:
            result = self.run_single_diagnostic_experiment(config, num_epochs)
            results.append(result)
        
        return results
    
    def run_extended_phase2a_experiment(self, num_epochs: int = 20) -> Dict:
        """
        Extended Phase 2a: Test if longer frontend-only training improves performance.
        
        Hypothesis: Phase 2a (frontend frozen) might be the optimal approach,
        and Phase 2b adds unnecessary complexity.
        
        Args:
            num_epochs: Number of additional epochs for Phase 2a
            
        Returns:
            Experiment result
        """
        print(f"\n🎯 EXTENDED PHASE 2A EXPERIMENT")
        print(f"{'='*50}")
        print("Hypothesis: Phase 2a optimization limit not reached, Phase 2b unnecessary")
        
        extended_2a_config = {
            'experiment_id': 'extended_phase2a',
            'keep_unet_frozen': True,  # Key difference - keep U-Net frozen
            'lr_frontend': 1e-4,  # Slightly higher LR for extended training
            'lr_film_generator': 1e-4,
            'film_generator_type': '2_layer',
            'description': 'Extended Phase 2a with U-Net frozen'
        }
        
        return self.run_single_diagnostic_experiment(extended_2a_config, num_epochs)
    
    def run_complete_diagnostic_suite(self, 
                                    num_epochs_phase2b: int = 30,
                                    num_epochs_extended_2a: int = 20) -> Dict:
        """
        Run the complete diagnostic experiment suite.
        
        Args:
            num_epochs_phase2b: Epochs for Phase 2b experiments
            num_epochs_extended_2a: Epochs for extended Phase 2a
            
        Returns:
            Summary of all diagnostic results
        """
        print(f"\n{'='*80}")
        print(f"🚀 STARTING COMPLETE PHASE 2B DIAGNOSTIC SUITE")
        print(f"{'='*80}")
        print(f"Target: Break through 0.0893% MAPE barrier from Phase 2a")
        print(f"Base checkpoint: {self.base_checkpoint_path}")
        
        start_time = datetime.now()
        
        # Run all diagnostic experiment categories
        ultra_low_results = self.run_ultra_low_unet_lr_experiments(num_epochs_phase2b)
        film_reg_results = self.run_film_regularization_experiments(num_epochs_phase2b)
        scheduler_results = self.run_scheduler_optimization_experiments(num_epochs_phase2b)
        extended_2a_result = self.run_extended_phase2a_experiment(num_epochs_extended_2a)
        
        end_time = datetime.now()
        
        # Generate summary
        summary = {
            'diagnostic_suite_summary': {
                'total_experiments': len(self.diagnostic_results),
                'successful_experiments': len([r for r in self.diagnostic_results if r.get('success', False)]),
                'best_overall_mape': self.best_overall_mape,
                'best_experiment_config': self.best_experiment_config,
                'baseline_mape_target': 0.0893,  # Phase 2a target to beat
                'improvement_achieved': self.best_overall_mape < 0.0893,
                'improvement_amount': 0.0893 - self.best_overall_mape if self.best_overall_mape < 0.0893 else 0,
                'execution_time': str(end_time - start_time),
                'timestamp': end_time.isoformat()
            },
            'experiment_categories': {
                'ultra_low_unet_lr': {
                    'count': len(ultra_low_results),
                    'best_mape': min([r.get('best_mape', float('inf')) for r in ultra_low_results if r.get('success', False)], default=float('inf'))
                },
                'film_regularization': {
                    'count': len(film_reg_results),
                    'best_mape': min([r.get('best_mape', float('inf')) for r in film_reg_results if r.get('success', False)], default=float('inf'))
                },
                'lr_scheduler': {
                    'count': len(scheduler_results),
                    'best_mape': min([r.get('best_mape', float('inf')) for r in scheduler_results if r.get('success', False)], default=float('inf'))
                },
                'extended_phase2a': {
                    'count': 1,
                    'best_mape': extended_2a_result.get('best_mape', float('inf')) if extended_2a_result.get('success', False) else float('inf')
                }
            },
            'all_results': self.diagnostic_results
        }
        
        # Save results
        self.save_diagnostic_results(summary)
        
        # Print final summary
        self.print_diagnostic_summary(summary)
        
        return summary
    
    def save_diagnostic_results(self, summary: Dict):
        """Save diagnostic results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_path = os.path.join(self.results_dir, f"phase2b_diagnostic_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 Diagnostic results saved: {results_path}")
    
    def print_diagnostic_summary(self, summary: Dict):
        """Print comprehensive diagnostic summary."""
        print(f"\n{'='*80}")
        print(f"📊 PHASE 2B DIAGNOSTIC SUMMARY")
        print(f"{'='*80}")
        
        suite_summary = summary['diagnostic_suite_summary']
        
        print(f"🎯 BASELINE TARGET: 0.0893% MAPE (Phase 2a)")
        print(f"🏆 BEST ACHIEVED: {suite_summary['best_overall_mape']:.6f}% MAPE")
        
        if suite_summary['improvement_achieved']:
            print(f"✅ IMPROVEMENT: {suite_summary['improvement_amount']:.6f}% MAPE reduction")
            print(f"🎉 BREAKTHROUGH ACHIEVED!")
            print(f"🏆 Best Configuration: {suite_summary['best_experiment_config']}")
        else:
            deficit = suite_summary['best_overall_mape'] - 0.0893
            print(f"⚠️  TARGET NOT REACHED: {deficit:.6f}% above baseline")
            print(f"💡 RECOMMENDATION: Consider architectural changes or longer training")
        
        print(f"\n📈 EXPERIMENT BREAKDOWN:")
        for category, stats in summary['experiment_categories'].items():
            if stats['count'] > 0:
                best_mape = stats['best_mape']
                if best_mape != float('inf'):
                    print(f"  {category}: {stats['count']} experiments, best MAPE: {best_mape:.6f}%")
                else:
                    print(f"  {category}: {stats['count']} experiments, all failed")
        
        print(f"\n⏱️ Total execution time: {suite_summary['execution_time']}")
        print(f"📊 Successful experiments: {suite_summary['successful_experiments']}/{suite_summary['total_experiments']}")
        
        print(f"\n{'='*80}")


def run_phase2b_diagnostics(base_checkpoint_path: str,
                           experiment_name: str = "Phase2b_Diagnostic_Suite",
                           num_epochs_phase2b: int = 30,
                           num_epochs_extended_2a: int = 20) -> Dict:
    """
    Main function to run Phase 2b diagnostic experiments.
    
    This function implements the systematic diagnostic approach outlined in the
    comprehensive project analysis to address Phase 2b stagnation.
    
    Args:
        base_checkpoint_path: Path to best Phase 2a checkpoint
        experiment_name: Name for the diagnostic experiment suite
        num_epochs_phase2b: Epochs for Phase 2b diagnostic experiments
        num_epochs_extended_2a: Epochs for extended Phase 2a experiment
        
    Returns:
        Complete diagnostic results summary
    """
    print(f"🚀 Starting Phase 2b diagnostic experiments...")
    print(f"Base checkpoint: {base_checkpoint_path}")
    
    # Initialize diagnostic framework
    framework = Phase2bDiagnosticFramework(
        base_checkpoint_path=base_checkpoint_path,
        experiment_base_name=experiment_name
    )
    
    # Run complete diagnostic suite
    results = framework.run_complete_diagnostic_suite(
        num_epochs_phase2b=num_epochs_phase2b,
        num_epochs_extended_2a=num_epochs_extended_2a
    )
    
    return results


# =====================================
# CONVENIENCE FUNCTIONS FOR QUICK TESTING
# =====================================

def quick_phase2b_diagnostic_test():
    """Quick test with reduced epochs for validation."""
    print("🧪 Running quick Phase 2b diagnostic test...")
    
    # Use available checkpoint
    checkpoint_path = "checkpoints/Corrected_FiLM_Linear_PhaseA_FrontendFrozen_best_mape.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith('.pth'):
                    print(f"  {f}")
        return None
    
    results = run_phase2b_diagnostics(
        base_checkpoint_path=checkpoint_path,
        experiment_name="Quick_Phase2b_Test",
        num_epochs_phase2b=5,  # Reduced for testing
        num_epochs_extended_2a=3
    )
    
    return results


def focused_ultra_low_lr_test():
    """Focused test on ultra-low U-Net learning rates only."""
    print("🎯 Running focused ultra-low LR test...")
    
    checkpoint_path = "checkpoints/Corrected_FiLM_Linear_PhaseA_FrontendFrozen_best_mape.pth"
    
    framework = Phase2bDiagnosticFramework(
        base_checkpoint_path=checkpoint_path,
        experiment_base_name="Focused_UltraLow_LR"
    )
    
    # Run only ultra-low LR experiments
    results = framework.run_ultra_low_unet_lr_experiments(num_epochs=15)
    
    return results


if __name__ == "__main__":
    # Run quick diagnostic test if executed directly
    print("🚀 Running Phase 2b Diagnostic Framework")
    quick_phase2b_diagnostic_test()