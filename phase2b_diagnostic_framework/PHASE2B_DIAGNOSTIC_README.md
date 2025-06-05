# Phase 2b Diagnostic Experiments Framework

## Overview

This framework implements systematic diagnostic experiments to address the **Phase 2b stagnation issue** where full fine-tuning (unfreezing U-Net) consistently underperforms Phase 2a (frontend frozen) training in the FiLM-enhanced SincGAT-UNet architecture.

**Current Challenge**: Phase 2a achieves 0.0893% MAPE (2-layer FiLM), but Phase 2b fails to improve beyond this baseline across multiple experiments.

**Goal**: Break through the 0.0893% MAPE barrier through systematic hyperparameter optimization and training strategy refinement.

## Architecture Context

The project implements a seismic velocity inversion model with the following architecture:

- **SincNet Seismic Encoder**: Processes 5 input shot records with learnable band-pass filters
- **GAT Fusion**: Combines shot embeddings into a single context vector  
- **FiLM Integration**: Feature-wise Linear Modulation at U-Net bottleneck using GAT context
- **U-Net Backbone**: Pre-trained asymmetric U-Net for velocity map generation

**Two-Stage Training**:
1. **Stage 1**: Pre-train U-Net directly (achieved 0.0952% MAPE)
2. **Stage 2**: Fine-tune with FiLM integration
   - **Phase 2a**: Frontend training with U-Net frozen (achieved 0.0893% MAPE)
   - **Phase 2b**: Full fine-tuning with U-Net unfrozen (**STAGNATION ISSUE**)

## Diagnostic Hypotheses

The framework tests four key hypotheses for Phase 2b underperformance:

### 1. Ultra-Low U-Net Learning Rates
**Hypothesis**: Standard U-Net LR (1e-5) too high for heavily pre-trained, FiLM-conditioned model
- **Experiments**: Test 2e-6, 1e-6, 5e-7 learning rates
- **Rationale**: FiLM conditioning changes U-Net optimization landscape

### 2. FiLM Regularization Tuning  
**Hypothesis**: Optimal FiLM regularization (λ_gamma_res, λ_beta_res) changes during Phase 2b
- **Experiments**: Test weaker (0.001, 0.0001), stronger (0.01, 0.001), asymmetric regularization
- **Rationale**: U-Net adaptation affects FiLM parameter dynamics

### 3. LR Scheduler Optimization
**Hypothesis**: Current scheduler (ReduceLROnPlateau, patience=5, factor=0.5) suboptimal
- **Experiments**: Test aggressive (patience=3, factor=0.2), cosine annealing, patient scheduling
- **Rationale**: Phase 2b requires different convergence strategy

### 4. Extended Phase 2a Training
**Hypothesis**: Phase 2a optimization limit not reached, Phase 2b adds unnecessary complexity
- **Experiments**: Extended frontend-only training with U-Net frozen
- **Rationale**: Simplicity may be superior to full fine-tuning

## Framework Components

### Core Files

- **`phase2b_diagnostic_experiments.py`**: Main diagnostic framework implementation
- **`run_diagnostic_experiments.py`**: Command-line execution script  
- **`PHASE2B_DIAGNOSTIC_README.md`**: This documentation

### Key Classes

- **`Phase2bDiagnosticFramework`**: Central experiment orchestration
- **`CompleteSincGAT_UNet`**: Integrated model architecture (from existing codebase)
- **`RefinedLogSpaceMAEHybridLoss`**: FiLM-aware loss function (from existing codebase)

## Quick Start

### Prerequisites

Ensure you have the existing project components:
- `complete_sincgat_unet_integration.py`
- `phase2_experimental_framework.py` 
- A Phase 2a checkpoint in `checkpoints/` directory

### Basic Usage

```bash
# Quick validation test (5 epochs)
python run_diagnostic_experiments.py --mode quick

# Focused ultra-low LR experiments (15 epochs)
python run_diagnostic_experiments.py --mode focused_lr --epochs 15

# Complete diagnostic suite (30 epochs, ~10 experiments)
python run_diagnostic_experiments.py --mode complete --epochs 30

# Custom single experiment
python run_diagnostic_experiments.py --mode custom --lr_unet 1e-6 --epochs 20
```

### Programmatic Usage

```python
from phase2b_diagnostic_experiments import run_phase2b_diagnostics

# Run complete diagnostic suite
results = run_phase2b_diagnostics(
    base_checkpoint_path="checkpoints/Corrected_FiLM_Linear_PhaseA_FrontendFrozen_best_mape.pth",
    experiment_name="My_Diagnostic_Suite",
    num_epochs_phase2b=30,
    num_epochs_extended_2a=20
)

# Check if breakthrough achieved
if results['diagnostic_suite_summary']['improvement_achieved']:
    print("🎉 Breakthrough! Phase 2b optimization successful!")
    print(f"Best MAPE: {results['diagnostic_suite_summary']['best_overall_mape']:.6f}%")
```

## Experiment Categories

### 1. Ultra-Low U-Net LR Experiments

```python
framework = Phase2bDiagnosticFramework(checkpoint_path)
results = framework.run_ultra_low_unet_lr_experiments(num_epochs=30)
```

**Configurations Tested**:
- `lr_unet=2e-6`: Ultra-low learning rate
- `lr_unet=1e-6`: Very low learning rate  
- `lr_unet=5e-7`: Extremely low learning rate

### 2. FiLM Regularization Experiments

```python
results = framework.run_film_regularization_experiments(num_epochs=30)
```

**Configurations Tested**:
- **Weak**: λ_gamma_res=0.001, λ_beta_res=0.0001
- **Strong**: λ_gamma_res=0.01, λ_beta_res=0.001
- **Asymmetric**: λ_gamma_res=0.01, λ_beta_res=0.0001

### 3. LR Scheduler Experiments

```python
results = framework.run_scheduler_optimization_experiments(num_epochs=30)
```

**Configurations Tested**:
- **Aggressive**: ReduceLROnPlateau(patience=3, factor=0.2)
- **Cosine**: CosineAnnealingLR(T_max=30)
- **Patient**: ReduceLROnPlateau(patience=8, factor=0.3)

### 4. Extended Phase 2a Experiment

```python
result = framework.run_extended_phase2a_experiment(num_epochs=20)
```

**Configuration**: Frontend training only, U-Net frozen, lr=1e-4

## Expected Outputs

### Results Structure

```json
{
  "diagnostic_suite_summary": {
    "total_experiments": 10,
    "successful_experiments": 9,
    "best_overall_mape": 0.0875,
    "best_experiment_config": {...},
    "baseline_mape_target": 0.0893,
    "improvement_achieved": true,
    "improvement_amount": 0.0018,
    "execution_time": "3:45:23",
    "timestamp": "2024-01-15T10:30:00"
  },
  "experiment_categories": {
    "ultra_low_unet_lr": {"count": 3, "best_mape": 0.0881},
    "film_regularization": {"count": 3, "best_mape": 0.0887},
    "lr_scheduler": {"count": 3, "best_mape": 0.0875},
    "extended_phase2a": {"count": 1, "best_mape": 0.0889}
  },
  "all_results": [...]
}
```

### Checkpoint Locations

Successful experiments save checkpoints to:
- `checkpoints/{experiment_name}_best.pth`

Results are saved to:
- `experiment_results/phase2b_diagnostic_results_{timestamp}.json`

## Interpretation Guide

### Success Criteria

✅ **Breakthrough Achieved**: Any experiment with MAPE < 0.0893%
- Indicates Phase 2b optimization can exceed Phase 2a performance
- Provides validated hyperparameter configuration for production use

⚠️ **Marginal Improvement**: MAPE between 0.0890% - 0.0893%
- Suggests potential with further refinement
- Consider longer training or architectural modifications

❌ **No Improvement**: All experiments > 0.0893%
- May indicate fundamental limitations of current approach
- Consider multi-scale FiLM or alternative architectural changes

### Next Steps Based on Results

**If Ultra-Low LR Succeeds**:
- Adopt winning U-Net learning rate for production
- Consider even lower rates or more sophisticated LR scheduling

**If FiLM Regularization Succeeds**:
- Fine-tune regularization strengths further
- Investigate dynamic regularization during training

**If Scheduler Optimization Succeeds**:
- Adopt winning scheduler configuration
- Consider hybrid or custom scheduling strategies

**If Extended Phase 2a Succeeds**:
- Skip Phase 2b entirely in production
- Focus optimization efforts on Phase 2a improvements

**If No Success**:
- Proceed to multi-scale FiLM implementation (conditioning skip connections)
- Consider alternative GAT context injection points
- Investigate architectural modifications beyond FiLM

## Advanced Customization

### Custom Experiment Configuration

```python
custom_config = {
    'experiment_id': 'custom_experiment',
    'lr_unet': 3e-6,
    'lr_frontend': 4e-5,  
    'lr_film_generator': 8e-5,
    'lambda_gamma_res': 0.002,
    'lambda_beta_res': 0.0002,
    'scheduler_type': 'ReduceLROnPlateau',
    'scheduler_patience': 4,
    'scheduler_factor': 0.3,
    'film_generator_type': '2_layer',
    'weight_decay': 0.01,
    'weight_decay_film': 5e-4,
    'batch_size': 4
}

framework = Phase2bDiagnosticFramework(checkpoint_path)
result = framework.run_single_diagnostic_experiment(custom_config, num_epochs=25)
```

### Adding New Experiment Categories

```python
def run_custom_experiment_category(self, num_epochs=30):
    """Add new diagnostic experiment category."""
    configs = [
        # Define your experimental configurations here
    ]
    
    results = []
    for config in configs:
        result = self.run_single_diagnostic_experiment(config, num_epochs)
        results.append(result)
    
    return results
```

## Performance Considerations

### Computational Requirements

- **GPU**: CUDA-capable GPU recommended (tested on A100)
- **Memory**: ~16GB GPU memory for batch_size=4
- **Time**: ~30 minutes per experiment (30 epochs)
- **Complete Suite**: 5-8 hours for all 10 experiments

### Optimization Tips

- Use `--mode quick` for initial validation (5 epochs)
- Start with `--mode focused_lr` for targeted testing
- Monitor GPU utilization and adjust batch_size if needed
- Save intermediate results in case of interruption

## Troubleshooting

### Common Issues

**Checkpoint Not Found**:
```
❌ Checkpoint not found: checkpoints/...
```
- Ensure Phase 2a checkpoint exists in `checkpoints/` directory
- Use `--checkpoint_path` to specify custom location

**CUDA Out of Memory**:
```
RuntimeError: CUDA out of memory
```
- Reduce batch_size in experiment configs
- Use gradient accumulation instead of larger batches

**Import Errors**:
```
ModuleNotFoundError: No module named '...'
```
- Ensure all existing project files are in Python path
- Install required dependencies (pytorch, torch-geometric, etc.)

**Slow Training**:
- Enable A100 TF32 optimization (automatically configured)
- Verify CUDA/cuDNN versions compatible with PyTorch
- Monitor GPU utilization with `nvidia-smi`

### Debug Mode

Enable detailed logging by modifying the training loop:

```python
# In simple_training_loop method
if batch_idx % 5 == 0:  # More frequent logging
    print(f"Detailed: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
```

## Integration with Existing Codebase

### Dependencies

The diagnostic framework integrates with existing project components:

- **`complete_sincgat_unet_integration.py`**: Model architecture and A100 stability
- **`phase2_experimental_framework.py`**: Loss functions and training utilities
- **Main notebook functions**: Data loading and MAPE calculation

### Minimal Integration

If running in isolated environment, ensure these functions are available:
- `setup_phase2_data_loaders()`: Data loader setup
- `calculate_mape()`: MAPE metric calculation  
- `train_with_film_awareness()`: Advanced training loop (optional)

## Results Reporting

### Academic/Research Context

The diagnostic framework provides systematic evidence for Phase 2b optimization strategies:

- **Hypothesis Testing**: Each experiment category tests specific hypotheses
- **Statistical Validation**: Multiple runs provide confidence intervals
- **Reproducibility**: All configurations and random seeds recorded
- **Comparative Analysis**: Direct comparison against Phase 2a baseline

### Production Context

Successful experiments provide:

- **Optimized Hyperparameters**: Validated learning rates and regularization
- **Training Protocols**: Proven scheduling and optimization strategies  
- **Performance Metrics**: Quantified improvement over baseline
- **Checkpoint Models**: Ready-to-deploy model weights

## Contributing

To extend the diagnostic framework:

1. **Add New Hypotheses**: Implement additional experiment categories
2. **Refine Configurations**: Test broader hyperparameter ranges
3. **Improve Training**: Enhance the simplified training loop
4. **Add Metrics**: Include additional evaluation criteria beyond MAPE

## Citation

If using this diagnostic framework in research, please cite the original project and note the systematic Phase 2b optimization approach.

---

**Framework Version**: 1.0  
**Target MAPE**: < 0.0893% (Phase 2a baseline)  
**Status**: Ready for systematic Phase 2b optimization  
**Last Updated**: 2024 (Implementation date) 