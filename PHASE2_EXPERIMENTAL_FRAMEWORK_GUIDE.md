# Phase 2: Advanced Loss Function Experimental Framework
*Systematic experimentation for seismic velocity inversion performance improvement*

## 🚨 IMPORTANT: RESOLVING SoftAdapt COMPATIBILITY ISSUE

If you encounter the error:
```
TypeError: LossWeightedSoftAdapt.__init__() got an unexpected keyword argument 'normalize_slopes'
```

**SOLUTION - Restart Kernel and Re-run:**
1. **Restart your Jupyter kernel** (Kernel → Restart in Jupyter)
2. **Re-run all cells** to reload the corrected framework
3. The issue has been fixed in the framework files, but cached code needs to be cleared

The error was caused by an incorrect parameter in the SoftAdapt initialization which has been corrected in `phase2_experimental_framework.py`.

## 🚨 CUDA TENSOR COMPATIBILITY FIX

If you encounter the error:
```
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

This has been **automatically fixed** in the framework. The SoftAdapt library expects CPU tensors, and the framework now properly handles device placement. No action needed from your side.

## 🚨 SOFTADAPT FINITE DIFFERENCE FIX

If you encounter the error:
```
ValueError: Accuracy orders larger than 5 must be even. Please check the arguments passed to the function.
```

This has been **automatically fixed** in the framework by setting `accuracy_order=2` in the SoftAdapt initialization, which ensures stable finite difference calculations for slope estimation. No action needed from your side.

## 🚨 TENSORFLOW TENSOR COMPATIBILITY FIX

If you encounter the error:
```
AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'to'
```

This has been **automatically fixed** in the framework. The SoftAdapt library sometimes returns TensorFlow tensors instead of PyTorch tensors. The framework now automatically detects and converts these to PyTorch tensors before using PyTorch-specific methods like `.to()`. No action needed from your side.

---

## 🎯 **Objective**
Systematically test and validate advanced loss functions to improve seismic velocity inversion performance beyond the baseline ~3.93% MAPE.

## 📋 **Framework Overview**

This experimental framework implements a rigorous scientific methodology for testing:

1. **AdaptiveLogSpaceMAE**: MAPE-proxy loss with adaptive parameter tuning
2. **SeismicMSSSIM**: Geological structure-aware similarity loss
3. **AnisotropicTotalVariationLoss**: Layer-specific smoothness regularization  
4. **LogSpaceMAEHybridLoss**: Multi-component loss with adaptive weighting

## 🚀 **Quick Start**

### Step 1: Setup in Your Notebook

```python
# Add this cell to your notebook after your existing model definitions:

# 1. Install required packages
!pip install pytorch-msssim softadapt scikit-image

# 2. Load the experimental framework
exec(open('phase2_experimental_framework.py').read())
exec(open('phase2_integration_notebook_cell.py').read())
```

### Step 2: Quick Validation Test

```python
# Run a quick 2-epoch test to verify everything works
results_test = quick_test_phase2_setup()
```

### Step 3: Full Experimental Suite

```python
# Run complete experiments (30 epochs each)
results_full = run_phase2_experiments_integrated(num_epochs=30)
```

## 🔬 **Experimental Design**

### **Controlled Variables**
- ✅ Same BaselineUNet architecture across all experiments
- ✅ Identical optimizer settings (AdamW, lr=1e-4, weight_decay=0.01)
- ✅ Consistent data splits (80/20 train/val, random_state=42)
- ✅ Same training epochs and batch size
- ✅ Fair comparison metrics (validation MAPE as primary)

### **Experimental Sequence**

| Experiment | Loss Function | Purpose |
|------------|---------------|---------|
| **Exp1A** | AdaptiveLogSpaceMAE | Test adaptive vs fixed parameter strategies |
| **Exp1B** | FixedCLogSpaceMAE | Baseline comparison for adaptive approach |
| **Exp3B** | HybridFixed | Multi-component loss with fixed weights |
| **Exp4** | HybridAdaptive | Full adaptive system with SoftAdapt |

### **Success Metrics**
- **Primary**: Validation MAPE (% improvement over baseline)
- **Secondary**: Validation MAE, training stability, geological realism
- **Tertiary**: Loss component analysis, adaptive weight behavior

## 📊 **Expected Performance Improvements**

Based on research synthesis, anticipated improvements over baseline:

| Component | Expected MAPE Improvement |
|-----------|---------------------------|
| AdaptiveLogSpaceMAE | 5-15% improvement |
| + SeismicMSSSIM | Additional 8-12% |
| + AnisotropicTV | Additional 3-7% |
| Full Hybrid Adaptive | 20-35% total improvement |

**Target**: Sub-4% MAPE (from baseline ~3.93%)

## 🛠 **Advanced Usage**

### **Individual Loss Function Testing**

```python
# Test individual components
from phase2_experimental_framework import AdaptiveLogSpaceMAE, SeismicMSSSIM

# Setup data loaders
train_loader, val_loader = setup_phase2_data_loaders()

# Test specific loss function
model = BaselineUNet(5, 1).to(device)
criterion = AdaptiveLogSpaceMAE(min_velocity=1.5, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

best_mape, history = train_validate_model(
    "Custom_AdaptiveLogMAE_Test", model, train_loader, val_loader,
    criterion, optimizer, 20, device, calculate_mape
)
```

### **Custom Hybrid Loss Configuration**

```python
# Create custom hybrid loss
custom_hybrid = LogSpaceMAEHybridLoss(
    min_velocity=1.5,
    use_adaptive_softadapt=True,
    fixed_weights_list=[1.0, 0.5, 0.01],  # Custom weights
    softadapt_beta=0.15,  # Custom SoftAdapt parameters
    atv_weight_h=1.2,     # Custom anisotropic weights
    atv_weight_v=0.25
)
```

### **Hyperparameter Sensitivity Analysis**

```python
# Test different c values for LogSpaceMAE
c_values = [0.01, 0.1, 0.5, 1.0]
results = {}

for c_val in c_values:
    criterion = FixedCLogSpaceMAE(fixed_c=c_val, min_velocity=1.5)
    best_mape, _ = train_validate_model(
        f"LogMAE_c{c_val}", model, train_loader, val_loader,
        criterion, optimizer, 15, device, calculate_mape
    )
    results[f"c={c_val}"] = best_mape
```

## 📈 **Results Analysis**

### **Automated Analysis**

The framework automatically provides:
- Best MAPE comparison across experiments
- Training history plots (loss, MAE, MAPE)
- Loss component analysis for hybrid losses
- Adaptive weight evolution tracking

### **Manual Analysis Commands**

```python
# Plot specific experiment results
plot_history(history_exp1a, "AdaptiveLogMAE_Analysis", save_path="results/exp1a.png")

# Compare multiple experiments
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(history_exp1a['val_mape'], label='Adaptive LogMAE', linewidth=2)
ax.plot(history_exp1b['val_mape'], label='Fixed LogMAE', linewidth=2)
ax.plot(history_hybrid['val_mape'], label='Hybrid Adaptive', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation MAPE (%)')
ax.set_title('Phase 2 Loss Function Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### **Model Checkpoint Management**

```python
# Load best performing model
best_model = BaselineUNet(5, 1)
best_model.load_state_dict(torch.load('checkpoints/Exp_HybridAdaptiveWeights_best_mape.pth'))
best_model.to(device)

# Test on validation set
best_model.eval()
with torch.no_grad():
    val_predictions = []
    val_targets = []
    for inputs, targets in val_loader:
        outputs = best_model(inputs.to(device))
        val_predictions.append(outputs.cpu().numpy())
        val_targets.append(targets.cpu().numpy())
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Memory Errors**
   ```python
   # Reduce batch size
   train_loader, val_loader = setup_phase2_data_loaders(batch_size=2)
   ```

2. **Import Errors**
   ```bash
   # Install missing packages
   pip install pytorch-msssim softadapt scikit-image
   ```

3. **Missing Components**
   ```python
   # Verify required components exist
   required = ['all_sample_folder_paths', 'BaselineUNet', 'SeismicDataset', 'calculate_mape', 'device']
   for component in required:
       if component not in globals():
           print(f"Missing: {component}")
   ```

### **Performance Optimization**

```python
# Enable mixed precision training (if GPU supports it)
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📝 **Experimental Log Template**

Document your experiments:

```
Experiment: [Name]
Date: [YYYY-MM-DD]
Objective: [Specific goal]
Configuration:
  - Loss Function: [Type and parameters]
  - Epochs: [Number]
  - Learning Rate: [Value]
  - Batch Size: [Value]
Results:
  - Best Validation MAPE: [Value]%
  - Improvement over baseline: [Value]%
  - Training stability: [Stable/Unstable/Notes]
Observations:
  - [Key findings]
  - [Unexpected behaviors]
  - [Recommendations for next experiments]
```

## 🎓 **Research Contributions**

This framework enables investigation of:

### **Novel Technical Contributions**
- Adaptive log-space MAPE proxy optimization
- Multi-scale geological structure preservation
- Domain-specific anisotropic regularization
- Multi-objective loss balancing with SoftAdapt

### **Methodological Contributions**
- Systematic ablation study framework
- Controlled experimental design for seismic inversion
- Reproducible research methodology
- Performance benchmarking standards

## 📚 **References & Further Reading**

### **Key Papers**
- MS-SSIM: Wang et al. "Image Quality Assessment: From Error Visibility to Structural Similarity"
- SoftAdapt: Heydari et al. "SoftAdapt: Techniques for Adaptive Loss Weighting"
- Anisotropic TV: Beck & Teboulle "Fast Gradient-Based Algorithms for Constrained Total Variation"

### **Implementation Details**
- See `phase2_experimental_framework.py` for complete technical implementation
- Loss function mathematical formulations in code documentation
- Numerical stability considerations and safeguards

---

## 🚀 **Ready to Start?**

1. Copy the framework files to your notebook directory
2. Run the quick setup test: `results_test = quick_test_phase2_setup()`
3. If successful, run full experiments: `results_full = run_phase2_experiments_integrated(num_epochs=30)`
4. Analyze results and iterate based on findings

**Expected Time**: 2-4 hours for full experimental suite (depending on hardware)

**Success Criteria**: Achieve validation MAPE < 3.5% (10%+ improvement over baseline)

Good luck with your experiments! 🎯 