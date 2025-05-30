# 🚀 COLAB QUICK REFERENCE - SincNet+GAT

**Ultra-fast setup guide for Google Colab**

## 📋 PREPARATION (30 seconds)

1. **Set GPU Runtime**: Runtime → Change runtime type → GPU (T4/A100)
2. **Upload Files**: Use folder icon to upload:
   - `sincnet_seismic_encoder.py`
   - `seismic_gat_fusion.py`  
   - `sincnet_integration_demo.py`

## ⚡ SETUP CELLS (Copy-paste into Colab)

### Cell 1: Install & Setup
```python
!pip install torch_geometric --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
```

### Cell 2: Import Architecture
```python
from sincnet_seismic_encoder import PerShotTemporalEncoder
from seismic_gat_fusion import SeismicSincNetGAT
from sincnet_integration_demo import (
    SpatiallyAwareSincNetGATModel, 
    EnhancedSpatialSincNetGATModel
)
print("Architecture imported!")
```

### Cell 3: Create & Test Models
```python
config = {
    'num_receivers': 31, 'sinc_out_channels': 40, 
    'shot_embedding_dim': 128, 'gat_embedding_dim': 128,
    'target_height': 300, 'target_width': 1259, 'num_shots': 5
}

# Path Alpha - Start here (6.35M params, 63% reduction)
model_alpha = SpatiallyAwareSincNetGATModel(**config).to(device)

# Path Beta - Advanced (4.24M params, 75% reduction)  
model_beta = EnhancedSpatialSincNetGATModel(**config).to(device)

print(f"Alpha: {sum(p.numel() for p in model_alpha.parameters()):,} params")
print(f"Beta:  {sum(p.numel() for p in model_beta.parameters()):,} params")

# Test compatibility
dummy = torch.randn(2, 5, 10001, 31).to(device)
with torch.no_grad():
    out_alpha = model_alpha(dummy)
    out_beta = model_beta(dummy)
    print(f"Alpha output: {out_alpha.shape}")  # (2, 1, 300, 1259)
    print(f"Beta output:  {out_beta.shape}")   # (2, 1, 300, 1259)
```

### Cell 4: Setup Training
```python
# Choose your model
selected_model = model_alpha  # Start with Alpha

# Setup optimizer  
optimizer = optim.AdamW(selected_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

print(f"Training ready: {sum(p.numel() for p in selected_model.parameters()):,} parameters")
```

## 🔧 INTEGRATION (Replace in your training code)

**OLD:**
```python
model = BaselineUNet(input_channels=5, output_channels=1).to(device)
```

**NEW:**
```python
from sincnet_integration_demo import SpatiallyAwareSincNetGATModel
model = SpatiallyAwareSincNetGATModel(
    num_receivers=31, sinc_out_channels=40,
    shot_embedding_dim=128, gat_embedding_dim=128,
    target_height=300, target_width=1259
).to(device)
```

**Everything else stays EXACTLY the same!**
- Same input format: `(B, 5, 10001, 31)`
- Same output format: `(B, 1, 300, 1259)`
- Same loss function: `RefinedLogSpaceMAEHybridLoss` with `[1.0, 0.12, 0.007]`
- Same training loop
- Same evaluation metrics

## 🎯 PERFORMANCE TARGETS

| Model | Parameters | Efficiency | Use Case |
|-------|------------|------------|----------|
| **Path Alpha** | 6.35M | 63% reduction | Initial validation |
| **Path Beta** | 4.24M | 75% reduction | Enhanced spatial detail |
| Champion | 17.26M | Baseline | 0.0862% MAPE |

## ⚡ QUICK TEST

```python
# Quick training test
def quick_test(model, steps=3):
    model.train()
    criterion = nn.MSELoss()
    for step in range(steps):
        inputs = torch.randn(2, 5, 10001, 31).to(device)
        targets = torch.randn(2, 1, 300, 1259).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}: Loss = {loss.item():.6f}")

# Run test
quick_test(selected_model)
```

## 💾 SAVE/LOAD

```python
# Save
torch.save({
    'model_state_dict': selected_model.state_dict(),
    'config': config
}, "sincnet_gat_best.pth")

# Load
checkpoint = torch.load("sincnet_gat_best.pth")
model = SpatiallyAwareSincNetGATModel(**checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🏆 SUCCESS METRICS

- ✅ **Primary**: Beat 0.0862% MAPE
- ✅ **Efficiency**: 63-75% parameter reduction
- ✅ **Quality**: Preserve geological features ("yellow anomaly")
- ✅ **Speed**: Faster training than champion

## 🚀 YOU'RE READY!

Your SincNet+GAT architecture is now a **drop-in replacement** for BaselineUNet with:
- **Enhanced temporal-frequency processing** (SincNet)
- **Cross-shot attention mechanisms** (GAT)  
- **Massive parameter efficiency** (63-75% reduction)
- **Same training infrastructure** (no code changes)

**GO BEAT THAT 0.0862% MAPE!** 🏆 