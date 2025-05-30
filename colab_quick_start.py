# ==============================================================================
# QUICK START: SincNet+GAT in Google Colab
# ==============================================================================
# Copy each section below into separate Colab cells and run in order

# CELL 1: Setup
# ------------------------------------------------------------------------------
!pip install torch_geometric --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("GPU optimization configured")

# CELL 2: Import Architecture
# ------------------------------------------------------------------------------
# Make sure you uploaded these files first:
# - sincnet_seismic_encoder.py
# - seismic_gat_fusion.py  
# - sincnet_integration_demo.py

from sincnet_seismic_encoder import PerShotTemporalEncoder
from seismic_gat_fusion import SeismicSincNetGAT, SpatiallyAwareLightweightGATFusion
from sincnet_integration_demo import (
    SpatiallyAwareSincNetGATModel, 
    EnhancedSpatialSincNetGATModel
)
print("Architecture imported successfully!")

# CELL 3: Create Models
# ------------------------------------------------------------------------------
config = {
    'num_receivers': 31,
    'sinc_out_channels': 40, 
    'shot_embedding_dim': 128,
    'gat_embedding_dim': 128,
    'target_height': 300,
    'target_width': 1259,
    'num_shots': 5
}

# Path Alpha - Start with this one
model_alpha = SpatiallyAwareSincNetGATModel(**config).to(device)
params_alpha = sum(p.numel() for p in model_alpha.parameters())

# Path Beta - More advanced
model_beta = EnhancedSpatialSincNetGATModel(**config).to(device)
params_beta = sum(p.numel() for p in model_beta.parameters())

print(f"Path Alpha: {params_alpha:,} parameters")
print(f"Path Beta:  {params_beta:,} parameters") 
print(f"Champion:   17,260,000 parameters")
print(f"Efficiency: {((17260000 - params_alpha) / 17260000 * 100):.1f}% reduction!")

# CELL 4: Test Models
# ------------------------------------------------------------------------------
batch_size = 2
dummy_shots = torch.randn(batch_size, 5, 10001, 31).to(device)

with torch.no_grad():
    output_alpha = model_alpha(dummy_shots)
    output_beta = model_beta(dummy_shots)
    
    print(f"Input: {dummy_shots.shape}")
    print(f"Path Alpha output: {output_alpha.shape}")
    print(f"Path Beta output:  {output_beta.shape}")
    
    expected = (batch_size, 1, 300, 1259)
    if output_alpha.shape == expected and output_beta.shape == expected:
        print("SUCCESS: Both models compatible with BaselineUNet!")
    else:
        print("ERROR: Shape mismatch")

# CELL 5: Choose Model and Setup Training
# ------------------------------------------------------------------------------
# Recommend starting with Path Alpha
selected_model = model_alpha  # or model_beta for advanced features
model_name = "Path Alpha"

print(f"Selected: {model_name}")
print(f"Parameters: {sum(p.numel() for p in selected_model.parameters()):,}")

# Setup optimizer (same as your existing setup)
optimizer = optim.AdamW(selected_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("Training setup complete!")

# CELL 6: Integration Example  
# ------------------------------------------------------------------------------
print("REPLACE YOUR BASELINE MODEL WITH:")
print("""
# OLD:
# model = BaselineUNet(input_channels=5, output_channels=1).to(device)

# NEW:
from sincnet_integration_demo import SpatiallyAwareSincNetGATModel
model = SpatiallyAwareSincNetGATModel(
    num_receivers=31,
    sinc_out_channels=40,
    shot_embedding_dim=128,
    gat_embedding_dim=128,
    target_height=300,
    target_width=1259
).to(device)

# Everything else stays exactly the same!
# Same loss function, same training loop, same evaluation
""")

# CELL 7: Quick Training Test (Optional)
# ------------------------------------------------------------------------------
def quick_test(model, steps=3):
    model.train()
    criterion = nn.MSELoss()
    
    for step in range(steps):
        # Dummy batch
        inputs = torch.randn(2, 5, 10001, 31).to(device)
        targets = torch.randn(2, 1, 300, 1259).to(device)
        
        # Forward/backward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: Loss = {loss.item():.6f}")
    
    print("Training test successful!")

# Uncomment to run:
# quick_test(selected_model)

# CELL 8: Save/Load Models
# ------------------------------------------------------------------------------
def save_model(model, name="sincnet_gat.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'parameters': sum(p.numel() for p in model.parameters())
    }, name)
    print(f"Model saved: {name}")

def load_model(name, model_class=SpatiallyAwareSincNetGATModel):
    checkpoint = torch.load(name, map_location=device)
    model = model_class(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded: {name} ({checkpoint['parameters']:,} params)")
    return model

# Usage:
# save_model(selected_model, "my_best_model.pth")
# loaded_model = load_model("my_best_model.pth")

print("\n" + "="*60)
print("COLAB SETUP COMPLETE!")
print("="*60)
print("Your SincNet+GAT model is ready to replace BaselineUNet!")
print(f"Target: Beat 0.0862% MAPE with {params_alpha:,} parameters")
print("63% fewer parameters than champion!")
print("="*60) 