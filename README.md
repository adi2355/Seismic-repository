<div align="center">
  <img src="./terminal-top-panel.svg" alt="Terminal Top Panel" width="100%" />
</div>

<br>

## Overview

A deep learning system for seismic velocity inversion — the geophysical inverse problem of predicting subsurface velocity structure from surface-recorded seismic waveforms. The architecture is a three-stage pipeline: a **SincNet temporal encoder** extracts physics-informed frequency features from raw shot gathers, a **Graph Attention Network** fuses multi-shot spatial relationships through learned attention, and a **U-Net decoder** with **FiLM conditioning** generates high-resolution 2D velocity maps.

The model processes 5 shot gathers per sample (10,001 time steps across 31 receivers each) and outputs a full 300 x 1,259 subsurface velocity field. The champion configuration achieves **~0.0655% MAPE** on the held-out evaluation set.

<br>

## Technology Stack

<table>
<tr>
<td><strong>Core Framework</strong></td>
<td><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" /> <img src="https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" /></td>
</tr>
<tr>
<td><strong>Graph Neural Networks</strong></td>
<td><img src="https://img.shields.io/badge/PyTorch_Geometric-3C2179?style=flat-square&logo=pyg&logoColor=white" alt="PyTorch Geometric" /> <img src="https://img.shields.io/badge/GATv2Conv-3C2179?style=flat-square&logoColor=white" alt="GATv2Conv" /></td>
</tr>
<tr>
<td><strong>Signal Processing</strong></td>
<td><img src="https://img.shields.io/badge/SincNet-FF6F00?style=flat-square&logoColor=white" alt="SincNet" /> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" /> <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy" /></td>
</tr>
<tr>
<td><strong>Loss & Optimization</strong></td>
<td><img src="https://img.shields.io/badge/MS--SSIM-4B8BBE?style=flat-square&logoColor=white" alt="MS-SSIM" /> <img src="https://img.shields.io/badge/SoftAdapt-4B8BBE?style=flat-square&logoColor=white" alt="SoftAdapt" /> <img src="https://img.shields.io/badge/SAM_Optimizer-4B8BBE?style=flat-square&logoColor=white" alt="SAM" /></td>
</tr>
<tr>
<td><strong>Compute</strong></td>
<td><img src="https://img.shields.io/badge/NVIDIA_A100-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="A100" /> <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Colab" /> <img src="https://img.shields.io/badge/Mixed_Precision-76B900?style=flat-square&logoColor=white" alt="AMP" /></td>
</tr>
<tr>
<td><strong>Visualization</strong></td>
<td><img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white" alt="Matplotlib" /> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" /></td>
</tr>
</table>

<br>

## Engineering Principles

### 1. Encode domain physics directly into learnable parameters

The SincNet encoder replaces generic 1D convolutions with parametric sinc-function bandpass filters. Each filter learns its own center frequency and bandwidth — the network discovers which seismic frequency bands carry velocity information rather than learning arbitrary kernel weights. Blackman windowing suppresses spectral leakage. Logarithmic filter initialization allocates finer resolution to the lower frequencies where seismic signals concentrate energy.

> **Goal:** Impose physically meaningful spectral constraints while preserving full gradient-based adaptability.

### 2. Model spatial relationships through structure, not concatenation

Five shot gathers recorded at different source positions encode complementary views of the same subsurface. Rather than concatenating these views and hoping a CNN disentangles them, the Graph Attention Network treats each shot as a node in a fully-connected graph. GATv2 attention heads learn which shot pairs carry complementary information for each spatial region. The graph pooling stage produces a single fused embedding that captures multi-offset spatial dependencies.

> **Goal:** Let the network learn inter-shot importance explicitly through attention, not implicitly through convolution.

### 3. Condition the decoder without corrupting it

The U-Net decoder is a proven architecture for dense spatial prediction. FiLM (Feature-wise Linear Modulation) injects the GAT-fused context at the bottleneck via learned scale and shift parameters, with a residual formulation: `output = target + (gamma * target + beta)`. The gamma and beta projections are zero-initialized, so the FiLM layer acts as an identity function at training start. The U-Net begins training as if unconditional, and gradually integrates multi-shot context as the FiLM parameters diverge from zero.

> **Goal:** Integrate global context without destabilizing the spatial decoder's convergence.

### 4. Compose losses that capture complementary error modalities

A single loss function cannot simultaneously optimize pixel-level accuracy, structural fidelity, and geological smoothness. The training pipeline combines log-space MAE (scale-invariant accuracy), multi-scale structural similarity (spatial coherence), and anisotropic total variation (geological layering bias) through SoftAdapt-weighted combination with curriculum warmup. The loss composition adapts its own weighting schedule based on relative improvement rates.

> **Goal:** Each loss component corrects failure modes the others miss, with adaptive balancing that prevents any single term from dominating.

---

## Model Architecture

### SincNet Temporal Encoder

Each of the 5 shot gathers is processed independently through a domain-adapted SincNet layer. The `SincConv1d_SeismicAdapted` module implements 60 learnable bandpass filters with 1,001-point kernels operating at stride 1 — a critical anti-aliasing decision that preserves temporal fidelity at the cost of compute. Filter frequencies are parameterized by learnable `low_hz` and `band_hz` variables normalized against the Nyquist frequency.

The raw SincNet output feeds into a hierarchical 2D CNN aggregator (`PerShotTemporalEncoder`) that reshapes the 1D temporal features into a 2D time-receiver grid and applies four progressive pooling stages with factors `[5, 5, 4, 2]` temporally and `[2, 2, 2, 1]` spatially. Anti-aliased downsampling via `BlurPool2D` (Gaussian pre-filtering) prevents aliasing artifacts at each stage. Each shot produces a 128-dimensional embedding.

**Key parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Filters | 60 | Optimal spectral coverage without redundancy |
| Kernel size | 1,001 | Low-frequency resolution down to ~10 Hz |
| Stride | 1 | Eliminates aliasing (critical fix over stride > 1) |
| Frequency range | 40–1,000 Hz | Matches seismic signal spectral content |
| Window | Blackman | Superior side-lobe suppression |
| Initialization | Logarithmic | Finer resolution at lower frequencies |

### Graph Attention Fusion

The 5 shot embeddings form a fully-connected graph of 5 nodes. `LightweightGATFusion` applies a single GATv2Conv layer with 4 attention heads (32 dimensions per head), feature dropout of 0.3, and attention dropout of 0.2. GATv2's dynamic attention mechanism computes attention coefficients as a function of both source and target node features, enabling the network to learn asymmetric shot-pair relationships.

A `GlobalAttention` pooling layer with a learned gate network produces the final 128-dimensional graph-level embedding. This fused representation encodes multi-offset spatial dependencies — the network determines which source positions contribute most to each prediction.

### U-Net with FiLM Conditioning

`BaselineUNet` implements a 4-stage encoder-decoder with asymmetric pooling factors `((4,2), (4,2), (5,2), (5,2))` designed for the non-square input geometry. The encoder compresses through progressively deeper feature maps up to 512 channels at the bottleneck. The decoder mirrors this with bilinear upsampling and skip connections. A final interpolation layer maps the output to the target resolution of 300 x 1,259.

At the bottleneck, `ResidualFiLMLayer` modulates the 512-channel feature maps using gamma and beta parameters projected from the 128-dimensional GAT context via a 2-layer MLP (hidden dimension 256). The residual formulation and zero-initialization ensure training stability — the decoder starts as a pure U-Net and progressively incorporates multi-shot context.

**Data flow:**

```
Input (B, 5, 10001, 31)
  → SincNet per shot → 5 × 128-dim embeddings
    → GAT fusion → 128-dim context vector
      → LayerNorm → FiLM modulation at U-Net bottleneck
        → U-Net decoder → Output (B, 1, 300, 1259)
```

---

## Training Pipeline

### Hybrid Loss Function

The champion model uses `RefinedLogSpaceMAEHybridLoss` with three components at fixed weights:

| Component | Weight | Function |
|---|---|---|
| `AdaptiveLogSpaceMAE` | 1.0 | Scale-invariant pixel accuracy with momentum-based adaptive offset |
| `StabilizedSeismicMSSSIM` | 0.12 | Multi-scale structural similarity in log-space with A100 stability fixes |
| `AnisotropicTotalVariationLoss` | 0.007 | Asymmetric regularization (horizontal=1.0, vertical=0.3) for geological layering |

The loss pipeline includes SoftAdapt adaptive weighting and curriculum warmup (log-MAE only for initial epochs). FiLM parameter regularization (L2 on residual gamma and beta) prevents the conditioning pathway from overpowering the spatial decoder.

### Optimization

- **Optimizer:** AdamW with Sharpness-Aware Minimization (SAM) for flatter loss landscape convergence
- **Schedule:** Plateau detection transitioning to cosine annealing
- **Precision:** Mixed-precision training with TF32 disabled for A100 numerical stability
- **Checkpointing:** Best-MAPE model selection with Google Drive persistence

---

## Hardest Problems Solved

### 1. Anti-Aliased Temporal Processing at Full Resolution

Early experiments with strided SincNet convolutions produced aliased frequency representations — the learnable bandpass filters captured correct center frequencies but the downsampled output folded high-frequency content into lower bands. The fix required processing 10,001 time steps at stride 1 (a significant compute cost), combined with `BlurPool1D`/`BlurPool2D` Gaussian pre-filtering before every spatial reduction. The model is alias-free from input to embedding.

> **Resolution:** Stride-1 SincNet convolution with anti-aliased progressive pooling. Correct signal processing at every stage, regardless of cost.

### 2. Stable Multi-Scale Loss on A100 Hardware

The multi-scale structural similarity loss produced NaN gradients on A100 GPUs due to TF32 precision interactions with small-valued intermediate computations in the MS-SSIM kernel. Disabling TF32 globally degraded training throughput. The solution was targeted: stabilize the SSIM computation with epsilon guards and min-clamping in the log-space transform while keeping TF32 disabled only for the loss computation path. This preserved full A100 training speed for the forward and backward passes.

> **Resolution:** Surgical numerical stabilization at the loss boundary, not global precision downgrade. Training speed preserved.

### 3. Zero-Shot FiLM Integration Without Warm-Start

Naively adding FiLM conditioning to a pre-trained U-Net destroyed convergence — the randomly initialized gamma/beta parameters immediately distorted learned feature maps. The `ResidualFiLMLayer` solves this through zero-initialization of the final projection layers. At step 0, `gamma_res = 0` and `beta_res = 0`, so the FiLM output equals the input. The conditioning signal emerges gradually as training progresses, never shocking the decoder. This enabled direct fine-tuning of a pre-trained U-Net with FiLM conditioning added at the bottleneck.

> **Resolution:** Zero-initialized residual FiLM conditioning. The U-Net starts unconditional and learns to integrate context without losing pre-trained spatial representations.

---

## Layering and System Domains

| Layer | Responsibility | Key Modules |
|---|---|---|
| **Temporal Encoding** | Physics-informed frequency decomposition of raw waveforms | `SincConv1d_SeismicAdapted`, `BlurPool1D`, `PerShotTemporalEncoder` |
| **Spatial Fusion** | Multi-shot relationship learning via graph attention | `ShotGraphBuilder`, `LightweightGATFusion`, `GlobalAttention` |
| **Dense Prediction** | High-resolution velocity map generation with context conditioning | `BaselineUNet`, `ResidualFiLMLayer` |
| **Loss Composition** | Multi-objective training with adaptive weighting | `AdaptiveLogSpaceMAE`, `StabilizedSeismicMSSSIM`, `AnisotropicTotalVariationLoss` |
| **Optimization** | Flat-minima seeking with curriculum scheduling | SAM optimizer, plateau-to-cosine schedule, FiLM regularization |
| **Inference** | Deterministic prediction with exact parameter recovery | `corrected_inference_pipeline.py`, champion checkpoint loading |

---

<details>
<summary><strong>Champion Model Configuration</strong></summary>

<br>

```python
CHAMPION_CONFIG = {
    # SincNet Temporal Encoder
    'sinc_out_channels': 60,
    'sinc_kernel_size': 1001,
    'sinc_stride': 1,
    'sinc_min_low_hz': 40,
    'sinc_max_learnable_hz': 1000,
    'sinc_min_band_hz': 10,
    'sinc_window_func': 'blackman',
    'sinc_init_type': 'logarithmic',
    'shot_embedding_dim': 128,

    # GAT Fusion
    'gat_hidden_per_head': 32,
    'gat_num_heads': 4,
    'gat_layers': 1,
    'gat_dropout_feat': 0.3,
    'gat_dropout_attn': 0.2,
    'fused_embedding_dim': 128,

    # U-Net Decoder
    'n_unet_output_channels': 1,
    'unet_bilinear': True,
    'unet_bottleneck_channels': 512,

    # FiLM Conditioning (CRITICAL: must be '2_layer', not 'linear')
    'film_context_dim': 128,
    'film_target_channels': 512,
    'film_generator_mlp_type': '2_layer',
    'film_mlp_hidden_dim': 256,

    # Loss Weights
    'loss_weights': [1.0, 0.12, 0.007],  # [LogMAE, MS-SSIM, ATV]

    # Input Specification
    'sample_rate': 10001,
    'num_receivers': 31,
    'num_shots': 5,
    'source_coordinates': [1, 75, 150, 225, 300],
}
```

**Checkpoint:** `cfg_06_plateau_to_cosine_PhaseB_FiLMFinetune_best_mape.pth`

</details>

<details>
<summary><strong>Repository Structure</strong></summary>

<br>

```
├── sincnet_seismic_encoder.py              # SincNet temporal encoder with anti-aliased pooling
├── seismic_gat_fusion.py                   # Graph Attention Network for multi-shot fusion
├── complete_sincgat_unet_integration.py    # Full SincNet-GAT-UNet pipeline with FiLM conditioning
├── phase2_experimental_framework.py        # Training pipeline: losses, optimization, experiments
├── phase2_integration_notebook_cell.py     # Integration utilities for notebook-based training
├── corrected_inference_pipeline.py         # Champion model inference with exact parameters
├── colab_ready_inference_pipeline.py       # Colab-compatible inference variant
├── enhanced_experimental_suite_with_checkpointing.py  # Experiment runner with Drive checkpointing
├── download_experimental_results.py        # Result download and analysis utilities
├── utils.py                                # Shared utilities and helper functions
├── INFERENCE_PIPELINE_SUMMARY.md           # Critical findings from inference pipeline construction
├── MAIN_898_with_diagnostic_framework.ipynb          # Primary training notebook with diagnostics
├── MAIN_898of_0_898model_speed_and_structure_starter_notebook.ipynb  # Competition starter notebook
└── collectCode.mjs                         # Code collection utility
```

</details>

<br>

---

<div align="center">
  <img src="./terminal-bottom-panel.svg" alt="Terminal Status Bar" width="100%" />
</div>
