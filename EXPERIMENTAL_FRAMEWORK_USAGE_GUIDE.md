# Stage 2 Experimental Framework - Complete Usage Guide

## Overview

The enhanced `Stage2ExperimentalFramework` now provides comprehensive exploration of your learning rate and scheduler hyperparameters. This guide explains how to run complete experiments and choose the right strategy for your research goals.

## Key Improvements

### 1. Complete Grid Exploration
- **Before**: Only tested 10 out of 99 possible LR/scheduler combinations
- **Now**: Can test all 99 combinations or any subset you choose
- **Includes**: All scheduler patience and factor combinations for ReduceLROnPlateau

### 2. Focused Experiments
- New `run_focused_lr_experiments_around_champion()` method
- Tests variations around your best known configuration (e.g., the 0.0931% result)
- Efficient for fine-tuning when you have a good baseline

### 3. Better Reporting
- Shows total possible experiments before running
- Provides breakdown of experiment types
- Clearer progress reporting and configuration details

## Experimental Scope Analysis

Your current parameter grids generate:

```
🎯 LR/Scheduler Experiments:
   • 9 LR pairs (3 frontend × 3 unet)
   • None scheduler: 9 configs
   • CosineAnnealingLR: 9 configs  
   • ReduceLROnPlateau: 81 configs (3 patience × 3 factor)
   📈 TOTAL: 99 experiments

🧠 GAT Configuration Experiments:
   📈 TOTAL: 324 experiments

🔗 Fusion Configuration Experiments:
   📈 TOTAL: 3 experiments (only concat_conv implemented)
```

## Recommended Experimental Strategies

### Strategy 1: Focused Tuning (RECOMMENDED)
**When**: You have a good baseline (like your 0.0931% result)  
**Goal**: Fine-tune around known good configuration  
**Time**: ~5 experiments, 3-4 hours  

```python
# Test focused variations around your champion configuration
framework = run_focused_lr_tuning(
    pretrained_weights_path="/path/to/Stage1_weights.pth",
    champion_frontend_lr=2e-5,  # From your 0.0931% result
    champion_unet_lr=5e-6
)
```

### Strategy 2: Comprehensive LR Exploration
**When**: You want to be thorough with LR/scheduler tuning  
**Goal**: Find the absolute best LR/scheduler combination  
**Time**: 99 experiments, 1-3 days depending on hardware  

```python
# Run ALL 99 LR/scheduler combinations
framework = run_comprehensive_lr_exploration(
    pretrained_weights_path="/path/to/Stage1_weights.pth"
)
```

### Strategy 3: Balanced Exploration
**When**: Initial exploration across multiple hyperparameter types  
**Goal**: Sample different areas to identify promising regions  
**Time**: 20 experiments, 6-12 hours  

```python
# Balanced sampling across LR, GAT, and fusion experiments
framework = run_balanced_exploration(
    pretrained_weights_path="/path/to/Stage1_weights.pth",
    total_experiments=20
)
```

### Strategy 4: Quick Validation
**When**: Testing the framework or quick experiments  
**Goal**: Validate setup and basic functionality  
**Time**: 5 experiments, 2-3 hours  

```python
# Quick test with limited experiments
framework = run_quick_lr_validation(
    pretrained_weights_path="/path/to/Stage1_weights.pth",
    num_experiments=5
)
```

## Detailed Usage Examples

### 1. Analyze Scope Before Running
```python
# See what experiments are possible without running any
scope = analyze_experimental_scope("/path/to/Stage1_weights.pth")
print(f"Total LR experiments possible: {scope['lr_schedule']}")
```

### 2. Custom Experiment Configuration
```python
# Run specific experiment types with custom limits
framework = run_systematic_stage2_experiments(
    pretrained_unet_weights_path="/path/to/Stage1_weights.pth",
    experiment_type="lr_schedule",  # or "lr_focused", "gat_config", "fusion", "all"
    max_experiments=25  # Limit to 25 out of 99 possible
)
```

### 3. Manual Framework Usage
```python
# For maximum control
framework = Stage2ExperimentalFramework(
    pretrained_unet_weights_path="/path/to/Stage1_weights.pth",
    base_experiment_name="Custom_Experiment",
    device=device
)

framework.define_parameter_grids()

# Run different experiment types
framework.run_lr_schedule_experiments(max_experiments=None)  # All 99
framework.run_focused_lr_experiments_around_champion(champion_lr_frontend=3e-5)
framework.run_gat_config_experiments(max_experiments=10)

# Generate comprehensive report
framework.generate_summary_report()
```

## Understanding the Results

### Best Configuration Identification
The framework automatically identifies and reports:
- Top 5 best configurations by final MAPE
- Detailed parameters of the best configuration
- Performance across both Phase A and Phase B

### Result Files
- `experiment_results/`: Individual experiment results
- `Stage2_Systematic_*_summary.txt`: Summary report with best configurations

### Key Metrics to Monitor
- **Final MAPE**: Primary performance metric
- **Phase A vs Phase B MAPE**: Understanding training progression
- **Duration**: Training efficiency
- **Configuration Parameters**: For reproducibility

## Making Informed Decisions

### When to Choose Comprehensive LR Exploration
✅ **Choose this when**:
- You have computational resources (1-3 days)
- LR tuning is critical for your research
- You want to establish definitive baselines
- You suspect current LRs are suboptimal

❌ **Avoid when**:
- Time/compute constrained
- Architecture changes are higher priority
- Current LRs already perform well

### When to Choose Focused Tuning
✅ **Choose this when**:
- You have a good baseline (like 0.0931%)
- Want quick improvements
- Time/compute constrained
- Testing scheduler benefits

### Next Steps After LR Tuning
1. **Update base configuration** with best LR settings
2. **Implement architectural changes** (e.g., FiLM fusion)
3. **Run GAT configuration experiments** with optimized LRs
4. **Consider iterative refinement** - re-tune LRs after architectural changes

## Architecture vs LR Tuning Priority

Based on your current results (0.0931% with basic settings), the recommended approach is:

1. **Quick focused LR tuning** (5 experiments) - validate schedulers help
2. **Implement FiLM fusion** - potentially bigger impact than LR fine-tuning
3. **Re-tune LRs for FiLM architecture** - optimal LRs may change
4. **GAT configuration optimization** - if needed after FiLM

This approach balances thoroughness with efficiency and prioritizes changes with higher potential impact.

## Command Summary

```python
# Quick analysis
analyze_experimental_scope("/path/to/weights.pth")

# Recommended first step
run_focused_lr_tuning("/path/to/weights.pth")

# If you need comprehensive exploration
run_comprehensive_lr_exploration("/path/to/weights.pth")

# For balanced initial exploration
run_balanced_exploration("/path/to/weights.pth", total_experiments=20)
``` 