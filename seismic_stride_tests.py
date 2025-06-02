#!/usr/bin/env python3
"""
Comprehensive Test Suite for Seismic Module and Complete U-Net Integration

This test suite specifically checks for:
1. Stride-related issues in the SincNet encoder
2. Dimensional consistency throughout the pipeline
3. Gradient flow and numerical stability
4. Memory efficiency and performance
5. Complete integration testing

Author: AI Assistant
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import traceback
from typing import Dict, List, Tuple, Optional

# Import our modules
from sincnet_seismic_encoder import SincConv1d_SeismicAdapted, PerShotTemporalEncoder
from seismic_gat_fusion import SeismicSincNetGAT, LightweightGATFusion, ShotGraphBuilder
from complete_sincgat_unet_integration import CompleteSincGAT_UNet, BaselineUNet


class SeismicStrideAnalyzer:
    """Analyzer for stride-related issues in seismic processing."""
    
    def __init__(self, sample_rate=10001):
        self.sample_rate = sample_rate
        self.results = {}
    
    def test_sincnet_stride_effects(self, input_length=10001, stride_values=[1, 2, 4, 8, 16, 50]):
        """Test how different stride values affect SincNet output."""
        print("🔍 Testing SincNet Stride Effects...")
        
        results = {}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy input
        dummy_input = torch.randn(2, 1, input_length).to(device)
        print(f"   Input shape: {dummy_input.shape}")
        
        for stride in stride_values:
            print(f"\n   Testing stride={stride}:")
            
            try:
                # Create SincNet with different stride
                sincnet = SincConv1d_SeismicAdapted(
                    out_channels=60,
                    kernel_size=1001,
                    sample_rate=self.sample_rate,
                    stride=stride,
                    padding='same'
                ).to(device)
                
                with torch.no_grad():
                    output = sincnet(dummy_input)
                
                # Calculate output length
                expected_length = input_length // stride
                actual_length = output.shape[2]
                
                # Calculate downsampling factor
                downsampling_factor = input_length / actual_length
                
                results[stride] = {
                    'output_shape': output.shape,
                    'expected_length': expected_length,
                    'actual_length': actual_length,
                    'downsampling_factor': downsampling_factor,
                    'output_range': (output.min().item(), output.max().item()),
                    'output_std': output.std().item(),
                    'memory_usage_mb': output.element_size() * output.nelement() / (1024**2)
                }
                
                print(f"      ✅ Output shape: {output.shape}")
                print(f"      📏 Length: {actual_length} (expected: {expected_length})")
                print(f"      📉 Downsampling: {downsampling_factor:.2f}x")
                print(f"      📊 Range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"      💾 Memory: {results[stride]['memory_usage_mb']:.2f} MB")
                
                # Check for potential issues
                if actual_length != expected_length:
                    print(f"      ⚠️  Length mismatch! Expected {expected_length}, got {actual_length}")
                
                if output.std() < 0.01:
                    print(f"      ⚠️  Very low output variance (std={output.std():.4f})")
                
                if torch.isnan(output).any():
                    print(f"      ❌ NaN detected in output!")
                
                if torch.isinf(output).any():
                    print(f"      ❌ Inf detected in output!")
                
            except Exception as e:
                print(f"      ❌ Failed with stride={stride}: {e}")
                results[stride] = {'error': str(e)}
        
        self.results['sincnet_stride_test'] = results
        return results
    
    def test_encoder_pipeline_dimensions(self, stride_values=[1, 2, 4, 10, 50]):
        """Test the complete encoder pipeline with different strides."""
        print("\n🔍 Testing Complete Encoder Pipeline Dimensions...")
        
        results = {}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create test shot gather: (batch, time, receivers)
        batch_size = 2
        shot_gather = torch.randn(batch_size, 10001, 31).to(device)
        print(f"   Input shot gather shape: {shot_gather.shape}")
        
        for stride in stride_values:
            print(f"\n   Testing complete pipeline with stride={stride}:")
            
            try:
                # Create encoder with specific stride
                encoder = PerShotTemporalEncoder(
                    sample_rate=self.sample_rate,
                    num_receivers=31,
                    time_samples=10001,
                    sinc_out_channels=60,
                    sinc_kernel_size=1001,
                    sinc_stride=stride,
                    sinc_min_low_hz=40,
                    sinc_max_learnable_hz=1000,
                    embedding_dim=128
                ).to(device)
                
                with torch.no_grad():
                    embedding = encoder(shot_gather)
                
                results[stride] = {
                    'embedding_shape': embedding.shape,
                    'expected_shape': (batch_size, 128),
                    'embedding_norm': embedding.norm().item(),
                    'embedding_mean': embedding.mean().item(),
                    'embedding_std': embedding.std().item()
                }
                
                print(f"      ✅ Embedding shape: {embedding.shape}")
                print(f"      📊 Norm: {embedding.norm():.3f}")
                print(f"      📊 Mean: {embedding.mean():.3f}, Std: {embedding.std():.3f}")
                
                # Check for issues
                if embedding.shape != (batch_size, 128):
                    print(f"      ⚠️  Shape mismatch! Expected (2, 128), got {embedding.shape}")
                
                if embedding.norm() < 0.1:
                    print(f"      ⚠️  Very small embedding norm: {embedding.norm():.4f}")
                elif embedding.norm() > 100:
                    print(f"      ⚠️  Very large embedding norm: {embedding.norm():.4f}")
                
                if torch.isnan(embedding).any():
                    print(f"      ❌ NaN detected in embedding!")
                
            except Exception as e:
                print(f"      ❌ Failed with stride={stride}: {e}")
                results[stride] = {'error': str(e)}
        
        self.results['encoder_pipeline_test'] = results
        return results
    
    def test_gat_integration(self):
        """Test GAT fusion with different embedding characteristics."""
        print("\n🔍 Testing GAT Integration...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test data
        batch_size = 2
        num_shots = 5
        embedding_dim = 128
        
        # Create test embeddings with different characteristics
        test_cases = {
            'normal': torch.randn(batch_size, num_shots, embedding_dim),
            'small_values': torch.randn(batch_size, num_shots, embedding_dim) * 0.01,
            'large_values': torch.randn(batch_size, num_shots, embedding_dim) * 10,
            'sparse': torch.randn(batch_size, num_shots, embedding_dim) * (torch.rand(batch_size, num_shots, embedding_dim) > 0.8).float()
        }
        
        results = {}
        
        # Create GAT fusion module
        gat_fusion = LightweightGATFusion(
            in_features=embedding_dim,
            hidden_per_head=32,
            num_heads=4,
            layers=1,
            output_dim=128
        ).to(device)
        
        graph_builder = ShotGraphBuilder(num_shots=num_shots, connectivity='full')
        
        for test_name, embeddings in test_cases.items():
            print(f"\n   Testing {test_name} embeddings:")
            embeddings = embeddings.to(device)
            
            try:
                with torch.no_grad():
                    # Create graph batch
                    x_nodes, edge_index, batch_vector = graph_builder.create_batch(embeddings)
                    
                    # Move to device
                    edge_index = edge_index.to(device)
                    batch_vector = batch_vector.to(device)
                    
                    # Forward through GAT
                    fused_embedding = gat_fusion(
                        x_nodes,
                        edge_index,
                        batch_vector
                    )
                
                results[test_name] = {
                    'input_range': (embeddings.min().item(), embeddings.max().item()),
                    'input_norm': embeddings.norm().item(),
                    'output_shape': fused_embedding.shape,
                    'output_range': (fused_embedding.min().item(), fused_embedding.max().item()),
                    'output_norm': fused_embedding.norm().item()
                }
                
                print(f"      ✅ Input range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
                print(f"      ✅ Output shape: {fused_embedding.shape}")
                print(f"      ✅ Output range: [{fused_embedding.min():.3f}, {fused_embedding.max():.3f}]")
                print(f"      📊 Norm change: {embeddings.norm():.3f} → {fused_embedding.norm():.3f}")
                
                # Check for issues
                if torch.isnan(fused_embedding).any():
                    print(f"      ❌ NaN detected in GAT output!")
                
                if fused_embedding.norm() < 0.01:
                    print(f"      ⚠️  Very small GAT output norm")
                
            except Exception as e:
                print(f"      ❌ Failed with {test_name} embeddings: {e}")
                results[test_name] = {'error': str(e)}
        
        self.results['gat_integration_test'] = results
        return results
    
    def analyze_computational_efficiency(self, stride_values=[1, 2, 4, 10, 50]):
        """Analyze computational efficiency with different strides."""
        print("\n⚡ Analyzing Computational Efficiency...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = {}
        
        # Test data
        batch_size = 4
        shot_gather = torch.randn(batch_size, 10001, 31).to(device)
        
        for stride in stride_values:
            print(f"\n   Testing efficiency with stride={stride}:")
            
            try:
                # Create encoder
                encoder = PerShotTemporalEncoder(
                    sample_rate=self.sample_rate,
                    sinc_stride=stride,
                    embedding_dim=128
                ).to(device)
                
                # Warm up
                with torch.no_grad():
                    _ = encoder(shot_gather)
                
                # Time the forward pass
                torch.cuda.synchronize() if device == 'cuda' else None
                start_time = time.time()
                
                num_runs = 50
                for _ in range(num_runs):
                    with torch.no_grad():
                        embedding = encoder(shot_gather)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_runs * 1000  # ms
                
                # Memory usage
                if device == 'cuda':
                    memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                else:
                    memory_used = 0
                
                results[stride] = {
                    'avg_time_ms': avg_time,
                    'memory_mb': memory_used,
                    'throughput_samples_per_sec': (batch_size * 10001) / (avg_time / 1000),
                    'embedding_shape': embedding.shape
                }
                
                print(f"      ⏱️  Avg time: {avg_time:.2f} ms")
                print(f"      💾 Memory: {memory_used:.1f} MB")
                print(f"      🚀 Throughput: {results[stride]['throughput_samples_per_sec']:.0f} samples/sec")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[stride] = {'error': str(e)}
        
        self.results['efficiency_test'] = results
        return results
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        # Summary of stride effects
        if 'sincnet_stride_test' in self.results:
            print("\n🔍 SincNet Stride Analysis:")
            for stride, data in self.results['sincnet_stride_test'].items():
                if 'error' not in data:
                    print(f"   Stride {stride}: {data['downsampling_factor']:.1f}x downsampling, "
                          f"{data['actual_length']} samples output")
                else:
                    print(f"   Stride {stride}: FAILED - {data['error']}")
        
        # Efficiency comparison
        if 'efficiency_test' in self.results:
            print("\n⚡ Efficiency Comparison:")
            print("   Stride  | Time (ms) | Memory (MB) | Throughput (samples/s)")
            print("   --------|-----------|-------------|----------------------")
            for stride, data in self.results['efficiency_test'].items():
                if 'error' not in data:
                    print(f"   {stride:6d}  | {data['avg_time_ms']:8.2f}  | {data['memory_mb']:10.1f}  | {data['throughput_samples_per_sec']:20.0f}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        # Analyze stride performance
        if 'efficiency_test' in self.results:
            best_stride = None
            best_score = float('inf')
            
            for stride, data in self.results['efficiency_test'].items():
                if 'error' not in data:
                    # Score based on time and memory (lower is better)
                    score = data['avg_time_ms'] + data['memory_mb'] * 0.1
                    if score < best_score:
                        best_score = score
                        best_stride = stride
            
            if best_stride is not None:
                print(f"   🏆 Best performing stride: {best_stride}")
        
        # Quality recommendations
        if 'sincnet_stride_test' in self.results:
            stable_strides = []
            for stride, data in self.results['sincnet_stride_test'].items():
                if 'error' not in data and data['output_std'] > 0.1:
                    stable_strides.append(stride)
            
            if stable_strides:
                print(f"   📊 Numerically stable strides: {stable_strides}")
        
        print("\n   🎯 For seismic data:")
        print("      - Use stride=1 for maximum frequency resolution")
        print("      - Use stride=2-4 for balanced performance/quality")
        print("      - Avoid stride>10 unless computational constraints are severe")
        print("      - Monitor for NaN/Inf values in outputs")


class CompleteIntegrationTester:
    """Comprehensive tester for the complete U-Net integration."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
    
    def test_model_creation(self):
        """Test model creation with different configurations."""
        print("🏗️  Testing Model Creation...")
        
        configs = {
            'default': {},
            'small_stride': {'sinc_stride': 1},
            'medium_stride': {'sinc_stride': 4},
            'large_stride': {'sinc_stride': 10},
            'small_filters': {'sinc_out_channels': 30},
            'large_filters': {'sinc_out_channels': 80}
        }
        
        results = {}
        
        for config_name, params in configs.items():
            print(f"\n   Testing {config_name} configuration:")
            
            try:
                model = CompleteSincGAT_UNet(**params).to(self.device)
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                results[config_name] = {
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'model_size_mb': total_params * 4 / (1024**2),
                    'creation_success': True
                }
                
                print(f"      ✅ Created successfully")
                print(f"      📊 Parameters: {total_params:,} ({trainable_params:,} trainable)")
                print(f"      💾 Model size: {results[config_name]['model_size_mb']:.1f} MB")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[config_name] = {'creation_success': False, 'error': str(e)}
        
        self.results['model_creation'] = results
        return results
    
    def test_forward_pass(self):
        """Test forward pass with various input configurations."""
        print("\n🚀 Testing Forward Pass...")
        
        # Create model
        model = CompleteSincGAT_UNet(sinc_stride=1).to(self.device)
        model.eval()
        
        test_cases = {
            'normal_batch': (2, 5, 10001, 31),
            'single_sample': (1, 5, 10001, 31),
            'large_batch': (8, 5, 10001, 31)
        }
        
        results = {}
        
        for test_name, input_shape in test_cases.items():
            print(f"\n   Testing {test_name} (shape: {input_shape}):")
            
            try:
                # Create input
                dummy_input = torch.randn(*input_shape).to(self.device)
                
                # Forward pass
                start_time = time.time()
                with torch.no_grad():
                    output = model(dummy_input)
                end_time = time.time()
                
                results[test_name] = {
                    'input_shape': input_shape,
                    'output_shape': output.shape,
                    'forward_time_ms': (end_time - start_time) * 1000,
                    'output_range': (output.min().item(), output.max().item()),
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item()
                }
                
                print(f"      ✅ Output shape: {output.shape}")
                print(f"      ⏱️  Forward time: {results[test_name]['forward_time_ms']:.2f} ms")
                print(f"      📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"      📊 Mean: {output.mean():.3f}, Std: {output.std():.3f}")
                
                if results[test_name]['has_nan']:
                    print(f"      ❌ NaN detected!")
                if results[test_name]['has_inf']:
                    print(f"      ❌ Inf detected!")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[test_name] = {'error': str(e)}
        
        self.results['forward_pass'] = results
        return results
    
    def test_gradient_flow(self):
        """Test gradient flow through the complete model."""
        print("\n🌊 Testing Gradient Flow...")
        
        model = CompleteSincGAT_UNet(sinc_stride=1).to(self.device)
        model.train()
        
        # Create dummy data and target
        dummy_input = torch.randn(2, 5, 10001, 31).to(self.device)
        dummy_target = torch.randn(2, 1, 300, 1259).to(self.device)
        
        try:
            # Forward pass
            output = model(dummy_input)
            
            # Compute loss
            loss = F.mse_loss(output, dummy_target)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            gradient_stats = {}
            
            # SincNet gradients
            if hasattr(model.shot_encoder.sinc_layer, 'f_center_norm'):
                if model.shot_encoder.sinc_layer.f_center_norm.grad is not None:
                    grad = model.shot_encoder.sinc_layer.f_center_norm.grad
                    gradient_stats['sincnet_center'] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.abs().max().item()
                    }
            
            # GAT gradients
            gat_grads = []
            for name, param in model.gat_fusion.named_parameters():
                if param.grad is not None:
                    gat_grads.append(param.grad.norm().item())
            
            if gat_grads:
                gradient_stats['gat'] = {
                    'mean_norm': np.mean(gat_grads),
                    'max_norm': np.max(gat_grads),
                    'min_norm': np.min(gat_grads)
                }
            
            # U-Net gradients
            unet_grads = []
            for name, param in model.unet.named_parameters():
                if param.grad is not None:
                    unet_grads.append(param.grad.norm().item())
            
            if unet_grads:
                gradient_stats['unet'] = {
                    'mean_norm': np.mean(unet_grads),
                    'max_norm': np.max(unet_grads),
                    'min_norm': np.min(unet_grads)
                }
            
            self.results['gradient_flow'] = {
                'loss': loss.item(),
                'gradient_stats': gradient_stats,
                'success': True
            }
            
            print(f"   ✅ Loss: {loss.item():.6f}")
            print(f"   📊 Gradient Statistics:")
            for component, stats in gradient_stats.items():
                print(f"      {component}: {stats}")
            
        except Exception as e:
            print(f"   ❌ Gradient flow test failed: {e}")
            self.results['gradient_flow'] = {'success': False, 'error': str(e)}
        
        return self.results.get('gradient_flow', {})
    
    def test_memory_usage(self):
        """Test memory usage patterns."""
        print("\n💾 Testing Memory Usage...")
        
        if self.device != 'cuda':
            print("   ⚠️  Memory testing only available on CUDA")
            return {}
        
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n   Testing batch size {batch_size}:")
            
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create model and input
                model = CompleteSincGAT_UNet(sinc_stride=1).to(self.device)
                dummy_input = torch.randn(batch_size, 5, 10001, 31).to(self.device)
                
                # Forward pass
                output = model(dummy_input)
                
                # Measure memory
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                memory_cached = torch.cuda.memory_reserved() / (1024**2)  # MB
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                
                results[batch_size] = {
                    'memory_allocated_mb': memory_allocated,
                    'memory_cached_mb': memory_cached,
                    'peak_memory_mb': peak_memory,
                    'output_shape': output.shape
                }
                
                print(f"      📊 Allocated: {memory_allocated:.1f} MB")
                print(f"      📊 Cached: {memory_cached:.1f} MB")
                print(f"      📊 Peak: {peak_memory:.1f} MB")
                
                # Cleanup
                del model, dummy_input, output
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[batch_size] = {'error': str(e)}
        
        self.results['memory_usage'] = results
        return results
    
    def test_asymmetric_unet_pooling(self):
        """Test U-Net encoder/decoder with specified asymmetric pooling/upsampling."""
        print("\n📉 Testing Asymmetric U-Net Pooling and Upsampling...")
        
        n_channels_in = 3  # Example input channels
        n_channels_out = 1 # Example output channels
        factor = 2 # Assuming bilinear=True for factor calculation as in BaselineUNet
        
        try:
            # Instantiate BaselineUNet
            # The BaselineUNet in complete_sincgat_unet_integration.py already has the champion asymmetric config
            model = BaselineUNet(n_channels_in=n_channels_in, n_channels_out=n_channels_out, bilinear=True).to(self.device)
            model.eval()
            print(f"   ✅ BaselineUNet with asymmetric pooling/upsampling instantiated.")

            # Create dummy input tensor (e.g., resembling an image after some initial convs)
            # For U-Net, a typical input might be (Batch, Channels, Height, Width)
            # Let's use a common starting size that will be downsampled, e.g., 256x256 or similar
            # However, the actual U-Net in CompleteSincGAT_UNet takes features from the SincNet+GAT part.
            # Let's infer a suitable starting H, W for the U-Net part from CompleteSincGAT_UNet structure.
            # The output of SincNet part is (B, sinc_C_out, T_sinc_out, R)
            # Example: sinc_C_out=60, T_sinc_out might be around 10001/sinc_stride, R=31.
            # The U-Net input n_channels_in is dynamic based on SincNet output. Let's use a fixed value for standalone test.
            # The U-Net in CompleteSincGAT_UNet is used on features extracted by the PerShotTemporalEncoder and processed by GAT.
            # The effective input to the U-Net (self.unet.inc) within CompleteSincGAT_UNet
            # comes from the `reshape_for_unet` which makes it (B, C_shot_encoder_output, H_spatial, W_temporal)
            # Let's use an initial dummy input size for U-Net that is large enough.
            # The champion config expects specific downsampling.
            # Initial (H,W) for U-Net: (300, 1259) after interpolation in the full model.
            # The U-Net encoder works on this. Let's use something that results in these dimensions BEFORE the final interpolation in BaselineUNet's decoder.
            # The final interpolation to (300, 1259) happens *after* self.outc.
            # Let's pick an initial size that the asymmetric pooling can handle, e.g., (256, 1024) for H,W
            # The provided pooling factors are: H: 4*4*5*5 = 400x, W: 1*1*1*1 = 1x (this seems wrong from the user query as W pooling is also (X,Y) pairs)
            # Let's re-check the Down class and the U-Net structure.
            # self.down1 = Down(64, 128,   pool_kernel_stride=((4,2), (4,1)))  -> H_pool_kernel=4, H_stride=4, W_pool_kernel=2, W_stride=1
            # This means H is divided by 4, W is divided by 1 (stride is 1).
            # This will require input H to be a multiple of 4*4*5*5 = 400 and W to be a multiple of 1*1*1*1 = 1
            # Let's use H=400, W=128 for testing the U-Net standalone. This will become H=1, W=128 at bottleneck.
            # The final output (before interpolation) should match this. Then upsampling should bring it back to H=400, W=128.

            dummy_input_unet = torch.randn(2, n_channels_in, 400, 256).to(self.device) # B, C, H, W
            print(f"   Input U-Net shape: {dummy_input_unet.shape}")

            # Forward encoder
            x1, x2, x3, x4, x5 = model.forward_encoder(dummy_input_unet)
            
            encoder_shapes = {
                'x1 (inc)': x1.shape,
                'x2 (down1)': x2.shape,
                'x3 (down2)': x3.shape,
                'x4 (down3)': x4.shape,
                'x5 (bottleneck)': x5.shape
            }
            print(f"   Encoder shapes: {encoder_shapes}")

            # Expected shapes based on pooling: pool_kernel_stride=((K_h,K_w), (S_h,S_w))
            # MaxPool2d uses (kernel_size, stride)
            # down1 = ((4,2), (4,1)) -> H_kernel=4, H_stride=4; W_kernel=2, W_stride=1. Output H_out = H_in/4, W_out = W_in/1 (if padding=0)
            # Let's assume kernel_size = (K_h,K_w) and stride=(S_h,S_w)
            # Corrected interpretation: pool_kernel_stride for MaxPool2d should be kernel_size and stride arguments directly.
            # If Down(..., pool_kernel_stride=((4,2), (4,1))) means kernel_size=(4,2), stride=(4,1)
            # H_in=400, W_in=256
            # x1: (B, 64, 400, 256)
            # x2 (down1 with ks=(4,2), s=(4,1)): H_out = (400-4)/4+1 = 100. W_out = (256-2)/1+1 = 255. Shape: (B, 128, 100, 255)
            # x3 (down2 with ks=(4,2), s=(4,1)): H_out = (100-4)/4+1 = 25. W_out = (255-2)/1+1 = 254. Shape: (B, 256, 25, 254)
            # x4 (down3 with ks=(5,2), s=(5,1)): H_out = (25-5)/5+1 = 5. W_out = (254-2)/1+1 = 253. Shape: (B, 512, 5, 253)
            # x5 (down4 with ks=(5,2), s=(5,1)): H_out = (5-5)/5+1 = 1. W_out = (253-2)/1+1 = 252. Shape: (B, 1024//f, 1, 252)

            expected_h = 400
            expected_w = 256
            current_h, current_w = expected_h, expected_w

            # Check x1
            assert x1.shape == (dummy_input_unet.shape[0], 64, current_h, current_w), f"x1 shape mismatch: {x1.shape}"
            # Check x2
            current_h = (current_h - 4) // 4 + 1; current_w = (current_w - 2) // 1 + 1
            assert x2.shape == (dummy_input_unet.shape[0], 128, current_h, current_w), f"x2 shape mismatch: {x2.shape}"
            # Check x3
            current_h = (current_h - 4) // 4 + 1; current_w = (current_w - 2) // 1 + 1
            assert x3.shape == (dummy_input_unet.shape[0], 256, current_h, current_w), f"x3 shape mismatch: {x3.shape}"
            # Check x4
            current_h = (current_h - 5) // 5 + 1; current_w = (current_w - 2) // 1 + 1
            assert x4.shape == (dummy_input_unet.shape[0], 512, current_h, current_w), f"x4 shape mismatch: {x4.shape}"
            # Check x5
            current_h = (current_h - 5) // 5 + 1; current_w = (current_w - 2) // 1 + 1
            assert x5.shape == (dummy_input_unet.shape[0], 1024 // factor, current_h, current_w), f"x5 shape mismatch: {x5.shape}"
            print(f"   ✅ Encoder shapes match expectations.")

            # Forward decoder
            # upsample_scale_factor=(Scale_H, Scale_W)
            # up1: (5,1), up2: (5,1), up3: (4,1), up4: (4,1)
            output_decoder = model.forward_decoder(x5, x4, x3, x2, x1)
            decoder_output_shape = output_decoder.shape # This is after the final F.interpolate
            print(f"   Decoder output shape (after final interpolate): {decoder_output_shape}")
            
            # The BaselineUNet's forward_decoder has a final F.interpolate(..., size=(300, 1259), ...)
            # So, the output_decoder.shape will be (B, n_channels_out, 300, 1259)
            expected_final_shape = (dummy_input_unet.shape[0], n_channels_out, 300, 1259)
            assert decoder_output_shape == expected_final_shape, f"Decoder output shape mismatch. Expected {expected_final_shape}, Got {decoder_output_shape}"
            print(f"   ✅ Decoder output shape matches final interpolation target.")

            # Check for NaN/Inf
            has_nan = torch.isnan(output_decoder).any().item()
            has_inf = torch.isinf(output_decoder).any().item()
            if not has_nan and not has_inf:
                print(f"   ✅ U-Net Asymmetric Test: Numerically stable.")
            else:
                print(f"   ❌ U-Net Asymmetric Test: NaN detected: {has_nan}, Inf detected: {has_inf}")
                self.results['asymmetric_unet_pooling'] = {'success': False, 'error': 'NaN or Inf in output'}
                return False

            self.results['asymmetric_unet_pooling'] = {'success': True, 'encoder_shapes': encoder_shapes, 'decoder_output_shape': decoder_output_shape}
            print("   🎉 Asymmetric U-Net pooling/upsampling test passed!")
            return True

        except Exception as e:
            print(f"   ❌ Asymmetric U-Net test failed: {e}")
            traceback.print_exc()
            self.results['asymmetric_unet_pooling'] = {'success': False, 'error': str(e)}
            return False

    def generate_integration_report(self):
        """Generate comprehensive integration test report."""
        print("\n" + "="*80)
        print("🔬 COMPLETE INTEGRATION TEST REPORT")
        print("="*80)
        
        # Model creation summary
        if 'model_creation' in self.results:
            print("\n🏗️  Model Creation Results:")
            for config, data in self.results['model_creation'].items():
                if data.get('creation_success', False):
                    print(f"   ✅ {config}: {data['total_params']:,} params, {data['model_size_mb']:.1f} MB")
                else:
                    print(f"   ❌ {config}: Failed")
        
        # Forward pass summary
        if 'forward_pass' in self.results:
            print("\n🚀 Forward Pass Results:")
            for test, data in self.results['forward_pass'].items():
                if 'error' not in data:
                    status = "✅" if not (data['has_nan'] or data['has_inf']) else "⚠️"
                    print(f"   {status} {test}: {data['forward_time_ms']:.1f} ms")
                else:
                    print(f"   ❌ {test}: Failed")
        
        # Gradient flow summary
        if 'gradient_flow' in self.results:
            print("\n🌊 Gradient Flow:")
            if self.results['gradient_flow'].get('success', False):
                print(f"   ✅ Gradients flowing correctly")
                print(f"   📊 Loss: {self.results['gradient_flow']['loss']:.6f}")
            else:
                print(f"   ❌ Gradient flow issues detected")
        
        # Memory usage summary
        if 'memory_usage' in self.results:
            print("\n💾 Memory Usage:")
            print("   Batch Size | Peak Memory (MB)")
            print("   -----------|-----------------")
            for batch_size, data in self.results['memory_usage'].items():
                if 'error' not in data:
                    print(f"   {batch_size:10d} | {data['peak_memory_mb']:15.1f}")
        
        # Add asymmetric U-Net test results to report
        if 'asymmetric_unet_pooling' in self.results:
            print("\n📉 Asymmetric U-Net Pooling/Upsampling Results:")
            if self.results['asymmetric_unet_pooling'].get('success', False):
                print(f"   ✅ Test Passed.")
                print(f"      Encoder Shapes: {self.results['asymmetric_unet_pooling']['encoder_shapes']}")
                print(f"      Decoder Output Shape: {self.results['asymmetric_unet_pooling']['decoder_output_shape']}")
            else:
                print(f"   ❌ Test Failed: {self.results['asymmetric_unet_pooling'].get('error', 'Unknown error')}")
        
        print("\n🎯 INTEGRATION STATUS:")
        all_tests_passed = True
        
        # Check each test category
        test_categories = ['model_creation', 'forward_pass', 'gradient_flow', 'asymmetric_unet_pooling']
        for category in test_categories:
            if category in self.results:
                if category == 'model_creation':
                    passed = any(data.get('creation_success', False) for data in self.results[category].values())
                elif category == 'forward_pass':
                    passed = any('error' not in data and not (data.get('has_nan', False) or data.get('has_inf', False)) 
                               for data in self.results[category].values())
                elif category == 'gradient_flow':
                    passed = self.results[category].get('success', False)
                elif category == 'asymmetric_unet_pooling':
                    passed = self.results[category].get('success', False)
                else:
                    passed = True
                
                print(f"   {'✅' if passed else '❌'} {category.replace('_', ' ').title()}")
                if not passed:
                    all_tests_passed = False
            else:
                print(f"   ⚠️  {category.replace('_', ' ').title()} - Not tested")
                all_tests_passed = False
        
        if all_tests_passed:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
            print("   Ready for training and deployment.")
        else:
            print("\n⚠️  SOME TESTS FAILED OR INCOMPLETE")
            print("   Please review the issues above before proceeding.")


def main():
    """Main test runner."""
    print("🧪 STARTING COMPREHENSIVE SEISMIC MODULE TESTING")
    print("="*80)
    
    # Test 1: Stride analysis
    print("\n📊 PHASE 1: STRIDE ANALYSIS")
    stride_analyzer = SeismicStrideAnalyzer(sample_rate=10001)
    
    stride_analyzer.test_sincnet_stride_effects()
    stride_analyzer.test_encoder_pipeline_dimensions()
    stride_analyzer.test_gat_integration()
    stride_analyzer.analyze_computational_efficiency()
    stride_analyzer.generate_report()
    
    # Test 2: Complete integration
    print("\n🔬 PHASE 2: COMPLETE INTEGRATION TESTING")
    integration_tester = CompleteIntegrationTester()
    
    integration_tester.test_model_creation()
    integration_tester.test_forward_pass()
    integration_tester.test_gradient_flow()
    
    if torch.cuda.is_available():
        integration_tester.test_memory_usage()
    
    integration_tester.test_asymmetric_unet_pooling()
    
    integration_tester.generate_integration_report()
    
    print("\n🏁 TESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main() 