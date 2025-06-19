import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv, global_mean_pool, GlobalAttention
from torch_geometric.data import Data, Batch
from sincnet_seismic_encoder import PerShotTemporalEncoder


class LightweightGATFusion(nn.Module):
    """
    Lightweight GAT fusion module for combining multi-shot seismic embeddings.
    
    Based on research findings:
    - GATv2Conv for improved expressiveness
    - 1-2 GAT layers to avoid over-smoothing
    - 4 attention heads with 32-dim per head
    - GlobalAttention for better graph-level pooling
    - Dropout for regularization (0.3 features, 0.2-0.3 attention for small graphs)
    """
    
    def __init__(self, 
                 in_features=128,  # Input from PerShotTemporalEncoder
                 hidden_per_head=32,  # Renamed from gat_hidden_channels_per_head
                 num_heads=4,
                 layers=1,  # Renamed from gat_layers
                 dropout_feat=0.3,
                 dropout_attn=0.2,  # CORRECTED: 0.2-0.3 optimal for small graphs (was 0.6)
                 output_dim=128,  # Renamed from output_embedding_dim
                 use_global_attention=True):
        super().__init__()
        
        self.gat_layers_list = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        current_dim = in_features
        
        # Build GAT layers
        for i in range(layers):  # Using renamed parameter
            self.gat_layers_list.append(
                GATv2Conv(
                    in_channels=current_dim,
                    out_channels=hidden_per_head,  # Using renamed parameter
                    heads=num_heads,
                    concat=True,  # Concatenate multi-head outputs
                    dropout=dropout_attn,
                    add_self_loops=True,
                    bias=True
                )
            )
            # Update dimension after concatenation of heads
            current_dim = hidden_per_head * num_heads  # Using renamed parameter
            self.layer_norms.append(nn.LayerNorm(current_dim))
        
        self.feature_dropout = nn.Dropout(dropout_feat)
        self.activation = nn.ELU(inplace=True)
        
        # Global pooling for graph-level representation
        if use_global_attention:
            # Learnable attention-based pooling
            gate_nn = nn.Sequential(
                nn.Linear(current_dim, current_dim // 2),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(current_dim // 2, 1)
            )
            self.readout_pooling = GlobalAttention(gate_nn)
        else:
            # Simple mean pooling fallback
            self.readout_pooling = global_mean_pool
        
        # Final projection to desired output dimension
        self.final_projection = nn.Linear(current_dim, output_dim)  # Using renamed parameter
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with appropriate scales"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.414)  # Good for ELU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_nodes, edge_index, batch_vector):
        """
        Forward pass through GAT layers and global pooling
        
        Args:
            x_nodes: (Total_Nodes_in_Batch, in_features) - Node features
            edge_index: (2, Num_Edges_Total) - Edge connectivity  
            batch_vector: (Total_Nodes_in_Batch,) - Batch assignment for each node
            
        Returns:
            graph_embeddings: (Batch_Size, output_embedding_dim) - Graph-level embeddings
        """
        h = x_nodes
        
        # Apply GAT layers with residual connections
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers_list, self.layer_norms)):
            h_prev = h
            
            # Feature dropout before GAT
            h = self.feature_dropout(h)
            
            # GAT layer
            h = gat_layer(h, edge_index)
            
            # Activation
            h = self.activation(h)
            
            # Layer norm with residual connection (if dimensions match)
            if h.size(-1) == h_prev.size(-1):
                h = layer_norm(h + h_prev)
            else:
                h = layer_norm(h)
        
        # Global pooling to get graph-level embeddings
        graph_embeddings = self.readout_pooling(h, batch_vector)
        
        # Final projection
        graph_embeddings = self.final_projection(graph_embeddings)
        
        return graph_embeddings


class ShotGraphBuilder:
    """
    Utility class to build graph structures from shot data.
    
    For seismic shots, we typically use:
    - 5 nodes (one per shot position: 1, 75, 150, 225, 300)
    - Fully connected or distance-based connectivity
    """
    
    def __init__(self, num_shots=5, connectivity='full'):
        self.num_shots = num_shots
        self.connectivity = connectivity
        
        # Pre-compute edge index for efficiency
        self.base_edge_index = self._create_base_edge_index()
    
    def _create_base_edge_index(self):
        """Create base edge index for a single graph"""
        if self.connectivity == 'full':
            # Fully connected graph (excluding self-loops, GAT will add them)
            edges = []
            for i in range(self.num_shots):
                for j in range(self.num_shots):
                    if i != j:  # No self-loops, GAT handles them
                        edges.append([i, j])
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                # Fallback: create minimal connectivity
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            
        elif self.connectivity == 'linear':
            # Linear chain connectivity (for spatial ordering)
            edges = []
            for i in range(self.num_shots - 1):
                edges.extend([[i, i+1], [i+1, i]])  # Bidirectional
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
        else:
            raise ValueError(f"Unknown connectivity: {self.connectivity}")
        
        return edge_index
    
    def create_batch(self, shot_embeddings_batch):
        """
        Create PyTorch Geometric batch from shot embeddings
        
        Args:
            shot_embeddings_batch: (B, num_shots, embedding_dim) tensor
            
        Returns:
            Batch object with x, edge_index, batch attributes
        """
        batch_size, num_shots, embedding_dim = shot_embeddings_batch.shape
        
        # Prepare node features: (B * num_shots, embedding_dim)
        x_nodes = shot_embeddings_batch.view(-1, embedding_dim)
        
        # Create edge indices for entire batch
        edge_indices = []
        batch_vector = []
        
        for b in range(batch_size):
            # Offset edge indices for this batch
            offset = b * num_shots
            batch_edge_index = self.base_edge_index + offset
            edge_indices.append(batch_edge_index)
            
            # Batch assignment vector
            batch_vector.extend([b] * num_shots)
        
        # Concatenate all edges
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        batch_vector = torch.tensor(batch_vector, dtype=torch.long)
        
        return x_nodes, edge_index, batch_vector


class SeismicSincNetGAT(nn.Module):
    """
    Complete seismic processing architecture combining SincNet and GAT.
    
    Architecture:
    1. PerShotTemporalEncoder (SincNet + 2D CNN) for each shot
    2. LightweightGATFusion for multi-shot integration  
    3. Output suitable for downstream processing (e.g., U-Net decoder)
    """
    
    def __init__(self,
                 # PerShotTemporalEncoder parameters
                 num_receivers=31,
                 sinc_out_channels=40,
                 sinc_kernel_size=251,
                 sinc_stride=50,
                 sample_rate=10001,  # UPDATED: 10001 Hz based on 1s = 10001 samples
                 sinc_min_low_hz=40,  # CORRECTED: Using sinc_ prefix and updated default value
                 sinc_min_band_hz=10, # CORRECTED: Using sinc_ prefix
                 shot_embedding_dim=128,
                 # GAT parameters
                 hidden_per_head=32,
                 gat_num_heads=4,
                 gat_layers=1,
                 gat_dropout_feat=0.3,
                 gat_dropout_attn=0.2,
                 final_embedding_dim=128,
                 # Graph structure
                 num_shots=5,
                 graph_connectivity='full'):
        super().__init__()
        
        self.num_shots = num_shots
        
        # Per-shot temporal encoder (shared across all shots)
        self.shot_encoder = PerShotTemporalEncoder(
            num_receivers=num_receivers,
            sinc_out_channels=sinc_out_channels,
            sinc_kernel_size=sinc_kernel_size,
            sinc_stride=sinc_stride,
            sample_rate=sample_rate,
            sinc_min_low_hz=sinc_min_low_hz,  # CORRECTED: Using proper parameter name
            sinc_min_band_hz=sinc_min_band_hz, # CORRECTED: Using proper parameter name
            embedding_dim=shot_embedding_dim
        )
        
        # GAT fusion module
        self.gat_fusion = LightweightGATFusion(
            in_features=shot_embedding_dim,
            hidden_per_head=hidden_per_head,
            num_heads=gat_num_heads,
            layers=gat_layers,
            dropout_feat=gat_dropout_feat,
            dropout_attn=gat_dropout_attn,
            output_dim=final_embedding_dim
        )
        
        # Graph builder
        self.graph_builder = ShotGraphBuilder(
            num_shots=num_shots,
            connectivity=graph_connectivity
        )
    
    def forward(self, shot_gathers):
        """
        Forward pass through complete architecture
        
        Args:
            shot_gathers: List of 5 tensors, each (B, 10001, 31)
                         OR tensor of shape (B, 5, 10001, 31)
            
        Returns:
            fused_embeddings: (B, final_embedding_dim) - Fused shot representations
        """
        # Handle different input formats
        if isinstance(shot_gathers, list):
            if len(shot_gathers) != self.num_shots:
                raise ValueError(f"Expected {self.num_shots} shots, got {len(shot_gathers)}")
            # Ensure consistent processing by stacking first
            shot_gathers_tensor = torch.stack(shot_gathers, dim=1)  # (B, 5, 10001, 31)
        elif isinstance(shot_gathers, torch.Tensor):
            if shot_gathers.dim() == 4 and shot_gathers.size(1) == self.num_shots:
                shot_gathers_tensor = shot_gathers  # Already in correct format
            else:
                raise ValueError(f"Invalid shot_gathers shape: {shot_gathers.shape}")
        else:
            raise ValueError("shot_gathers must be list or tensor")
        
        # Convert to list for consistent processing
        shot_list = [shot_gathers_tensor[:, i] for i in range(self.num_shots)]
        
        # Encode each shot independently
        shot_embeddings = []
        for shot in shot_list:
            embedding = self.shot_encoder(shot)  # (B, shot_embedding_dim)
            shot_embeddings.append(embedding)
        
        # Stack embeddings: (B, num_shots, shot_embedding_dim)
        shot_embeddings_batch = torch.stack(shot_embeddings, dim=1)
        
        # Create graph batch
        x_nodes, edge_index, batch_vector = self.graph_builder.create_batch(shot_embeddings_batch)
        
        # Move to same device as input
        device = shot_gathers_tensor.device
        edge_index = edge_index.to(device)
        batch_vector = batch_vector.to(device)
        
        # GAT fusion
        fused_embeddings = self.gat_fusion(x_nodes, edge_index, batch_vector)
        
        return fused_embeddings


# Test function
def test_seismic_sincnet_gat():
    """Test the complete SincNet + GAT architecture"""
    print("🧪 Testing Complete Seismic SincNet + GAT Architecture...")
    
    # Test parameters
    batch_size = 2
    num_shots = 5
    
    # Create dummy shot data
    shot_gathers = []
    for i in range(num_shots):
        shot = torch.randn(batch_size, 10001, 31)
        shot_gathers.append(shot)
    
    print(f"Input: {num_shots} shots, each shape {shot_gathers[0].shape}")
    
    # Create model
    model = SeismicSincNetGAT(
        num_receivers=31,
        sinc_out_channels=40,
        sinc_kernel_size=251,
        sinc_stride=50,
        sample_rate=10001,  # Updated sample rate
        sinc_min_low_hz=40, # CORRECTED: Using proper parameter name and value
        sinc_min_band_hz=10, # CORRECTED: Using proper parameter name
        shot_embedding_dim=128,
        hidden_per_head=32,
        gat_num_heads=4,
        final_embedding_dim=128,
        num_shots=num_shots
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters")
    
    # Forward pass
    try:
        with torch.no_grad():
            fused_embeddings = model(shot_gathers)
            
            print(f"✅ Output shape: {fused_embeddings.shape}")
            print(f"✅ Output range: [{fused_embeddings.min():.3f}, {fused_embeddings.max():.3f}]")
            
            # Check for numerical issues
            if torch.isnan(fused_embeddings).any():
                print("❌ NaN detected in output!")
                return False
            elif torch.isinf(fused_embeddings).any():
                print("❌ Inf detected in output!")
                return False
            else:
                print("✅ Output is numerically stable")
        
        # Test alternative input format (batched tensor)
        print("\n🔄 Testing batched tensor input format...")
        shot_gathers_tensor = torch.stack(shot_gathers, dim=1)  # (B, 5, 10001, 31)
        print(f"Batched input shape: {shot_gathers_tensor.shape}")
        
        with torch.no_grad():
            fused_embeddings_alt = model(shot_gathers_tensor)
            print(f"✅ Alternative input format works: {fused_embeddings_alt.shape}")
            
            # Should be identical results
            if torch.allclose(fused_embeddings, fused_embeddings_alt, atol=1e-6):
                print("✅ Consistent results across input formats")
            else:
                print("⚠️ Different results across input formats")
        
        print("🎉 Complete SincNet + GAT test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_seismic_sincnet_gat() 
