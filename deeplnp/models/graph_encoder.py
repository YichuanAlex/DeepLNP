"""
Graph neural network encoders for molecular representation.
Includes GNN and Uni-Mol based encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GraphEncoder(nn.Module):
    """
    Graph neural network encoder for molecular graphs.
    Uses message passing neural networks (MPNN).
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 39,
        bond_feature_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.atom_feature_dim = atom_feature_dim
        self.bond_feature_dim = bond_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Atom embedding
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        
        # Bond embedding
        self.bond_embedding = nn.Linear(bond_feature_dim, hidden_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def _prepare_edge_index(
        self,
        edge_index: Optional[torch.Tensor],
        num_atoms: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return a safe [2, E] edge_index and a mask over the original edges."""
        if edge_index is None:
            return torch.zeros(2, 0, dtype=torch.long, device=device), None
        
        edge_index = edge_index.to(device)
        if edge_index.dim() == 1:
            if edge_index.numel() == 2:
                edge_index = edge_index.view(2, 1)
            else:
                return torch.zeros(2, 0, dtype=torch.long, device=device), None
        
        if edge_index.dim() != 2:
            return torch.zeros(2, 0, dtype=torch.long, device=device), None
        
        if edge_index.size(0) != 2:
            if edge_index.size(1) == 2:
                edge_index = edge_index.t().contiguous()
            else:
                return torch.zeros(2, 0, dtype=torch.long, device=device), None
        
        edge_index = edge_index.long()
        valid_edges = (
            (edge_index[0] >= 0)
            & (edge_index[1] >= 0)
            & (edge_index[0] < num_atoms)
            & (edge_index[1] < num_atoms)
        )
        return edge_index[:, valid_edges], valid_edges
        
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for graph encoder.
        
        Args:
            atom_features: (N, atom_feat_dim) - padded atom features (N = batch_size * max_atoms)
            bond_features: (E, bond_feat_dim) - concatenated bond features
            edge_index: (2, E) - global edge indices
            batch: (N,) - batch indices for each atom
        
        Returns:
            Dictionary with:
            - graph_embeddings: (batch_size, hidden_dim) per-graph embeddings
        """
        edge_index, valid_edges = self._prepare_edge_index(
            edge_index=edge_index,
            num_atoms=atom_features.size(0),
            device=atom_features.device,
        )
        if bond_features.dim() == 1:
            bond_features = bond_features.unsqueeze(0)
        if valid_edges is not None and valid_edges.numel() == bond_features.size(0):
            bond_features = bond_features[valid_edges.to(bond_features.device)]
        
        # Embed atoms and bonds
        h = self.dropout(F.relu(self.atom_embedding(atom_features)))
        e = self.dropout(F.relu(self.bond_embedding(bond_features)))
        edge_count = edge_index.size(1)
        if e.size(0) != edge_count:
            if edge_count == 0:
                e = e[:0]
            elif e.size(0) > edge_count:
                e = e[:edge_count]
            else:
                pad = torch.zeros(edge_count - e.size(0), e.size(1), dtype=e.dtype, device=e.device)
                e = torch.cat([e, pad], dim=0)
        
        # 确保 h 和 e 数据类型一致（处理混合精度训练）
        if h.dtype != e.dtype:
            e = e.to(h.dtype)
        
        # Create mask for valid atoms (non-padding)
        atom_mask = (atom_features.abs().sum(dim=1) > 0)
        
        # Process all disconnected graphs in one vectorized message-passing pass.
        # The previous Python loop inspected every edge for every graph, making a
        # single epoch take hours and preventing the end-to-end pipeline from scaling.
        if batch is not None:
            for layer in self.message_layers:
                h, e = layer(h, e, edge_index)
            batch = batch.long().to(h.device)
            batch_size = int(batch.max().item()) + 1
            valid_h = h * atom_mask.unsqueeze(-1).to(h.dtype)
            graph_emb = torch.zeros(batch_size, h.size(-1), dtype=h.dtype, device=h.device)
            graph_emb.scatter_add_(0, batch.unsqueeze(-1).expand_as(valid_h), valid_h)
            counts = torch.zeros(batch_size, dtype=h.dtype, device=h.device)
            counts.scatter_add_(0, batch, atom_mask.to(h.dtype))
            graph_emb = graph_emb / counts.clamp(min=1).unsqueeze(-1)
        else:
            # Single graph processing
            for layer in self.message_layers:
                h, e = layer(h, e, edge_index)
            
            # Global pooling (mean over valid atoms)
            if atom_mask.sum() > 0:
                graph_emb = h[atom_mask].mean(dim=0, keepdim=True)
            else:
                graph_emb = h.mean(dim=0, keepdim=True)
        
        # Project
        graph_emb = self.output_proj(graph_emb)
        
        return {
            "graph_embeddings": graph_emb,
        }
    
    def _global_pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Global pooling over atoms in each molecule."""
        # Sum pooling
        num_graphs = batch.max().item() + 1
        graph_emb = torch.zeros(
            num_graphs, h.size(1), dtype=h.dtype, device=h.device
        )
        graph_emb.scatter_add_(0, batch.unsqueeze(1).expand(-1, h.size(1)), h)
        return graph_emb


class MessagePassingLayer(nn.Module):
    """Single message passing layer."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.message_nn = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.update_nn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (N, hidden_dim) node features
            e: (E, hidden_dim) edge features
            edge_index: (2, E) connectivity
        
        Returns:
            Updated node and edge features
        """
        row, col = edge_index
        
        # Message: concatenate sender, receiver, and edge features
        h_row = h[row]
        h_col = h[col]
        msg_input = torch.cat([h_row, h_col, e], dim=-1)
        
        msg = self.message_nn(msg_input)
        
        # 确保 msg 和 h 数据类型一致（处理 message_nn 可能的类型转换）
        if msg.dtype != h.dtype:
            msg = msg.to(h.dtype)
        
        # Aggregate messages (sum)
        # 确保 col 是 long 类型，msg 和 agg 是相同类型
        col = col.long()
        agg = torch.zeros_like(h)
        
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, h.size(1)), msg)
        
        # Update
        h_new = self.update_nn(torch.cat([h, agg], dim=-1))
        h_new = self.layer_norm(h_new + h)  # Residual connection
        
        # Edge update (simple)
        e_new = e  # Can be extended
        
        return h_new, e_new


class UniMolEncoder(nn.Module):
    """
    Uni-Mol based encoder for 3D molecular representation.
    Adapted from TransMA's Uni-Mol implementation.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Token embedding
        self.embed_tokens = nn.Embedding(512, embed_dim)  # Simplified vocabulary
        
        # 3D Transformer encoder
        from .transformer_encoder import TransformerEncoderWithPair
        
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=num_layers,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            attention_heads=num_heads,
            dropout=dropout,
        )
        
        # Distance embedding for 3D structure
        self.distance_embedding = nn.Embedding(128, embed_dim)
        
        # Load pretrained weights
        if pretrain_path:
            self.load_pretrained(pretrain_path)
        
    def forward(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            atom_types: (N,) tensor of atom types
            coords: (N, 3) tensor of 3D coordinates
            padding_mask: (N,) optional padding mask
        
        Returns:
            Dictionary with molecular embeddings
        """
        # Compute pairwise distances
        dist_matrix = self._compute_distances(coords)
        
        # Embed tokens
        x = self.embed_tokens(atom_types)
        
        # Embed distances
        dist_emb = self.distance_embedding(dist_matrix)
        
        # Transformer encoding
        encoder_out, _, _, _, _ = self.encoder(
            emb=x,
            attn_mask=dist_emb,
            padding_mask=padding_mask,
        )
        
        # CLS token representation
        cls_repr = encoder_out[:, 0, :]
        
        return {
            "cls_embedding": cls_repr,
            "atom_embeddings": encoder_out,
        }
    
    def _compute_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix and discretize."""
        # coords: (N, 3)
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (N, N)
        
        # Discretize
        bins = torch.linspace(0, 20, 127, device=coords.device)
        dist_digitized = torch.bucketize(dist, bins)
        
        return dist_digitized
    
    def load_pretrained(self, path: str):
        """Load pretrained Uni-Mol weights."""
        import torch
        
        state_dict = torch.load(path, map_location="cpu")
        
        # Filter and load
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                filtered_dict[k] = v
        
        self.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded pretrained Uni-Mol weights from {path}")
