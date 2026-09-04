"""
Multi-modal Fusion Module for LNP formulation modeling.

This module implements cross-attention fusion for combining:
1. Molecular representations (1D SMILES, 2D Graph, 3D Conformer)
2. Formulation parameters (molar ratios, process conditions)
3. Physicochemical descriptors (pKa, LogP, etc.)

Key innovations:
1. Cross-attention between different modalities
2. Mixture of Experts (MoE) for heterogeneous data
3. Adaptive weighting of different features
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return x * self.weight + self.bias


class CrossAttention(nn.Module):
    """
    Cross-Attention module for fusing two different modalities.
    
    Example: Use molecular features to query formulation features
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        # Projection layers
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(value_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multi-head output projection
        self.multihead_out = nn.Linear(embed_dim, embed_dim)
        
        self.scaling = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: (batch, seq_len_q, query_dim)
            key: (batch, seq_len_k, key_dim)
            value: (batch, seq_len_k, value_dim)
            key_padding_mask: (batch, seq_len_k)
            attn_mask: (seq_len_q, seq_len_k)
        
        Returns:
            output: (batch, seq_len_q, embed_dim)
            attn_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        bsz, tgt_len, _ = query.size()
        src_len = key.size(1)
        
        # Project to Q, K, V
        q = self.q_proj(query)  # (batch, seq_len_q, embed_dim)
        k = self.k_proj(key)    # (batch, seq_len_k, embed_dim)
        v = self.v_proj(value)  # (batch, seq_len_k, embed_dim)
        
        # Reshape for multihead: (batch, num_heads, seq_len, head_dim)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply masks
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len_q, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        output = self.multihead_out(attn_output)
        
        return output, attn_weights


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) for handling heterogeneous LNP data.
    
    Different experts specialize in:
    1. Ionizable lipid features
    2. Helper lipid features
    3. Cholesterol features
    4. PEG-lipid features
    5. Process parameters
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim or input_dim * 2
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, output_dim),
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, input_dim)
        
        Returns:
            output: (batch, output_dim) - Combined output from all experts
            gate_weights: (batch, num_experts) - Expert weighting
        """
        # Compute gate weights
        gate_logits = self.gate(x)  # (batch, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, num_experts)
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch, output_dim)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch, output_dim, num_experts)
        
        # Weighted combination
        output = torch.matmul(expert_outputs, gate_weights.unsqueeze(-1)).squeeze(-1)
        
        return output, gate_weights


class MultiModalFusion(nn.Module):
    """
    Multi-modal Fusion module for LNP formulation modeling.
    
    Combines:
    1. Molecular representations (from GNN/Transformer)
    2. 3D structural features (from Uni-Mol)
    3. Image features (from CNN)
    4. Embedding features (from MLP)
    5. Formulation parameters (molar ratios, process conditions)
    6. Physicochemical descriptors
    
    Architecture:
    1. Modality-specific encoders
    2. Cross-attention fusion
    3. MoE for final integration
    """
    
    def __init__(
        self,
        mol_feat_dim: int = 512,
        struct_feat_dim: int = 768,
        formul_feat_dim: int = 256,
        physchem_feat_dim: int = 128,
        image_feat_dim: Optional[int] = None,
        embedding_feat_dim: Optional[int] = None,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_experts: int = 4,
        use_cross_attention: bool = True,
        use_3d_conformer: bool = True,
    ):
        super().__init__()
        
        self.mol_feat_dim = mol_feat_dim
        self.struct_feat_dim = struct_feat_dim
        self.formul_feat_dim = formul_feat_dim
        self.physchem_feat_dim = physchem_feat_dim
        self.image_feat_dim = image_feat_dim
        self.embedding_feat_dim = embedding_feat_dim
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_3d_conformer = use_3d_conformer
        
        # Feature projection to common dimension
        self.mol_proj = nn.Linear(mol_feat_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_feat_dim, hidden_dim) if struct_feat_dim else None
        self.formul_proj = nn.Linear(formul_feat_dim, hidden_dim)
        self.physchem_proj = nn.Linear(physchem_feat_dim, hidden_dim)
        
        # Image and embedding projections (if available)
        self.image_proj = nn.Linear(image_feat_dim, hidden_dim) if image_feat_dim else None
        self.embedding_proj = nn.Linear(embedding_feat_dim, hidden_dim) if embedding_feat_dim else None
        
        # 3D Conformer-aware Cross-Attention
        if use_cross_attention and use_3d_conformer and struct_feat_dim:
            # Molecular features query 3D structural features
            self.mol_struct_attn = CrossAttention(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.mol_struct_attn = None

        if use_cross_attention:
            # Formulation features query molecular-structural fusion
            self.formul_mol_attn = CrossAttention(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.formul_mol_attn = None
        
        # Layer normalization
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)
        self.norm4 = LayerNorm(hidden_dim * 2)
        
        # Dropout for regularization
        self.fusion_dropout = nn.Dropout(dropout)
        
        # Projection for concatenated features
        concat_input_dim = hidden_dim * 2
        self.concat_proj = nn.Linear(concat_input_dim, hidden_dim * 2)
        
        # Mixture of Experts - 增强版，更多专家
        self.moe = MixtureOfExperts(
            input_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )
        
        # Final fusion
        final_input_dim = hidden_dim * 3  # pre-MoE fused features + MoE output
        self.fusion_proj = nn.Linear(final_input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        mol_features: Tensor,
        struct_features: Optional[Tensor] = None,
        formulation_features: Optional[Tensor] = None,
        physchem_features: Optional[Tensor] = None,
        image_features: Optional[Tensor] = None,
        embedding_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            mol_features: (batch, mol_feat_dim) - Molecular representation
            struct_features: (batch, struct_feat_dim) - 3D structural features
            formulation_features: (batch, formul_feat_dim) - Formulation parameters
            physchem_features: (batch, physchem_feat_dim) - Physicochemical descriptors
            image_features: (batch, image_feat_dim) - Image features
            embedding_features: (batch, embedding_feat_dim) - Embedding features
        
        Returns:
            fused_features: (batch, hidden_dim) - Fused representation
            attention_weights: Dict - Cross-attention weights for interpretability
            expert_weights: (batch, num_experts) - MoE expert weights
        """
        # Project to common dimension
        mol_feat = self.mol_proj(mol_features)  # (batch, hidden_dim)
        mol_feat = self.norm1(mol_feat)
        
        # Cross-attention: Molecular -> Structural (if available)
        # ========== 3D Conformer-aware Cross-Attention Fusion ==========
        # 1. Molecular features query 3D structural features (if available)
        if struct_features is not None and self.struct_proj is not None and self.mol_struct_attn is not None:
            struct_feat = self.struct_proj(struct_features)
            struct_feat = struct_feat.unsqueeze(1)  # (batch, 1, hidden_dim)
            mol_feat_expanded = mol_feat.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            mol_struct_feat, mol_struct_attn = self.mol_struct_attn(
                query=mol_feat_expanded,
                key=struct_feat,
                value=struct_feat,
            )
            mol_struct_feat = mol_struct_feat.squeeze(1)  # (batch, hidden_dim)
            # Residual connection
            mol_struct_feat = self.norm1(mol_feat + mol_struct_feat)
        else:
            mol_struct_feat = mol_feat
            mol_struct_attn = None
        
        # 2. Add image features if available (细胞图像特征)
        if image_features is not None and self.image_proj is not None:
            img_feat = self.image_proj(image_features)
            img_feat = self.fusion_dropout(img_feat)
            mol_struct_feat = mol_struct_feat + img_feat
            mol_struct_feat = self.norm2(mol_struct_feat)
        
        # 3. Add embedding features if available (预训练分子嵌入)
        if embedding_features is not None and self.embedding_proj is not None:
            emb_feat = self.embedding_proj(embedding_features)
            emb_feat = self.fusion_dropout(emb_feat)
            mol_struct_feat = mol_struct_feat + emb_feat
            mol_struct_feat = self.norm2(mol_struct_feat)
        
        # 4. Formulation features query molecular-structural fusion (配方感知)
        if formulation_features is not None:
            formul_feat = self.formul_proj(formulation_features)
            formul_feat = formul_feat.unsqueeze(1)  # (batch, 1, hidden_dim)
            mol_struct_feat_expanded = mol_struct_feat.unsqueeze(1)
            
            if self.formul_mol_attn is not None:
                formul_mol_feat, formul_mol_attn = self.formul_mol_attn(
                    query=formul_feat,
                    key=mol_struct_feat_expanded,
                    value=mol_struct_feat_expanded,
                )
                formul_mol_feat = formul_mol_feat.squeeze(1)
                fused_feat = mol_struct_feat + formul_mol_feat + formul_feat.squeeze(1)
            else:
                fused_feat = mol_struct_feat + formul_feat.squeeze(1)
            fused_feat = self.norm3(fused_feat)
        else:
            fused_feat = mol_struct_feat
            formul_mol_attn = None
        
        # 5. Add physicochemical features (理化描述符增强)
        if physchem_features is not None:
            physchem_feat = self.physchem_proj(physchem_features)
            # Concatenate fused features with physchem descriptors
            fused_feat = torch.cat([fused_feat, physchem_feat], dim=-1)
            # Project to correct dimension for MoE
            fused_feat = self.concat_proj(fused_feat)
            fused_feat = self.norm4(fused_feat)
        else:
            # Fallback: concatenate with molecular features
            fused_feat = torch.cat([fused_feat, mol_feat], dim=-1)
            fused_feat = self.norm4(fused_feat)
        
        # 6. Mixture of Experts integration (多专家协同优化)
        moe_output, expert_weights = self.moe(fused_feat)
        
        # 7. Final projection - concatenate fused_feat and moe_output
        final_feat = torch.cat([fused_feat, moe_output], dim=-1)
        final_feat = self.fusion_proj(final_feat)
        final_feat = self.norm3(final_feat)
        final_feat = self.fusion_dropout(final_feat)
        
        # Collect attention weights for interpretability
        attention_weights = {
            'mol_struct': mol_struct_attn,
            'formul_mol': formul_mol_attn,
        }
        
        return {
            'fused_features': final_feat,
            'attention_weights': attention_weights,
            'expert_weights': expert_weights,
        }


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion with learnable weighting.
    
    Alternative to Cross-Attention, uses learnable parameters
    to weight different modalities.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim * 2
        self.num_modalities = len(feature_dims)
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, output_dim),
            )
            for dim in feature_dims
        ])
        
        # Learnable weights for each modality
        self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * self.num_modalities, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, output_dim),
        )
    
    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Args:
            features: List of (batch, feat_dim) tensors
        
        Returns:
            fused: (batch, output_dim)
        """
        assert len(features) == self.num_modalities
        
        # Project each modality
        projected = []
        for i, feat in enumerate(features):
            proj_feat = self.modality_projections[i](feat)
            weighted_feat = proj_feat * self.modality_weights[i]
            projected.append(weighted_feat)
        
        # Concatenate and fuse
        concatenated = torch.cat(projected, dim=-1)
        fused = self.fusion(concatenated)
        
        return fused
