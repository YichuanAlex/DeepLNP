"""
Transformer Encoder with Pair Attention for LNP formulation modeling.

This module is adapted from TransMA (https://github.com/DP Technology/Uni-Mol)
and optimized for LNP formulation encoding.

Key innovations:
1. Handles multi-component formulations (ionizable lipid + helper + cholesterol + PEG)
2. Incorporates molar ratio embeddings
3. Supports 3D structural information
"""

from typing import Optional, Tuple, Dict, Any

import math
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


class MultiheadAttention(nn.Module):
    """Multihead Self-Attention"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.scaling = self.head_dim ** -0.5
    
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
            query: (seq_len, batch, embed_dim)
            key: (seq_len, batch, embed_dim)
            value: (seq_len, batch, embed_dim)
            key_padding_mask: (batch, seq_len) - bool tensor, True for padding
            attn_mask: (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len)
        
        Returns:
            attn_output: (seq_len, batch, embed_dim)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        bsz, tgt_len, embed_dim = query.size()
        
        # Project to Q, K, V
        q = self.q_proj(query).transpose(0, 1)  # (batch, seq_len, embed_dim)
        k = self.k_proj(key).transpose(0, 1)
        v = self.v_proj(value).transpose(0, 1)
        
        # Reshape for multihead attention: (batch, num_heads, seq_len, head_dim)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (batch, num_heads, tgt_len, src_len)
        
        # Apply attention mask
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, tgt_len, head_dim)
        
        # Reshape back to (batch, tgt_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with optional pair attention"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_heads = num_heads
        self.post_ln = post_ln
        
        # Self-attention
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        
        # Feed-forward network
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
        # Normalization
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        
        # Activation function
        if activation_fn == "gelu":
            self.activation_fn = F.gelu
        elif activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "silu":
            self.activation_fn = F.silu
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")
    
    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attn_mask: (batch, seq_len, seq_len) or (seq_len, seq_len)
            padding_mask: (batch, seq_len) - bool tensor
            return_attn: whether to return attention weights
        
        Returns:
            x: (batch, seq_len, embed_dim)
            attn_weights: (batch, num_heads, seq_len, seq_len) if return_attn
        """
        residual = x
        
        # Pre-LN or Post-LN
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        
        # Self-attention
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
        )
        
        x = self.dropout(x)
        x = residual + x
        
        # Post-LN
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        
        # Feed-forward
        residual = x
        
        if not self.post_ln:
            x = self.final_layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        
        if self.post_ln:
            x = self.final_layer_norm(x)
        
        if return_attn:
            return x, attn_weights
        return x, None


class TransformerEncoderWithPair(nn.Module):
    """
    Transformer Encoder with Pair Attention for molecular and formulation encoding.
    
    This is the core architecture used in TransMA and adapted for DeepLNP.
    It can handle:
    1. Molecular graphs (atoms as tokens)
    2. LNP formulations (components as tokens)
    3. 3D structural information (via distance-based attention bias)
    """
    
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 512,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        
        # Layer normalization
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None
        
        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=self.embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                num_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                post_ln=post_ln,
            )
            for _ in range(encoder_layers)
        ])
    
    def forward(
        self,
        emb: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            emb: (batch, seq_len, embed_dim) - Input embeddings
            attn_mask: (batch, seq_len, seq_len) or (seq_len, seq_len) - Attention bias
            padding_mask: (batch, seq_len) - bool tensor, True for padding
        
        Returns:
            encoder_rep: (batch, seq_len, embed_dim) - Token representations
            attn_mask: (batch, seq_len, seq_len, num_heads) - Final attention map
            delta_pair_repr: (batch, seq_len, seq_len, num_heads) - Pair representation
            x_norm: scalar - Representation norm loss
            delta_pair_repr_norm: scalar - Pair representation norm loss
        """
        bsz = emb.size(0)
        seq_len = emb.size(1)
        
        # Embedding layer norm and dropout
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        
        # Apply padding mask
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask
        
        # Fill attention mask with padding
        def fill_attn_mask(
            attn_mask: Optional[Tensor],
            padding_mask: Optional[Tensor],
            fill_val: float = float("-inf")
        ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
            if attn_mask is not None and padding_mask is not None:
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask
        
        assert attn_mask is not None, "Attention mask is required"
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
        
        # Pass through encoder layers
        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x,
                padding_mask=padding_mask,
                attn_bias=attn_mask,
                return_attn=True,
            )
        
        # Norm loss for stability
        def norm_loss(x: Tensor, eps: float = 1e-10, tolerance: float = 1.0) -> Tensor:
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error
        
        def masked_mean(mask: Tensor, value: Tensor, dim: int = -1, eps: float = 1e-10) -> Tensor:
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()
        
        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)
        
        # Final layer norm
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        
        # Pair representation
        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )
        
        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
        
        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
