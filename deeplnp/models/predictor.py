"""
LNP Predictor - Multi-task prediction head for LNP formulation properties.

This module implements the main prediction architecture for:
1. Transfection efficiency
2. Particle size
3. Zeta potential
4. Toxicity

Key innovations:
1. Multi-task learning with shared representations
2. Uncertainty estimation via MC Dropout
3. Property-constrained prediction (pKa, LogP, etc.)
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_encoder import GraphEncoder
from .transformer_encoder import TransformerEncoderWithPair
from .multimodal_fusion import MultiModalFusion


class PropertyPredictionHead(nn.Module):
    """
    Prediction head for a single property.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class GaussianRBFEncoder(nn.Module):
    """Smooth continuous encoding used for formulation fractions."""

    def __init__(self, output_dim: int, num_basis: int = 16) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, 1.0, num_basis))
        self.log_width = nn.Parameter(torch.tensor(-2.5))
        self.proj = nn.Sequential(
            nn.Linear(num_basis, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )

    def forward(self, value: Tensor) -> Tensor:
        width = self.log_width.exp().clamp(min=1e-3)
        basis = torch.exp(-0.5 * ((value - self.centers) / width).pow(2))
        return self.proj(basis)


class PairwiseBiasedAttentionBlock(nn.Module):
    """Self-attention whose logits contain learned component-pair chemistry."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("feature dimension must be divisible by attention heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, pair_bias: Tensor, active_mask: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, token_count, dim = x.shape
        z = self.norm1(x)
        qkv = self.qkv(z).reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) / self.head_dim ** 0.5
        logits = logits + pair_bias.permute(0, 3, 1, 2)
        logits = logits.masked_fill(~active_mask[:, None, None, :], -1e4)
        attention = logits.softmax(dim=-1)
        attended = torch.einsum("bhij,bjhd->bihd", attention, v).reshape(batch_size, token_count, dim)
        x = x + self.dropout(self.proj(attended))
        x = x + self.ffn(self.norm2(x))
        return x, attention


class ComponentInteractionTransformer(nn.Module):
    """DeepLNP-SPACE: pair-aware four-component and target-conditioned encoder."""

    def __init__(
        self,
        feature_dim: int,
        spatial_dim: int,
        context_dim: int,
        target_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int = 2,
        use_pair_bias: bool = True,
        use_target_conditioning: bool = True,
        use_spatial_features: bool = True,
        use_ratio_features: bool = True,
        use_gaussian_ratio: bool = False,
        use_structure_mask_features: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.spatial_dim = spatial_dim
        self.use_pair_bias = use_pair_bias
        self.use_target_conditioning = use_target_conditioning
        self.use_spatial_features = use_spatial_features
        self.use_ratio_features = use_ratio_features
        self.use_structure_mask_features = use_structure_mask_features
        self.role_embedding = nn.Embedding(4, feature_dim)
        self.ratio_encoder = (
            GaussianRBFEncoder(feature_dim)
            if use_gaussian_ratio else
            nn.Sequential(nn.Linear(1, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        )
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU()
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU()
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU()
        )
        self.missing_embedding = nn.Parameter(torch.zeros(4, feature_dim))
        pair_input_dim = spatial_dim * 3 + 4
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_input_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, num_heads)
        )
        self.blocks = nn.ModuleList([
            PairwiseBiasedAttentionBlock(feature_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.pool_query = nn.Parameter(torch.zeros(feature_dim))
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        component_features: Tensor,
        molar_ratios: Optional[Tensor],
        spatial_features: Optional[Tensor],
        context_features: Optional[Tensor],
        target_features: Optional[Tensor],
        structure_mask: Optional[Tensor],
        active_mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        batch_size = component_features.size(0)
        roles = self.role_embedding(torch.arange(4, device=component_features.device)).unsqueeze(0)
        if molar_ratios is None:
            ratios = component_features.new_full((batch_size, 4), 0.25)
        else:
            ratios = molar_ratios.to(component_features.dtype).clamp(min=0)
            ratios = ratios / ratios.sum(dim=1, keepdim=True).clamp(min=1e-8)
        if structure_mask is None:
            structure_mask = torch.ones(batch_size, 4, device=component_features.device, dtype=torch.bool)
        else:
            structure_mask = structure_mask.bool()
        if active_mask is None:
            active_mask = ratios.gt(0)
        else:
            active_mask = active_mask.bool()
        # Attention requires at least one key; ionizable is the safe fallback.
        empty = ~active_mask.any(dim=1)
        if empty.any():
            active_mask = active_mask.clone()
            active_mask[empty, 0] = True
        tokens = component_features + roles
        if self.use_ratio_features:
            tokens = tokens + self.ratio_encoder(ratios.unsqueeze(-1))
        if self.use_spatial_features and spatial_features is not None:
            tokens = tokens + self.spatial_encoder(spatial_features.to(tokens.dtype))
        if self.use_structure_mask_features:
            tokens = tokens + (~structure_mask).to(tokens.dtype).unsqueeze(-1) * self.missing_embedding.unsqueeze(0)

        if spatial_features is None or not self.use_spatial_features:
            spatial_features = tokens.new_zeros(batch_size, 4, self.spatial_dim)
        s_i = spatial_features[:, :, None, :].expand(-1, -1, 4, -1)
        s_j = spatial_features[:, None, :, :].expand(-1, 4, -1, -1)
        pair_ratios = ratios if self.use_ratio_features else torch.zeros_like(ratios)
        r_i = pair_ratios[:, :, None, None].expand(-1, -1, 4, -1)
        r_j = pair_ratios[:, None, :, None].expand(-1, 4, -1, -1)
        pair_mask = structure_mask if self.use_structure_mask_features else torch.ones_like(structure_mask)
        m_i = pair_mask[:, :, None, None].expand(-1, -1, 4, -1).to(tokens.dtype)
        m_j = pair_mask[:, None, :, None].expand(-1, 4, -1, -1).to(tokens.dtype)
        pair = torch.cat([s_i, s_j, (s_i - s_j).abs(), r_i, r_j, m_i, m_j], dim=-1)
        bias = self.pair_bias(pair) if self.use_pair_bias else pair.new_zeros(
            batch_size, 4, 4, self.blocks[0].num_heads
        )
        attention_maps = []
        for block in self.blocks:
            tokens, attention = block(tokens, bias, active_mask)
            attention_maps.append(attention)

        pool_scores = torch.einsum("bnd,d->bn", tokens, self.pool_query)
        pool_scores = pool_scores.masked_fill(~active_mask, -1e4)
        unconditional = torch.einsum("bn,bnd->bd", pool_scores.softmax(-1), tokens)
        if context_features is None:
            context = component_features.new_zeros(batch_size, self.context_dim)
        else:
            context = context_features.to(component_features.dtype)
        if target_features is None or not self.use_target_conditioning:
            target_features = component_features.new_zeros(batch_size, self.target_encoder[0].in_features)
        query = self.context_encoder(context) + self.target_encoder(target_features.to(tokens.dtype))
        cross, cross_weights = self.cross_attention(
            query.unsqueeze(1), tokens, tokens, key_padding_mask=~active_mask,
            need_weights=True, average_attn_weights=False,
        )
        conditional = self.output_norm(unconditional + query + cross.squeeze(1))
        return {
            "conditional": conditional,
            "unconditional": self.output_norm(unconditional),
            "component_tokens": tokens,
            "pair_attention": torch.stack(attention_maps, dim=1),
            "target_attention": cross_weights,
        }


class LNPPredictor(nn.Module):
    """
    DeepLNP Predictor - 多任务模型用于完整 LNP 配方预测
    
    核心创新：
    1. 四组分配方编码（可电离脂质 + 辅助脂质 + 胆固醇 + PEG 脂质）
    2. 摩尔比感知融合机制
    3. 3D 构象感知编码器
    4. 多任务学习（转染效率、粒径、Zeta 电位、毒性）
    
    输入模态：
    - 四种脂质的 SMILES 和分子图
    - 摩尔比 [ionizable, helper, cholesterol, PEG]
    - 可选：3D 结构、图像、嵌入
    
    输出：
    - transfection_efficiency: 转染效率
    - particle_size: 粒径 (nm)
    - zeta_potential: Zeta 电位 (mV)
    - toxicity: 细胞毒性
    """
    
    def __init__(
        self,
        # Molecular encoder settings
        mol_encoder_type: str = "graph",  # "graph" or "transformer"
        mol_feat_dim: int = 512,
        atom_feat_dim: int = 39,
        bond_feat_dim: int = 4,
        num_gnn_layers: int = 6,
        # Transformer settings
        transformer_layers: int = 6,
        transformer_embed_dim: int = 768,
        transformer_heads: int = 8,
        # 3D encoder settings
        use_3d: bool = True,
        struct_feat_dim: int = 768,
        # Formulation settings - 核心修改：支持四组分
        formul_feat_dim: int = 256,
        num_components: int = 4,  # ionizable, helper, cholesterol, PEG
        formulation_input_dim: int = 7,
        context_feat_dim: int = 39,
        target_feat_dim: int = 11,
        num_target_classes: int = 11,
        spatial_feat_dim: int = 12,
        use_component_transformer: bool = True,
        use_pair_bias: bool = True,
        use_target_conditioning: bool = True,
        use_spatial_features: bool = True,
        use_ratio_features: bool = True,
        use_gaussian_ratio: bool = False,
        use_structure_mask_features: bool = True,
        formulation_noise_std: float = 0.0,
        use_explicit_features: bool = False,
        fingerprint_input_dim: int = 2048,
        explicit_fusion_mode: str = "residual",
        explicit_gate_init: float = -4.0,
        use_direct_morgan_head: bool = False,
        direct_morgan_mode: str = "replace",
        direct_morgan_gate_init: float = -1.0,
        use_separate_rank_head: bool = False,
        rank_head_mode: str = "separate",
        rank_detach_backbone: bool = False,
        rank_gate_init: float = -1.0,
        # Physicochemical settings
        physchem_feat_dim: int = 128,
        # Image encoder settings
        use_images: bool = True,
        image_feat_dim: int = 512,
        image_channels: int = 3,
        image_size: int = 224,
        # Embedding encoder settings
        use_embeddings: bool = True,
        embedding_feat_dim: int = 512,
        max_embedding_dim: int = 2048,
        # Fusion settings
        fusion_hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_experts: int = 6,  # MoE 专家数量
        # Prediction head settings
        prediction_hidden_dim: int = 256,
        num_tasks: int = 4,
        # MC Dropout for uncertainty
        use_mc_dropout: bool = True,
        mc_dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.mol_encoder_type = mol_encoder_type
        self.use_3d = use_3d
        self.num_tasks = num_tasks
        self.use_mc_dropout = use_mc_dropout
        self.mol_feat_dim = mol_feat_dim
        self.num_components = num_components  # 四组分
        self.mol_feat_dim = mol_feat_dim
        self.atom_feat_dim = atom_feat_dim
        self.context_feat_dim = context_feat_dim
        self.target_feat_dim = target_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        self.use_component_transformer = use_component_transformer
        self.formulation_noise_std = float(formulation_noise_std)
        self.use_explicit_features = bool(use_explicit_features)
        self.explicit_fusion_mode = str(explicit_fusion_mode).lower()
        self.use_direct_morgan_head = bool(use_direct_morgan_head)
        self.direct_morgan_mode = str(direct_morgan_mode).lower()
        self.use_separate_rank_head = bool(use_separate_rank_head)
        self.rank_head_mode = str(rank_head_mode).lower()
        if self.rank_head_mode not in {"separate", "residual"}:
            raise ValueError("rank_head_mode must be 'separate' or 'residual'")
        self.rank_detach_backbone = bool(rank_detach_backbone)
        if self.explicit_fusion_mode not in {"residual", "replace"}:
            raise ValueError(
                "explicit_fusion_mode must be either 'residual' or 'replace', "
                f"got {explicit_fusion_mode!r}"
            )
        if self.direct_morgan_mode not in {
            "residual", "replace", "rank_only", "rank_residual"
        }:
            raise ValueError(
                "direct_morgan_mode must be 'residual', 'replace', "
                "'rank_only' or 'rank_residual'"
            )
        if self.direct_morgan_mode == "rank_residual" and not self.use_separate_rank_head:
            raise ValueError("rank_residual requires use_separate_rank_head=True")
        
        # ========== 核心修改：四组分分子编码器 ==========
        # 1. 可电离脂质编码器（主要功能组分）
        if mol_encoder_type == "graph":
            self.component_encoder = GraphEncoder(
                atom_feature_dim=atom_feat_dim,
                bond_feature_dim=bond_feat_dim,
                hidden_dim=mol_feat_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
            )
            # The four roles deliberately share this encoder; role identity is
            # introduced by ComponentInteractionTransformer embeddings.
            self.fingerprint_encoder = nn.Sequential(
                nn.Linear(fingerprint_input_dim, mol_feat_dim * 2),
                nn.LayerNorm(mol_feat_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mol_feat_dim * 2, mol_feat_dim),
            ) if self.use_explicit_features else None
            # Start as an almost exact graph-only model.  The scalar gate can
            # grow only if validation-supported gradients find the explicit
            # count fingerprint useful, avoiding a destructive cold-start.
            self.explicit_gate_logit = (
                nn.Parameter(torch.tensor(float(explicit_gate_init)))
                if self.use_explicit_features and self.explicit_fusion_mode == "residual"
                else None
            )
            direct_input_dim = (
                fingerprint_input_dim + spatial_feat_dim + formulation_input_dim
                + context_feat_dim + target_feat_dim + 2 * num_components
            )
            self.direct_morgan_encoder = nn.Sequential(
                nn.Linear(direct_input_dim, prediction_hidden_dim),
                nn.LayerNorm(prediction_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(prediction_hidden_dim, prediction_hidden_dim // 2),
                nn.GELU(),
            ) if self.use_direct_morgan_head else None
            self.direct_morgan_point = (
                nn.Linear(prediction_hidden_dim // 2, 1)
                if self.use_direct_morgan_head else None
            )
            self.direct_morgan_rank = (
                nn.Linear(prediction_hidden_dim // 2, 1)
                if self.use_direct_morgan_head else None
            )
            self.direct_morgan_gate_logit = (
                nn.Parameter(torch.tensor(float(direct_morgan_gate_init)))
                if self.use_direct_morgan_head
                and self.direct_morgan_mode in {"residual", "rank_residual"}
                else None
            )
            self.component_interaction = ComponentInteractionTransformer(
                feature_dim=mol_feat_dim,
                spatial_dim=spatial_feat_dim,
                context_dim=context_feat_dim,
                target_dim=target_feat_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_pair_bias=use_pair_bias,
                use_target_conditioning=use_target_conditioning,
                use_spatial_features=use_spatial_features,
                use_ratio_features=use_ratio_features,
                use_gaussian_ratio=use_gaussian_ratio,
                use_structure_mask_features=use_structure_mask_features,
            ) if use_component_transformer else None
        elif mol_encoder_type == "transformer":
            # Transformer 版本（共用编码器权重）
            self.mol_encoder = TransformerEncoderWithPair(
                encoder_layers=transformer_layers,
                embed_dim=transformer_embed_dim,
                ffn_embed_dim=transformer_embed_dim * 4,
                attention_heads=transformer_heads,
                dropout=dropout,
                attention_dropout=dropout,
            )
            mol_feat_dim = transformer_embed_dim
        else:
            raise ValueError(f"Unknown encoder type: {mol_encoder_type}")
        
        # 2. 3D Structure Encoder (optional)
        if use_3d:
            self.struct_encoder = TransformerEncoderWithPair(
                encoder_layers=6,
                embed_dim=struct_feat_dim,
                ffn_embed_dim=struct_feat_dim * 4,
                attention_heads=8,
                dropout=dropout,
            )
        else:
            self.struct_encoder = None
        
        # 3. Image Encoder (CNN backbone)
        if use_images:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, image_feat_dim),
                nn.LayerNorm(image_feat_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.image_encoder = None
        
        # 4. Embedding Encoder
        if use_embeddings:
            self.embedding_encoder = nn.Sequential(
                nn.Linear(max_embedding_dim, embedding_feat_dim * 2),
                nn.LayerNorm(embedding_feat_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_feat_dim * 2, embedding_feat_dim),
                nn.LayerNorm(embedding_feat_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.embedding_encoder = None
        
        # 后备投影层：当没有 atom_features 时使用 fingerprint 或原子特征均值
        self.fallback_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, mol_feat_dim),
            nn.LayerNorm(mol_feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 5. Formulation Encoder
        self.formul_encoder = nn.Sequential(
            nn.Linear(formulation_input_dim, formul_feat_dim),
            nn.LayerNorm(formul_feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 6. Physicochemical Encoder
        self.physchem_encoder = nn.Sequential(
            nn.Linear(physchem_feat_dim, physchem_feat_dim),
            nn.LayerNorm(physchem_feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 7. Multi-modal Fusion - 引入 3D 构象感知和 Cross-Attention
        self.fusion = MultiModalFusion(
            mol_feat_dim=mol_feat_dim,
            struct_feat_dim=struct_feat_dim if use_3d else None,
            formul_feat_dim=formul_feat_dim,
            physchem_feat_dim=physchem_feat_dim,
            image_feat_dim=image_feat_dim if use_images else None,
            embedding_feat_dim=embedding_feat_dim if use_embeddings else None,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_experts=num_experts,
            use_cross_attention=True,  # 启用 Cross-Attention
            use_3d_conformer=use_3d,  # 启用 3D 构象感知
        )
        
        # 6. Multi-task Prediction Heads
        # Support flexible task configuration
        all_tasks = ['efficiency', 'particle_size', 'zeta_potential', 'pdi', 'encapsulation', 'toxicity']
        task_names = all_tasks[:num_tasks] if num_tasks <= len(all_tasks) else all_tasks
        
        self.task_heads = nn.ModuleDict({
            task_name: PropertyPredictionHead(
                input_dim=fusion_hidden_dim,
                hidden_dim=prediction_hidden_dim,
                output_dim=1,
                dropout=dropout,
            )
            for task_name in task_names
        })
        self.uncertainty_heads = nn.ModuleDict({
            task_name: nn.Linear(fusion_hidden_dim, 1) for task_name in task_names
        })
        self.efficiency_rank_head = (
            PropertyPredictionHead(
                input_dim=fusion_hidden_dim,
                hidden_dim=prediction_hidden_dim,
                output_dim=1,
                dropout=dropout,
            )
            if self.use_separate_rank_head and "efficiency" in task_names else None
        )
        self.rank_gate_logit = (
            nn.Parameter(torch.tensor(float(rank_gate_init)))
            if self.efficiency_rank_head is not None and self.rank_head_mode == "residual"
            else None
        )
        self.target_head = nn.Sequential(
            nn.Linear(mol_feat_dim, prediction_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, num_target_classes),
        )
        
        # Property constraint heads (for generation guidance)
        self.property_constraints = nn.ModuleDict({
            'pka': nn.Linear(fusion_hidden_dim, 1),
            'logp': nn.Linear(fusion_hidden_dim, 1),
            'molecular_weight': nn.Linear(fusion_hidden_dim, 1),
            'biodegradability': nn.Linear(fusion_hidden_dim, 1),
        })
        
        # Dropout for MC Dropout uncertainty estimation
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights (only for newly created layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with proper scaling - only for newly created layers."""
        # Only initialize the task heads and constraint heads, not the encoders
        # This prevents destroying pre-trained encoder weights
        for task_name, head in self.task_heads.items():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if self.efficiency_rank_head is not None:
            for module in self.efficiency_rank_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        for prop_name, head in self.property_constraints.items():
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
    
    def forward(
        self,
        # ========== 核心修改：四组分配方输入 ==========
        # 1. 可电离脂质（主要功能组分）
        ionizable_atom_features: Optional[Tensor] = None,
        ionizable_edge_index: Optional[Tensor] = None,
        ionizable_bond_features: Optional[Tensor] = None,
        ionizable_atom_mask: Optional[Tensor] = None,
        ionizable_batch: Optional[Tensor] = None,  # batch 向量
        
        # 2. 辅助脂质
        helper_atom_features: Optional[Tensor] = None,
        helper_edge_index: Optional[Tensor] = None,
        helper_bond_features: Optional[Tensor] = None,
        helper_atom_mask: Optional[Tensor] = None,
        helper_batch: Optional[Tensor] = None,  # batch 向量
        
        # 3. 胆固醇
        cholesterol_atom_features: Optional[Tensor] = None,
        cholesterol_edge_index: Optional[Tensor] = None,
        cholesterol_bond_features: Optional[Tensor] = None,
        cholesterol_atom_mask: Optional[Tensor] = None,
        cholesterol_batch: Optional[Tensor] = None,  # batch 向量
        
        # 4. PEG 脂质
        peg_atom_features: Optional[Tensor] = None,
        peg_edge_index: Optional[Tensor] = None,
        peg_bond_features: Optional[Tensor] = None,
        peg_atom_mask: Optional[Tensor] = None,
        peg_batch: Optional[Tensor] = None,  # batch 向量

        ionizable_spatial_features: Optional[Tensor] = None,
        helper_spatial_features: Optional[Tensor] = None,
        cholesterol_spatial_features: Optional[Tensor] = None,
        peg_spatial_features: Optional[Tensor] = None,
        ionizable_fingerprint: Optional[Tensor] = None,
        helper_fingerprint: Optional[Tensor] = None,
        cholesterol_fingerprint: Optional[Tensor] = None,
        peg_fingerprint: Optional[Tensor] = None,
        
        # 摩尔比（核心配方参数）
        molar_ratios: Optional[Tensor] = None,  # (batch, 4)
        
        # ========== 旧版兼容输入（向后兼容） ==========
        atom_features: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
        bond_features: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        
        # 3D structure input
        src_tokens: Optional[Tensor] = None,
        src_distance: Optional[Tensor] = None,
        src_coord: Optional[Tensor] = None,
        src_edge_type: Optional[Tensor] = None,
        # Image input
        images: Optional[Tensor] = None,
        # Embedding input
        embeddings: Optional[Tensor] = None,
        # Formulation input
        formulation_features: Optional[Tensor] = None,
        context_features: Optional[Tensor] = None,
        target_features: Optional[Tensor] = None,
        component_structure_mask: Optional[Tensor] = None,
        component_active_mask: Optional[Tensor] = None,
        # Physicochemical input
        physchem_features: Optional[Tensor] = None,
        # Return intermediate representations
        return_features: bool = False,
        # MC Dropout sampling
        num_samples: int = 1,
    ) -> Dict[str, Tensor]:
        """
        前向传播 - 支持四组分配方
        
        Args:
            # 四组分配方
            ionizable_atom_features: 可电离脂质原子特征
            ionizable_edge_index: 可电离脂质边索引
            ionizable_bond_features: 可电离脂质键特征
            helper_atom_features: 辅助脂质原子特征
            helper_edge_index: 辅助脂质边索引
            helper_bond_features: 辅助脂质键特征
            cholesterol_atom_features: 胆固醇原子特征
            cholesterol_edge_index: 胆固醇边索引
            cholesterol_bond_features: 胆固醇键特征
            peg_atom_features: PEG 脂质原子特征
            peg_edge_index: PEG 脂质边索引
            peg_bond_features: PEG 脂质键特征
            molar_ratios: 摩尔比 (batch, 4) - [ionizable, helper, cholesterol, PEG]
            
            # 旧版兼容
            atom_features, edge_index, etc: 向后兼容
            
            return_features: 是否返回融合特征
            num_samples: MC Dropout 采样次数
        
        Returns:
            predictions: Dict with keys ['efficiency', 'particle_size', 'zeta_potential', 'toxicity']
            features: (batch, fusion_hidden_dim) - if return_features
            uncertainty: Dict with uncertainty estimates - if num_samples > 1
        """
        # ========== 1. 编码四组分分子特征 ==========
        unconditional_mol_features = None
        component_attention = None
        if self.training and self.formulation_noise_std > 0 and molar_ratios is not None:
            ratio_noise = 1.0 + torch.randn_like(molar_ratios) * self.formulation_noise_std
            molar_ratios = (molar_ratios * ratio_noise).clamp(min=0.0)
            molar_ratios = molar_ratios / molar_ratios.sum(dim=1, keepdim=True).clamp(min=1e-8)
        if self.mol_encoder_type == "graph":
            # 编码可电离脂质（必需）
            if ionizable_atom_features is not None and ionizable_edge_index is not None:
                ionizable_out = self.component_encoder(
                    atom_features=ionizable_atom_features,
                    bond_features=ionizable_bond_features,
                    edge_index=ionizable_edge_index,
                    batch=ionizable_batch,  # 使用传入的 batch 向量
                )
                ionizable_features = ionizable_out["graph_embeddings"]
            else:
                # 向后兼容
                if atom_features is not None and edge_index is not None:
                    mol_encoder_output = self.component_encoder(
                        atom_features=atom_features,
                        bond_features=bond_features,
                        edge_index=edge_index,
                        batch=batch,
                    )
                    ionizable_features = mol_encoder_output["graph_embeddings"]
                elif embeddings is not None and self.embedding_encoder is not None:
                    ionizable_features = self.embedding_encoder(embeddings)
                elif atom_features is not None and atom_features.dim() == 3:
                    atom_mean = atom_features.mean(dim=1)
                    ionizable_features = self.fallback_proj(atom_mean)
                else:
                    ionizable_features = None
            
            # 编码辅助脂质（可选）
            if helper_atom_features is not None and helper_edge_index is not None:
                helper_out = self.component_encoder(
                    atom_features=helper_atom_features,
                    bond_features=helper_bond_features,
                    edge_index=helper_edge_index,
                    batch=helper_batch,
                )
                helper_features = helper_out["graph_embeddings"]
            else:
                helper_features = None
            
            # 编码胆固醇（可选）
            if cholesterol_atom_features is not None and cholesterol_edge_index is not None:
                cholesterol_out = self.component_encoder(
                    atom_features=cholesterol_atom_features,
                    bond_features=cholesterol_bond_features,
                    edge_index=cholesterol_edge_index,
                    batch=cholesterol_batch,
                )
                cholesterol_features = cholesterol_out["graph_embeddings"]
            else:
                cholesterol_features = None
            
            # 编码 PEG 脂质（可选）
            if peg_atom_features is not None and peg_edge_index is not None:
                peg_out = self.component_encoder(
                    atom_features=peg_atom_features,
                    bond_features=peg_bond_features,
                    edge_index=peg_edge_index,
                    batch=peg_batch,
                )
                peg_features = peg_out["graph_embeddings"]
            else:
                peg_features = None

            # Fuse learned graph embeddings with log-count Morgan features.
            # A learnable residual gate lets the model ignore the explicit
            # branch when it is unhelpful; the feature is fully ablatable.
            if self.use_explicit_features and self.fingerprint_encoder is not None:
                graph_features = [
                    ionizable_features, helper_features,
                    cholesterol_features, peg_features,
                ]
                fingerprints = [
                    ionizable_fingerprint, helper_fingerprint,
                    cholesterol_fingerprint, peg_fingerprint,
                ]
                fused_components = []
                reference_dtype = ionizable_features.dtype
                for graph_feature, fingerprint in zip(graph_features, fingerprints):
                    if fingerprint is None:
                        fused_components.append(graph_feature)
                        continue
                    explicit = self.fingerprint_encoder(fingerprint.to(dtype=reference_dtype))
                    if graph_feature is None or self.explicit_fusion_mode == "replace":
                        fused_components.append(explicit)
                    else:
                        gate = torch.sigmoid(self.explicit_gate_logit)
                        fused_components.append(graph_feature + gate * explicit)
                ionizable_features, helper_features, cholesterol_features, peg_features = fused_components
            
            # ========== 2. 摩尔比加权融合 ==========
            if ionizable_features is not None:
                component_features = [
                    ionizable_features,
                    helper_features if helper_features is not None else torch.zeros_like(ionizable_features),
                    cholesterol_features if cholesterol_features is not None else torch.zeros_like(ionizable_features),
                    peg_features if peg_features is not None else torch.zeros_like(ionizable_features),
                ]
                all_lipid_features = torch.stack(component_features, dim=1)
                
                spatial_list = [
                    ionizable_spatial_features,
                    helper_spatial_features,
                    cholesterol_spatial_features,
                    peg_spatial_features,
                ]
                spatial_features = None
                if all(value is not None for value in spatial_list):
                    spatial_features = torch.stack(spatial_list, dim=1)

                if self.component_interaction is not None:
                    interaction = self.component_interaction(
                        all_lipid_features,
                        molar_ratios,
                        spatial_features,
                        context_features,
                        target_features,
                        component_structure_mask,
                        component_active_mask,
                    )
                    mol_features = interaction["conditional"]
                    unconditional_mol_features = interaction["unconditional"]
                    component_attention = interaction
                elif molar_ratios is not None:
                    weights = molar_ratios[:, :4].to(all_lipid_features.dtype).clamp(min=0)
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
                    mol_features = (all_lipid_features * weights.unsqueeze(-1)).sum(dim=1)
                else:
                    active = torch.tensor(
                        [
                            1.0,
                            1.0 if helper_features is not None else 0.0,
                            1.0 if cholesterol_features is not None else 0.0,
                            1.0 if peg_features is not None else 0.0,
                        ],
                        device=ionizable_features.device,
                        dtype=all_lipid_features.dtype,
                    ).unsqueeze(0).expand(ionizable_features.size(0), -1)
                    weights = active / (active.sum(dim=1, keepdim=True) + 1e-8)
                    mol_features = (all_lipid_features * weights.unsqueeze(-1)).sum(dim=1)
            else:
                mol_features = None
        else:
            # Transformer 版本
            assert src_tokens is not None
            mol_features = self.mol_encoder(
                emb=src_tokens,
                attn_mask=None,
                padding_mask=None,
            )[0][:, 0, :]
        
        # 确保 mol_features 不为 None（向后兼容）
        if mol_features is None:
            if embeddings is not None and self.embedding_encoder is not None:
                mol_features = self.embedding_encoder(embeddings)
            elif ionizable_atom_features is not None and ionizable_atom_features.dim() == 3:
                atom_mean = ionizable_atom_features.mean(dim=1)
                mol_features = self.fallback_proj(atom_mean)  # (batch, mol_feat_dim)
            else:
                # 创建全零特征作为后备（使用正确的 batch size）
                # 从可用的输入张量中获取 batch size
                batch_size = 1
                if embeddings is not None:
                    batch_size = embeddings.size(0)
                elif formulation_features is not None:
                    batch_size = formulation_features.size(0)
                elif images is not None:
                    batch_size = images.size(0)
                elif ionizable_atom_features is not None:
                    batch_size = ionizable_atom_features.size(0)
                
                feat_dim = getattr(self.component_encoder, 'hidden_dim', self.mol_feat_dim)
                # 尝试从其他张量获取设备信息
                device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
                if embeddings is not None:
                    device = embeddings.device
                elif formulation_features is not None:
                    device = formulation_features.device
                elif images is not None:
                    device = images.device
                elif ionizable_atom_features is not None:
                    device = ionizable_atom_features.device
                mol_features = torch.zeros(batch_size, feat_dim, device=device)
        
        # 2. Encode 3D structural features (if available)
        if self.use_3d and self.struct_encoder is not None:
            if src_tokens is not None and src_distance is not None:
                struct_features = self.struct_encoder(
                    emb=src_tokens,
                    attn_mask=src_distance,
                    padding_mask=None,
                )[0][:, 0, :]  # Use CLS token
            else:
                struct_features = None
        else:
            struct_features = None
        
        # 3. Encode image features (if available)
        if self.image_encoder is not None and images is not None:
            image_features = self.image_encoder(images)
        else:
            image_features = None
        
        # 4. Encode embedding features (if available)
        if self.embedding_encoder is not None and embeddings is not None:
            embedding_features = self.embedding_encoder(embeddings)
        else:
            embedding_features = None
        
        # 5. Encode formulation features
        if formulation_features is not None:
            if self.training and self.formulation_noise_std > 0:
                value_count = formulation_features.size(-1) // 2
                formulation_features = formulation_features.clone()
                noise = 1.0 + torch.randn_like(formulation_features[:, :value_count]) * self.formulation_noise_std
                formulation_features[:, :value_count] = formulation_features[:, :value_count] * noise
            formul_features = self.formul_encoder(formulation_features)
        else:
            formul_features = None
        
        # 6. Encode physicochemical features
        if physchem_features is not None:
            physchem_feats = self.physchem_encoder(physchem_features)
        else:
            physchem_feats = None
        
        # 7. Multi-modal fusion
        fusion_output = self.fusion(
            mol_features=mol_features,
            struct_features=struct_features,
            formulation_features=formul_features,
            physchem_features=physchem_feats,
            image_features=image_features,
            embedding_features=embedding_features,
        )
        
        fused_features = fusion_output['fused_features']
        
        # Only apply dropout during training, not during evaluation
        if self.training:
            fused_features = self.dropout(fused_features)
        
        # Standard prediction
        predictions = self._predict_heads(fused_features)
        if self.direct_morgan_encoder is not None and ionizable_fingerprint is not None:
            batch_size = ionizable_fingerprint.size(0)
            reference = ionizable_fingerprint

            def dense(value: Optional[Tensor], width: int) -> Tensor:
                if value is None:
                    return reference.new_zeros((batch_size, width))
                return value.to(device=reference.device, dtype=reference.dtype)

            direct_input = torch.cat(
                [
                    reference,
                    dense(ionizable_spatial_features, self.spatial_feat_dim),
                    dense(formulation_features, self.formul_encoder[0].in_features),
                    dense(context_features, self.context_feat_dim),
                    dense(target_features, self.target_feat_dim),
                    dense(component_structure_mask, self.num_components),
                    dense(component_active_mask, self.num_components),
                ],
                dim=1,
            )
            direct_features = self.direct_morgan_encoder(direct_input)
            direct_point = self.direct_morgan_point(direct_features).squeeze(-1)
            direct_rank = self.direct_morgan_rank(direct_features).squeeze(-1)
            if self.direct_morgan_mode == "replace":
                predictions["efficiency"] = direct_point
                predictions["efficiency_rank_score"] = direct_rank
            elif self.direct_morgan_mode == "rank_only":
                predictions["efficiency_rank_score"] = direct_rank
            elif self.direct_morgan_mode == "rank_residual":
                rank_features = (
                    fused_features.detach() if self.rank_detach_backbone else fused_features
                )
                rank_delta = self.efficiency_rank_head(rank_features).squeeze(-1)
                graph_rank = (
                    predictions["efficiency"].detach()
                    + torch.sigmoid(self.rank_gate_logit) * rank_delta
                )
                predictions["efficiency_rank_score"] = (
                    graph_rank
                    + torch.sigmoid(self.direct_morgan_gate_logit) * direct_rank
                )
            else:
                gate = torch.sigmoid(self.direct_morgan_gate_logit)
                predictions["efficiency"] = predictions["efficiency"] + gate * direct_point
                predictions["efficiency_rank_score"] = (
                    predictions["efficiency"].detach() + gate * direct_rank
                )
        if self.efficiency_rank_head is not None and not self.use_direct_morgan_head:
            rank_features = fused_features.detach() if self.rank_detach_backbone else fused_features
            rank_delta = self.efficiency_rank_head(rank_features).squeeze(-1)
            if self.rank_head_mode == "residual":
                predictions["efficiency_rank_score"] = (
                    predictions["efficiency"].detach()
                    + torch.sigmoid(self.rank_gate_logit) * rank_delta
                )
            else:
                predictions["efficiency_rank_score"] = rank_delta
        if unconditional_mol_features is None:
            unconditional_mol_features = mol_features
        predictions["target_logits"] = self.target_head(unconditional_mol_features)
        
        if return_features:
            return {
                'predictions': predictions,
                'features': fused_features,
                'component_attention': component_attention,
            }
        else:
            return {
                'predictions': predictions,
            }
    
    def _predict_heads(self, features: Tensor) -> Dict[str, Tensor]:
        """Apply task-specific prediction heads."""
        predictions = {}
        
        # Main tasks
        for task_name, head in self.task_heads.items():
            pred = head(features)
            predictions[task_name] = pred.squeeze(-1)
            predictions[f"{task_name}_log_variance"] = self.uncertainty_heads[task_name](features).squeeze(-1).clamp(-8, 8)
        
        # Property constraints (for generation guidance)
        for prop_name, head in self.property_constraints.items():
            pred = head(features)
            predictions[f'predicted_{prop_name}'] = pred.squeeze(-1)
        
        return predictions
    
    def predict(
        self,
        batch: Dict[str, Tensor],
        return_uncertainty: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Convenience method for prediction with proper batching.
        
        Args:
            batch: Dictionary containing all input features
            return_uncertainty: whether to return uncertainty estimates
        
        Returns:
            predictions: Dict with predictions and optionally uncertainty
        """
        with torch.no_grad():
            output = self(
                ionizable_atom_features=batch.get('ionizable_atom_features', batch.get('atom_features')),
                ionizable_edge_index=batch.get('ionizable_edge_index', batch.get('edge_index')),
                ionizable_bond_features=batch.get('ionizable_bond_features', batch.get('bond_features')),
                ionizable_batch=batch.get('ionizable_batch', batch.get('batch')),
                helper_atom_features=batch.get('helper_atom_features'),
                helper_edge_index=batch.get('helper_edge_index'),
                helper_bond_features=batch.get('helper_bond_features'),
                helper_batch=batch.get('helper_batch'),
                cholesterol_atom_features=batch.get('cholesterol_atom_features'),
                cholesterol_edge_index=batch.get('cholesterol_edge_index'),
                cholesterol_bond_features=batch.get('cholesterol_bond_features'),
                cholesterol_batch=batch.get('cholesterol_batch'),
                peg_atom_features=batch.get('peg_atom_features'),
                peg_edge_index=batch.get('peg_edge_index'),
                peg_bond_features=batch.get('peg_bond_features'),
                peg_batch=batch.get('peg_batch'),
                ionizable_spatial_features=batch.get('ionizable_spatial_features'),
                helper_spatial_features=batch.get('helper_spatial_features'),
                cholesterol_spatial_features=batch.get('cholesterol_spatial_features'),
                peg_spatial_features=batch.get('peg_spatial_features'),
                molar_ratios=batch.get('molar_ratios'),
                atom_features=batch.get('atom_features'),
                edge_index=batch.get('edge_index'),
                edge_features=batch.get('edge_features'),
                bond_features=batch.get('bond_features'),
                batch=batch.get('batch'),
                src_tokens=batch.get('src_tokens'),
                src_distance=batch.get('src_distance'),
                src_coord=batch.get('src_coord'),
                src_edge_type=batch.get('src_edge_type'),
                images=batch.get('images', batch.get('image')),
                embeddings=batch.get('embeddings', batch.get('embedding')),
                formulation_features=batch.get('formulation_features'),
                context_features=batch.get('context_features'),
                target_features=batch.get('target_features'),
                component_structure_mask=batch.get('component_structure_mask'),
                component_active_mask=batch.get('component_active_mask'),
                physchem_features=batch.get('physchem_features'),
                num_samples=10 if return_uncertainty else 1,
            )
        
        return output
    
    def get_attention_weights(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Extract attention weights for interpretability.
        
        Args:
            batch: Dictionary containing all input features
        
        Returns:
            attention_weights: Dict with attention maps
        """
        # Forward pass through molecular encoder
        if self.mol_encoder_type == "graph":
            # For GNN, we need to extract attention from message passing
            # This is more complex and would require modifying the GNN
            mol_features = self.mol_encoder(
                atom_features=batch.get('atom_features'),
                edge_index=batch.get('edge_index'),
                edge_features=batch.get('edge_features'),
            )
            mol_attn = None
        else:
            mol_output = self.mol_encoder(
                emb=batch.get('src_tokens'),
                attn_mask=None,
                padding_mask=None,
            )
            mol_features = mol_output[0][:, 0, :]
            mol_attn = mol_output[1]  # Attention map
        
        # Get fusion attention weights
        fusion_output = self.fusion(
            mol_features=mol_features,
            struct_features=None,
            formulation_features=None,
            physchem_features=None,
        )
        
        return {
            'molecular_attention': mol_attn,
            'fusion_attention': fusion_output['attention_weights'],
            'expert_weights': fusion_output['expert_weights'],
        }


class LNPClassifier(nn.Module):
    """
    Binary/Multi-class classifier for LNP formulations.
    
    Use cases:
    1. High/Low efficiency classification
    2. Tissue targeting (liver/lung/spleen)
    3. Toxicity classification
    """
    
    def __init__(
        self,
        predictor: LNPPredictor,
        num_classes: int = 2,
        freeze_predictor: bool = False,
    ):
        super().__init__()
        
        self.predictor = predictor
        self.num_classes = num_classes
        
        if freeze_predictor:
            for param in self.predictor.parameters():
                param.requires_grad = False
        
        # Classification head
        fusion_dim = predictor.fusion.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, num_classes),
        )
    
    def forward(
        self,
        atom_features: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
        formulation_features: Optional[Tensor] = None,
        physchem_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            atom_features, edge_index, edge_features: Molecular input
            formulation_features: Formulation input
            physchem_features: Physicochemical input
        
        Returns:
            logits: (batch, num_classes)
        """
        # Get fused features
        output = self.predictor(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_features=edge_features,
            formulation_features=formulation_features,
            physchem_features=physchem_features,
            return_features=True,
        )
        
        features = output['features']
        
        # Classification
        logits = self.classifier(features)
        
        return logits
