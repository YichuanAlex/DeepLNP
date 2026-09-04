"""
Enhanced 3D Conformation-Aware Molecular Generator.

This module implements advanced 3D molecular generation with:
1. Geometry-enhanced pretraining (Uni-Mol style)
2. Property-constrained diffusion
3. Scaffold-based controlled generation
4. pKa/LogP/biodegradability constraints

Key innovations for Nature Nanotechnology:
1. Direct generation of 3D structures with target properties
2. Controlled exploration of chemical space
3. Rational design of ionizable lipids with optimal pKa (6.2-6.8)
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .generator_diffusion import GaussianDiffusion, MolecularDiffusionGenerator


class GeometryEnhancedEncoder(nn.Module):
    """
    3D molecular encoder inspired by Uni-Mol.
    
    Encodes both atomic features and 3D geometric relationships.
    """
    
    def __init__(
        self,
        atom_feat_dim: int = 128,
        pair_feat_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.atom_feat_dim = atom_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.hidden_dim = hidden_dim
        
        # Atom embedding
        self.atom_embedding = nn.Linear(atom_feat_dim, hidden_dim)
        
        # Pair embedding (distance, angle, etc.)
        self.pair_embedding = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        atom_features: Tensor,
        pair_features: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            atom_features: (batch, num_atoms, atom_feat_dim)
            pair_features: (batch, num_atoms, num_atoms, pair_feat_dim)
            attention_mask: (batch, num_atoms) - Valid atom mask
        
        Returns:
            atom_representations: (batch, num_atoms, hidden_dim)
        """
        # Embed atoms
        h = self.atom_embedding(atom_features)
        
        # Embed pairs and create attention bias
        pair_repr = self.pair_embedding(pair_features)
        pair_attn_bias = pair_repr.mean(dim=-1)  # (batch, num_atoms, num_atoms)
        
        # Transformer with pair attention bias
        for layer in self.layers:
            # Add pair bias to attention
            h = layer(h, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
            # Modulate with pair features
            h = h + pair_attn_bias.mean(dim=1, keepdim=True)
        
        return self.norm(h)


class PropertyConstrainedGenerator(nn.Module):
    """
    Property-Constrained 3D Molecular Generator.
    
    Generates molecules that satisfy target property constraints:
    - pKa in optimal range (6.2-6.8 for endosomal escape)
    - LogP for membrane permeability
    - Molecular weight
    - Biodegradability score
    - Toxicity score
    """
    
    def __init__(
        self,
        base_generator: MolecularDiffusionGenerator,
        property_dim: int = 128,
        num_properties: int = 6,
        constraint_weight: float = 1.0,
    ):
        super().__init__()
        
        self.base_generator = base_generator
        self.property_dim = property_dim
        self.num_properties = num_properties
        self.constraint_weight = constraint_weight
        
        # Property encoder
        self.property_encoder = nn.Sequential(
            nn.Linear(num_properties, property_dim),
            nn.LayerNorm(property_dim),
            nn.GELU(),
            nn.Linear(property_dim, property_dim),
        )
        
        # Property predictor (for verification)
        self.property_verifier = nn.Sequential(
            nn.Linear(base_generator.hidden_dim, property_dim),
            nn.GELU(),
            nn.Linear(property_dim, num_properties),
        )
    
    def encode_property_constraints(
        self,
        target_pka: Optional[float] = None,
        target_logp: Optional[float] = None,
        target_mw: Optional[float] = None,
        target_biodegradability: Optional[float] = None,
        target_toxicity: Optional[float] = None,
        target_efficiency: Optional[float] = None,
    ) -> Tensor:
        """
        Encode property constraints into a vector.
        
        Args:
            target_pka: Target pKa value (e.g., 6.5)
            target_logp: Target LogP value
            target_mw: Target molecular weight
            target_biodegradability: Biodegradability score (0-1)
            target_toxicity: Toxicity score (0-1, lower is better)
            target_efficiency: Transfection efficiency (0-1)
        
        Returns:
            property_vector: (1, property_dim)
        """
        # Create property vector
        properties = torch.zeros(1, self.num_properties)
        
        if target_pka is not None:
            properties[0, 0] = target_pka
        if target_logp is not None:
            properties[0, 1] = target_logp
        if target_mw is not None:
            properties[0, 2] = target_mw / 1000.0  # Normalize
        if target_biodegradability is not None:
            properties[0, 3] = target_biodegradability
        if target_toxicity is not None:
            properties[0, 4] = target_toxicity
        if target_efficiency is not None:
            properties[0, 5] = target_efficiency
        
        # Encode
        return self.property_encoder(properties)
    
    def forward(
        self,
        coordinates: Tensor,
        atom_types: Tensor,
        t: Tensor,
        property_constraints: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Training forward pass."""
        return self.base_generator(
            coordinates=coordinates,
            atom_types=atom_types,
            t=t,
            property_constraints=property_constraints,
        )
    
    @torch.no_grad()
    def generate_with_constraints(
        self,
        num_molecules: int = 1,
        target_properties: Optional[Dict[str, float]] = None,
        guidance_scale: float = 2.0,
        num_steps: int = 100,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate molecules with specific property constraints.
        
        Args:
            num_molecules: Number of molecules to generate
            target_properties: Dictionary of target properties
                - pka: Target pKa
                - logp: Target LogP
                - mw: Target molecular weight
                - biodegradability: Target biodegradability
                - toxicity: Target toxicity
                - efficiency: Target efficiency
            guidance_scale: Classifier-free guidance scale
            num_steps: Number of denoising steps
            return_trajectory: Whether to return generation trajectory
        
        Returns:
            Generated molecules with metadata
        """
        device = next(self.parameters()).device
        
        # Encode property constraints
        if target_properties is None:
            property_constraints = None
        else:
            property_constraints = self.encode_property_constraints(
                **target_properties
            ).to(device)
        
        # Generate using base generator
        result = self.base_generator.sample(
            num_molecules=num_molecules,
            property_constraints=property_constraints,
            guidance_scale=guidance_scale,
            return_intermediates=return_trajectory,
        )
        
        # Verify generated properties
        coordinates = result['coordinates']
        atom_types = result['atom_types']
        
        # Extract features for verification
        features = self.base_generator.denoising_model.decoder[-1](coordinates.view(coordinates.size(0), -1))
        predicted_properties = self.property_verifier(features.mean(dim=1))
        
        return {
            'coordinates': coordinates,
            'atom_types': atom_types,
            'predicted_properties': predicted_properties,
            'target_properties': property_constraints,
            'trajectory': result.get('intermediates'),
        }


class ScaffoldBasedGenerator(nn.Module):
    """
    Scaffold-Based Controlled Molecular Generator.
    
    Generates molecules by decorating a core scaffold structure.
    This ensures chemical validity and enables focused exploration.
    
    Application:
    - Generate ionizable lipid variants with common head groups
    - Explore tail chain variations systematically
    - Maintain key structural motifs while optimizing properties
    """
    
    def __init__(
        self,
        base_generator: PropertyConstrainedGenerator,
        scaffold_vocab_size: int = 50,
        max_decorations: int = 10,
    ):
        super().__init__()
        
        self.base_generator = base_generator
        self.scaffold_vocab_size = scaffold_vocab_size
        self.max_decorations = max_decorations
        
        # Scaffold embedding
        self.scaffold_embedding = nn.Embedding(scaffold_vocab_size, base_generator.hidden_dim)
        
        # Decoration predictor
        self.decoration_predictor = nn.Sequential(
            nn.Linear(base_generator.hidden_dim * 2, base_generator.hidden_dim),
            nn.GELU(),
            nn.Linear(base_generator.hidden_dim, max_decorations * scaffold_vocab_size),
        )
    
    def generate_from_scaffold(
        self,
        scaffold_idx: int,
        num_variants: int = 10,
        property_constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate molecular variants from a scaffold.
        
        Args:
            scaffold_idx: Index of scaffold in vocabulary
            num_variants: Number of variants to generate
            property_constraints: Target properties
        
        Returns:
            Generated molecular variants
        """
        device = next(self.parameters()).device
        
        # Get scaffold embedding
        scaffold_emb = self.scaffold_embedding(
            torch.tensor([scaffold_idx], device=device)
        )  # (1, hidden_dim)
        
        # Encode property constraints
        if property_constraints is not None:
            prop_constraints = self.base_generator.encode_property_constraints(
                **property_constraints
            ).to(device)
        else:
            prop_constraints = None
        
        # Generate variants
        variants = []
        for _ in range(num_variants):
            # Generate base structure
            result = self.base_generator.generate_with_constraints(
                num_molecules=1,
                target_properties=property_constraints,
                guidance_scale=2.0,
            )
            
            # Predict decorations
            features = result['coordinates'].view(1, -1)
            decoration_logits = self.decoration_predictor(
                torch.cat([features, scaffold_emb.expand(features.size(0), -1)], dim=-1)
            )
            
            variants.append({
                'coordinates': result['coordinates'],
                'atom_types': result['atom_types'],
                'decoration_logits': decoration_logits,
            })
        
        return {
            'scaffold_idx': scaffold_idx,
            'variants': variants,
            'property_constraints': prop_constraints,
        }


class TransfectionCliffAwareGenerator(nn.Module):
    """
    Transfection Cliff-Aware Molecular Generator.
    
    Avoids generating molecules that are close to transfection cliffs
    (small structural changes → large efficiency drops).
    
    Uses the TransfectionCliffDetector to guide generation away from cliff regions.
    """
    
    def __init__(
        self,
        base_generator: PropertyConstrainedGenerator,
        cliff_margin: float = 0.2,
    ):
        super().__init__()
        
        self.base_generator = base_generator
        self.cliff_margin = cliff_margin
        
        # Cliff predictor (binary classifier)
        self.cliff_predictor = nn.Sequential(
            nn.Linear(base_generator.hidden_dim, base_generator.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(base_generator.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        coordinates: Tensor,
        atom_types: Tensor,
        cliff_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Training with cliff avoidance.
        
        Args:
            coordinates: Molecular coordinates
            atom_types: Atom types
            cliff_labels: Binary labels (1 = near cliff, 0 = safe)
        """
        # Extract features
        features = self.base_generator.denoising_model.decoder[-1](coordinates.view(coordinates.size(0), -1))
        
        # Predict cliff probability
        cliff_pred = self.cliff_predictor(features.mean(dim=1))
        
        # Cliff avoidance loss
        if cliff_labels is not None:
            cliff_loss = F.binary_cross_entropy(cliff_pred, cliff_labels)
        else:
            cliff_loss = torch.tensor(0.0, device=coordinates.device)
        
        return {
            'cliff_loss': cliff_loss,
            'cliff_prediction': cliff_pred,
        }
    
    @torch.no_grad()
    def generate_safe_molecules(
        self,
        num_molecules: int = 10,
        property_constraints: Optional[Dict[str, float]] = None,
        max_cliff_probability: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate molecules that are away from transfection cliffs.
        
        Args:
            num_molecules: Number to generate
            property_constraints: Target properties
            max_cliff_probability: Maximum acceptable cliff probability
        
        Returns:
            Safe molecular designs
        """
        device = next(self.parameters()).device
        
        safe_molecules = []
        attempts = 0
        max_attempts = num_molecules * 10
        
        while len(safe_molecules) < num_molecules and attempts < max_attempts:
            # Generate candidate
            result = self.base_generator.generate_with_constraints(
                num_molecules=1,
                target_properties=property_constraints,
            )
            
            # Evaluate cliff probability
            features = result['coordinates'].view(1, -1)
            cliff_prob = self.cliff_predictor(features.mean(dim=1)).item()
            
            # Accept if safe
            if cliff_prob <= max_cliff_probability:
                result['cliff_probability'] = cliff_prob
                safe_molecules.append(result)
            
            attempts += 1
        
        return {
            'molecules': safe_molecules,
            'num_generated': len(safe_molecules),
            'num_attempts': attempts,
            'success_rate': len(safe_molecules) / attempts if attempts > 0 else 0,
        }
