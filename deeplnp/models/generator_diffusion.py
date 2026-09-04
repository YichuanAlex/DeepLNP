"""
3D Molecular Diffusion Generator for de novo ionizable lipid design.

This module implements a 3D diffusion model for generating novel molecular structures
with constrained properties (pKa, LogP, biodegradability, etc.).

Key innovations:
1. 3D geometry-aware generation
2. Property-constrained diffusion
3. Scaffold-based controlled generation

Architecture inspired by:
- GeoDiff (Geometry-Enhanced Diffusion)
- EDM (Elucidating Diffusion Models)
- TransMA's 3D Transformer
"""

from typing import Dict, List, Optional, Tuple, Any, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process for 3D molecular generation.
    
    Forward process: gradually add noise
    Reverse process: learn to denoise
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        # Beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            t = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alphas_cumprod = torch.cos((t / num_timesteps) * (math.pi / 2)) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.register_buffer('betas', betas)
        
        # Precompute useful quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # For reverse process
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        betas_prev = torch.cat([torch.tensor([self.betas[1]]), betas[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def q_sample(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_0: (batch, ..., dim) - Original data
            t: (batch,) - Timestep
            noise: (batch, ..., dim) - Noise to add
        
        Returns:
            x_t: (batch, ..., dim) - Noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior(
        self,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute q(x_{t-1} | x_t, x_0)
        
        Returns:
            mean: Posterior mean
            var: Posterior variance
            log_var: Log of posterior variance
        """
        posterior_mean_coef1 = (
            self.betas[t] * torch.sqrt(self.alphas_cumprod[t]) / (1 - self.alphas_cumprod[t])
        )
        posterior_mean_coef2 = (
            torch.sqrt(self.alphas[t]) * (1 - self.alphas_cumprod[t-1]) / (1 - self.alphas_cumprod[t])
        )
        
        posterior_mean = (
            posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        )
        
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var = torch.log(posterior_var)
        
        return posterior_mean, posterior_var, posterior_log_var
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample from p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model
            x_t: Current noisy state
            t: Timestep
            condition: Conditional input (e.g., property constraints)
        
        Returns:
            x_{t-1}: Denoised state
        """
        # Predict noise
        noise_pred = model(x_t, t, condition)
        
        # Compute posterior mean
        pred_x_0 = (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise_pred
        )
        
        posterior_mean, posterior_var, _ = self.q_posterior(pred_x_0, x_t, t)
        
        # Sample
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_var) * noise
        
        return x_prev
    
    def _extract(
        self,
        a: Tensor,
        t: Tensor,
        x_shape: torch.Size,
    ) -> Tensor:
        """Extract values from a at indices t and reshape to match x_shape."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for 3D molecular denoising.
    
    Architecture:
    1. Time embedding
    2. Property condition embedding
    3. 3D coordinate processing
    4. Skip connections
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # 3D coordinates
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        condition_dim: int = 128,
        time_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Property condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim if i > 0 else input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for i in range(num_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (batch, input_dim) - Noisy input
            t: (batch,) - Timestep
            condition: (batch, condition_dim) - Property constraints
        
        Returns:
            noise_pred: (batch, input_dim) - Predicted noise
        """
        # Time embedding
        time_emb = self.time_embed(self._timestep_embedding(t))
        
        # Condition embedding
        if condition is not None:
            cond_emb = self.condition_embed(condition)
            time_emb = time_emb + cond_emb
        
        # Encoder with skip connections
        skip_features = []
        h = x
        for i, encoder_layer in enumerate(self.encoder):
            h = encoder_layer(h)
            h = h + time_emb
            skip_features.append(h)
        
        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            skip_feat = skip_features[-(i+1)]
            h = torch.cat([h, skip_feat], dim=-1)
            h = decoder_layer(h)
            h = h + time_emb
        
        # Output
        noise_pred = self.output(h)
        
        return noise_pred
    
    def _timestep_embedding(
        self,
        timesteps: Tensor,
        max_period: int = 10000,
    ) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: (batch,)
            max_period: Maximum period
        
        Returns:
            embedding: (batch, dim)
        """
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding


class MolecularDiffusionGenerator(nn.Module):
    """
    3D Molecular Diffusion Generator for ionizable lipid design.
    
    Key features:
    1. Generate 3D molecular structures
    2. Property-constrained generation (pKa, LogP, etc.)
    3. Scaffold-based controlled generation
    """
    
    def __init__(
        self,
        # Diffusion parameters
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        # Model parameters
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        # Condition parameters
        property_dim: int = 128,
        # Vocab for atom types
        num_atom_types: int = 20,
        max_atoms: int = 50,
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.max_atoms = max_atoms
        
        # Diffusion process
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        
        # Denoising model
        self.denoising_model = ConditionalUNet(
            input_dim=3,  # 3D coordinates
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            condition_dim=property_dim,
        )
        
        # Atom type prediction
        self.atom_type_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_atom_types),
        )
        
        # Property predictor (for guidance)
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, property_dim),
        )
    
    def forward(
        self,
        coordinates: Tensor,
        atom_types: Tensor,
        t: Tensor,
        property_constraints: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Training forward pass.
        
        Args:
            coordinates: (batch, num_atoms, 3) - 3D coordinates
            atom_types: (batch, num_atoms) - Atom types
            t: (batch,) - Timestep
            property_constraints: (batch, property_dim) - Target properties
        
        Returns:
            loss: Total loss
            coord_loss: Coordinate reconstruction loss
            atom_loss: Atom type prediction loss
            property_loss: Property prediction loss
        """
        batch_size = coordinates.size(0)
        
        # Add noise
        noise = torch.randn_like(coordinates)
        noisy_coords = self.diffusion.q_sample(coordinates, t, noise)
        
        # Predict noise
        noise_pred = self.denoising_model(noisy_coords, t, property_constraints)
        
        # Coordinate reconstruction loss
        coord_loss = F.mse_loss(noise_pred, noise)
        
        # Atom type prediction (from denoised features)
        features = self.denoising_model.decoder[-1](noisy_coords)
        atom_pred = self.atom_type_predictor(features)
        atom_loss = F.cross_entropy(atom_pred.view(-1, self.num_atom_types), atom_types.view(-1))
        
        # Property prediction (for consistency)
        if property_constraints is not None:
            prop_pred = self.property_predictor(features.mean(dim=1))
            property_loss = F.mse_loss(prop_pred, property_constraints)
        else:
            property_loss = torch.tensor(0.0, device=coordinates.device)
        
        # Total loss
        loss = coord_loss + atom_loss + 0.1 * property_loss
        
        return {
            'loss': loss,
            'coord_loss': coord_loss,
            'atom_loss': atom_loss,
            'property_loss': property_loss,
        }
    
    @torch.no_grad()
    def sample(
        self,
        num_molecules: int = 1,
        property_constraints: Optional[Tensor] = None,
        guidance_scale: float = 1.0,
        return_intermediates: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Generate new molecules via reverse diffusion.
        
        Args:
            num_molecules: Number of molecules to generate
            property_constraints: (num_molecules, property_dim) - Target properties
            guidance_scale: Classifier-free guidance scale
            return_intermediates: Whether to return intermediate states
        
        Returns:
            coordinates: (num_molecules, num_atoms, 3) - Generated 3D coordinates
            atom_types: (num_molecules, num_atoms) - Generated atom types
            intermediates: List of intermediate states (if return_intermediates)
        """
        device = next(self.parameters()).device
        
        # Start from pure noise
        coordinates = torch.randn(
            num_molecules, self.max_atoms, 3, device=device
        )
        
        intermediates = []
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_molecules,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.denoising_model(coordinates, t_batch, property_constraints)
            
            # Classifier-free guidance (if unconditional condition is available)
            if guidance_scale != 1.0 and property_constraints is not None:
                # Unconditional prediction
                uncond_pred = self.denoising_model(
                    coordinates, t_batch, torch.zeros_like(property_constraints)
                )
                # Guided prediction
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
            
            # Sample x_{t-1}
            coordinates = self.diffusion.p_sample(
                self.denoising_model,
                coordinates,
                t_batch,
                property_constraints,
            )
            
            if return_intermediates and t % 100 == 0:
                intermediates.append(coordinates.clone())
        
        # Predict atom types
        features = self.denoising_model.decoder[-1](coordinates)
        atom_logits = self.atom_type_predictor(features)
        atom_types = atom_logits.argmax(dim=-1)
        
        result = {
            'coordinates': coordinates,
            'atom_types': atom_types,
        }
        
        if return_intermediates:
            result['intermediates'] = intermediates
        
        return result
    
    def generate_with_constraints(
        self,
        num_molecules: int = 100,
        pka_range: Tuple[float, float] = (6.2, 6.8),
        logp_range: Tuple[float, float] = (3.0, 6.0),
        mol_wt_range: Tuple[float, float] = (400, 800),
        min_ester_bonds: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate molecules with property constraints.
        
        Args:
            num_molecules: Number of molecules to generate
            pka_range: Target pKa range
            logp_range: Target LogP range
            mol_wt_range: Target molecular weight range
            min_ester_bonds: Minimum number of ester bonds (for biodegradability)
        
        Returns:
            molecules: List of generated molecules with properties
        """
        # Encode property constraints
        property_constraints = self._encode_constraints(
            pka_range=pka_range,
            logp_range=logp_range,
            mol_wt_range=mol_wt_range,
            min_ester_bonds=min_ester_bonds,
            num_molecules=num_molecules,
        )
        
        # Generate
        output = self.sample(
            num_molecules=num_molecules,
            property_constraints=property_constraints,
            guidance_scale=1.5,
        )
        
        # Convert to RDKit molecules
        molecules = self._coordinates_to_molecules(
            coordinates=output['coordinates'],
            atom_types=output['atom_types'],
        )
        
        return molecules
    
    def _encode_constraints(
        self,
        pka_range: Tuple[float, float],
        logp_range: Tuple[float, float],
        mol_wt_range: Tuple[float, float],
        min_ester_bonds: int,
        num_molecules: int,
    ) -> Tensor:
        """Encode property constraints as embedding."""
        device = next(self.parameters()).device
        
        # Normalize constraints
        pka = (pka_range[0] + pka_range[1]) / 2 / 14.0  # Normalize to [0, 1]
        logp = (logp_range[0] + logp_range[1]) / 2 / 10.0
        mol_wt = (mol_wt_range[0] + mol_wt_range[1]) / 2 / 1000.0
        ester = min_ester_bonds / 10.0
        
        # Create constraint vector
        constraints = torch.tensor(
            [[pka, logp, mol_wt, ester]],
            device=device,
            dtype=torch.float32,
        ).repeat(num_molecules, 1)
        
        # Embed
        return self.property_predictor[0](constraints)
    
    def _coordinates_to_molecules(
        self,
        coordinates: Tensor,
        atom_types: Tensor,
    ) -> List[Dict[str, Any]]:
        """Convert coordinates and atom types to RDKit molecules."""
        # This would require RDKit integration
        # For now, return raw data
        molecules = []
        for i in range(coordinates.size(0)):
            mol = {
                'coordinates': coordinates[i].cpu().numpy(),
                'atom_types': atom_types[i].cpu().numpy(),
            }
            molecules.append(mol)
        
        return molecules
