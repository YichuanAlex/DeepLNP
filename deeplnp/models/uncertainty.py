"""
Uncertainty Estimation Module for DeepLNP.

This module implements multiple uncertainty estimation methods:
1. MC Dropout (Monte Carlo Dropout)
2. Deep Ensembles
3. Evidential Deep Learning

Key applications:
1. Active learning (select uncertain samples)
2. Risk-aware optimization (avoid high-uncertainty regions)
3. Model calibration (assess prediction reliability)
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, Gamma


class MCDropoutUncertainty(nn.Module):
    """
    Uncertainty estimation via Monte Carlo Dropout.
    
    During inference, keep dropout enabled and sample multiple predictions.
    Uncertainty is estimated as the variance across samples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def forward(
        self,
        batch: Dict[str, Tensor],
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            batch: Input batch
            task_names: List of task names to predict
        
        Returns:
            predictions: Mean predictions
            uncertainty: Predictive variance (uncertainty)
            epistemic: Epistemic uncertainty
            aleatoric: Aleatoric uncertainty
        """
        # Enable training mode for dropout
        self.model.train()
        
        # Collect predictions from multiple forward passes
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(batch)
                if task_names is None:
                    task_names = list(output.keys())
                
                # Stack predictions
                preds = torch.stack([output[task] for task in task_names], dim=-1)
                all_predictions.append(preds)
        
        # Stack all samples: (num_samples, batch, num_tasks)
        all_predictions = torch.stack(all_predictions, dim=0)
        
        # Calculate mean and variance
        mean_pred = all_predictions.mean(dim=0)  # (batch, num_tasks)
        var_pred = all_predictions.var(dim=0)    # (batch, num_tasks)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = var_pred
        
        # Aleatoric uncertainty (data uncertainty) - approximated from mean
        aleatoric = torch.abs(mean_pred) * 0.1  # Simple approximation
        
        # Disable training mode
        self.model.eval()
        
        return {
            'predictions': mean_pred,
            'uncertainty': var_pred,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
        }


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty estimation.
    
    Train multiple models with different initializations
    and aggregate their predictions.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = "mean",
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.aggregation = aggregation
    
    def forward(
        self,
        batch: Dict[str, Tensor],
        return_individual: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Args:
            batch: Input batch
            return_individual: whether to return individual model predictions
        
        Returns:
            predictions: Ensemble mean predictions
            uncertainty: Predictive variance
            individual: Individual model predictions (optional)
        """
        all_predictions = []
        
        # Collect predictions from all models
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(batch)
                
                # Stack task predictions
                task_names = list(output.keys())
                preds = torch.stack([output[task] for task in task_names], dim=-1)
                all_predictions.append(preds)
        
        # Stack: (num_models, batch, num_tasks)
        all_predictions = torch.stack(all_predictions, dim=0)
        
        # Ensemble statistics
        mean_pred = all_predictions.mean(dim=0)
        var_pred = all_predictions.var(dim=0)
        
        result = {
            'predictions': mean_pred,
            'uncertainty': var_pred,
        }
        
        if return_individual:
            result['individual'] = all_predictions
        
        return result
    
    @classmethod
    def from_checkpoint(
        cls,
        model_class: type,
        checkpoint_paths: List[str],
        **model_kwargs,
    ) -> 'DeepEnsemble':
        """
        Create ensemble from multiple checkpoint files.
        
        Args:
            model_class: Model class to instantiate
            checkpoint_paths: List of checkpoint file paths
            model_kwargs: Arguments for model initialization
        
        Returns:
            DeepEnsemble instance
        """
        models = []
        for path in checkpoint_paths:
            model = model_class(**model_kwargs)
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            models.append(model)
        
        return cls(models)


class EvidentialRegression(nn.Module):
    """
    Evidential Deep Learning for regression.
    
    Predicts parameters of a Normal Inverse Gamma distribution
    to model both prediction and uncertainty.
    
    Reference:
    "Evidential Deep Learning to Quantify Classification Uncertainty"
    (Sensoy et al., NeurIPS 2018)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Predict NIG parameters: gamma, nu, alpha, beta
        # For each output dimension
        self.nig_head = nn.Linear(hidden_dim, output_dim * 4)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: (batch, input_dim)
        
        Returns:
            gamma: Predicted mean
            nu: Uncertainty parameter
            alpha: Precision parameter
            beta: Variance parameter
        """
        features = self.feature_extractor(x)
        nig_params = self.nig_head(features)
        
        # Split into NIG parameters
        gamma, nu, alpha, beta = nig_params.chunk(4, dim=-1)
        
        # Apply constraints
        nu = F.softplus(nu)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)
        
        return {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
        }
    
    def compute_uncertainty(
        self,
        gamma: Tensor,
        nu: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute uncertainty metrics from NIG parameters.
        
        Returns:
            pred_mean: Predicted mean
            pred_var: Predicted variance
            epistemic: Epistemic uncertainty
            aleatoric: Aleatoric uncertainty
        """
        # Predicted mean
        pred_mean = gamma
        
        # Predicted variance
        pred_var = beta / (nu * (alpha - 1))
        
        # Epistemic uncertainty (from nu)
        epistemic = 1 / nu
        
        # Aleatoric uncertainty (from alpha, beta)
        aleatoric = beta / (alpha - 1)
        
        return {
            'pred_mean': pred_mean,
            'pred_var': pred_var,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
        }


class EvidentialLoss(nn.Module):
    """
    Loss function for Evidential Regression.
    
    Combines NLL loss with KL divergence regularization.
    """
    
    def __init__(self, lambda_nll: float = 1.0, lambda_kl: float = 0.01):
        super().__init__()
        
        self.lambda_nll = lambda_nll
        self.lambda_kl = lambda_kl
    
    def forward(
        self,
        nig_params: Dict[str, Tensor],
        targets: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            nig_params: Dictionary with gamma, nu, alpha, beta
            targets: (batch, output_dim)
        
        Returns:
            total_loss: Combined NLL + KL loss
            nll_loss: Negative log-likelihood
            kl_loss: KL divergence regularization
        """
        gamma = nig_params['gamma']
        nu = nig_params['nu']
        alpha = nig_params['alpha']
        beta = nig_params['beta']
        
        # NLL loss
        nll_loss = self._nll_loss(gamma, nu, alpha, beta, targets)
        
        # KL divergence regularization
        kl_loss = self._kl_divergence(nu, alpha, beta)
        
        # Total loss
        total_loss = self.lambda_nll * nll_loss + self.lambda_kl * kl_loss
        
        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'kl_loss': kl_loss,
        }
    
    def _nll_loss(
        self,
        gamma: Tensor,
        nu: Tensor,
        alpha: Tensor,
        beta: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Negative log-likelihood loss."""
        error = torch.abs(targets - gamma)
        
        nll = 0.5 * torch.log(torch.pi / nu) - alpha * torch.log(
            2 * beta * (1 + nu)
        ) + (alpha + 0.5) * torch.log(2 * beta * (1 + nu) + nu * error ** 2)
        
        return nll.mean()
    
    def _kl_divergence(
        self,
        nu: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """KL divergence regularization."""
        # Evidence for the prior
        nu_0 = 1.0
        alpha_0 = 1.0
        beta_0 = 1.0
        
        # KL divergence
        kl = (
            0.5 * torch.log(nu_0 / nu)
            - 0.5 * torch.log(alpha_0 / alpha)
            + 0.5 * (alpha - alpha_0)
            + 0.5 * (beta - beta_0) / beta
        )
        
        return kl.mean()


class UncertaintyEstimator:
    """
    High-level interface for uncertainty estimation.
    
    Supports multiple methods:
    1. MC Dropout
    2. Deep Ensembles
    3. Evidential Deep Learning
    """
    
    def __init__(
        self,
        method: str = "mc_dropout",
        **kwargs,
    ):
        """
        Args:
            method: "mc_dropout", "ensemble", or "evidential"
            **kwargs: Method-specific arguments
        """
        self.method = method
        self.kwargs = kwargs
        
        if method == "mc_dropout":
            self.estimator = MCDropoutUncertainty(
                model=kwargs.get('model'),
                num_samples=kwargs.get('num_samples', 10),
            )
        elif method == "ensemble":
            self.estimator = DeepEnsemble(
                models=kwargs.get('models', []),
            )
        elif method == "evidential":
            self.estimator = EvidentialRegression(
                input_dim=kwargs.get('input_dim', 512),
                hidden_dim=kwargs.get('hidden_dim', 256),
                output_dim=kwargs.get('output_dim', 1),
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Predict with uncertainty estimates.
        
        Args:
            batch: Input batch
        
        Returns:
            predictions: Mean predictions
            uncertainty: Uncertainty estimates
            epistemic: Epistemic uncertainty
            aleatoric: Aleatoric uncertainty
        """
        if self.method in ["mc_dropout", "ensemble"]:
            output = self.estimator(batch)
            return {
                'predictions': output['predictions'],
                'uncertainty': output['uncertainty'],
                'epistemic': output.get('epistemic', output['uncertainty']),
                'aleatoric': output.get('aleatoric', output['uncertainty'] * 0.5),
            }
        elif self.method == "evidential":
            nig_params = self.estimator(batch)
            uncertainty = self.estimator.compute_uncertainty(**nig_params)
            return {
                'predictions': uncertainty['pred_mean'],
                'uncertainty': uncertainty['pred_var'],
                'epistemic': uncertainty['epistemic'],
                'aleatoric': uncertainty['aleatoric'],
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")
