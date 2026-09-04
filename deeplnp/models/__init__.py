"""Models module for DeepLNP."""

from .graph_encoder import GraphEncoder, UniMolEncoder
from .transformer_encoder import TransformerEncoderWithPair
from .multimodal_fusion import MultiModalFusion, CrossAttention, MixtureOfExperts, AdaptiveFeatureFusion
from .predictor import LNPPredictor, LNPClassifier
from .generator_diffusion import MolecularDiffusionGenerator, GaussianDiffusion, ConditionalUNet
from .generator_3d_enhanced import (
    GeometryEnhancedEncoder,
    PropertyConstrainedGenerator,
    ScaffoldBasedGenerator,
    TransfectionCliffAwareGenerator,
)
from .uncertainty import (
    UncertaintyEstimator,
    MCDropoutUncertainty,
    DeepEnsemble,
    EvidentialRegression,
    EvidentialLoss,
)

def evaluate_and_select_candidates(*args, **kwargs):
    from evaluate import evaluate_and_select_candidates as _evaluate_and_select_candidates
    return _evaluate_and_select_candidates(*args, **kwargs)

__all__ = [
    "GraphEncoder",
    "UniMolEncoder",
    "TransformerEncoderWithPair",
    "MultiModalFusion",
    "CrossAttention",
    "MixtureOfExperts",
    "AdaptiveFeatureFusion",
    "LNPPredictor",
    "LNPClassifier",
    "MolecularDiffusionGenerator",
    "GaussianDiffusion",
    "ConditionalUNet",
    "GeometryEnhancedEncoder",
    "PropertyConstrainedGenerator",
    "ScaffoldBasedGenerator",
    "TransfectionCliffAwareGenerator",
    "UncertaintyEstimator",
    "MCDropoutUncertainty",
    "DeepEnsemble",
    "EvidentialRegression",
    "EvidentialLoss",
    "evaluate_and_select_candidates",
]
