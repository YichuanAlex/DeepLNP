"""Configuration for DeepLNP models and training."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Configuration for DeepLNP models."""
    
    # Model architecture
    model_type: str = "deeplnp"  # deeplnp, comet, transma, lantern
    
    # Encoder settings
    encoder_type: str = "multimodal"  # multimodal, graph_only, transformer_only
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # 3D encoder (Uni-Mol)
    use_3d: bool = True
    embed_dim: int = 768
    ffn_embed_dim: int = 3072
    
    # Graph encoder
    atom_feature_dim: int = 39
    bond_feature_dim: int = 4
    
    # Transformer encoder
    max_seq_len: int = 256
    vocab_size: int = 100
    
    # Fusion
    fusion_type: str = "cross_attention"  # cross_attention, concat, moe
    num_experts: int = 4
    
    # Predictor heads
    num_tasks: int = 4  # efficiency, size, zeta, toxicity
    task_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.3, 0.5])
    
    # Generator (diffusion model)
    generator_type: str = "diffusion"
    diffusion_steps: int = 1000
    latent_dim: int = 256


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Data
    data_dir: str = "data/processed"
    dataset_name: str = "lnp_atlas"
    split_strategy: str = "scaffold"  # random, scaffold, cliff
    
    # Training
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    
    # Regularization
    dropout: float = 0.1
    early_stopping_patience: int = 20
    
    # Multi-task
    use_multi_task: bool = True
    
    # Uncertainty
    use_uncertainty: bool = True
    num_ensemble: int = 5
    
    # Device
    device: str = "auto"
    num_workers: int = 4
    
    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_every: int = 100
    
    # WandB
    use_wandb: bool = True
    project_name: str = "DeepLNP"
    run_name: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for molecule generation."""
    
    # Generation
    num_candidates: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # Constraints
    min_mol_wt: float = 200
    max_mol_wt: float = 1000
    min_logp: float = 0
    max_logp: float = 8
    target_pka_range: List[float] = field(default_factory=lambda: [6.2, 6.8])
    
    # Optimization
    use_bayesian_opt: bool = True
    num_iterations: int = 50
    acquisition_function: str = "ei"  # ei, ucb, poi


@dataclass
class DeepLNPConfig:
    """Main configuration for DeepLNP."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Metadata
    seed: int = 42
    version: str = "0.1.0"
    
    def save(self, path: str):
        """Save config to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "DeepLNPConfig":
        """Load config from YAML."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": {
                "model_type": self.model.model_type,
                "encoder_type": self.model.encoder_type,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "dropout": self.model.dropout,
                "use_3d": self.model.use_3d,
                "embed_dim": self.model.embed_dim,
                "ffn_embed_dim": self.model.ffn_embed_dim,
                "fusion_type": self.model.fusion_type,
                "num_tasks": self.model.num_tasks,
            },
            "training": {
                "data_dir": self.training.data_dir,
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "optimizer": self.training.optimizer,
                "scheduler": self.training.scheduler,
                "early_stopping_patience": self.training.early_stopping_patience,
                "device": self.training.device,
                "log_dir": self.training.log_dir,
                "checkpoint_dir": self.training.checkpoint_dir,
            },
            "generation": {
                "num_candidates": self.generation.num_candidates,
                "temperature": self.generation.temperature,
                "target_pka_range": self.generation.target_pka_range,
                "use_bayesian_opt": self.generation.use_bayesian_opt,
            },
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DeepLNPConfig":
        """Create config from dictionary."""
        config = cls()
        
        if "model" in config_dict:
            for k, v in config_dict["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        
        if "training" in config_dict:
            for k, v in config_dict["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        if "generation" in config_dict:
            for k, v in config_dict["generation"].items():
                if hasattr(config.generation, k):
                    setattr(config.generation, k, v)
        
        if "seed" in config_dict:
            config.seed = config_dict["seed"]
        
        return config


def get_config(config_path: Optional[str] = None) -> DeepLNPConfig:
    """Get configuration, loading from file if provided."""
    if config_path and Path(config_path).exists():
        return DeepLNPConfig.load(config_path)
    else:
        return DeepLNPConfig()
