"""
Configuration loading utilities.
"""

import os
from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class Config:
    # Model parameters
    base_model_name: str
    output_model_name: str
    device: str
    
    # Dataset parameters
    dataset: str
    max_examples: Optional[int]
    
    # Optimizer settings
    optimizer_type: Optional[str]
    lr: float
    weight_decay: float
    gamma: float
    dropout: Optional[float]
    
    # Training schedule
    num_epochs: int
    batch_size: int
    grad_accumulation_steps: int
    warmup_steps: Optional[int]
    warmup_ratio: Optional[float]
    scheduler_type: Optional[str]
    
    
    
    # Training control
    seed: int
    log_every: int
    
    # Training parameters
    max_seq_length: int
    train_on_answers_only: Optional[bool]
    
    # Output configuration
    output_dir: str
    save_every: int
    
    # Weights & Biases configuration
    wandb_project: Optional[str]
    wandb_name: Optional[str]
    
    # Environment variable names for authentication
    hf_token_env: Optional[str]
    wandb_token_env: Optional[str]




def _get_all_paths(cfg: OmegaConf, path: str = "") -> set:
    """Recursively get all paths (keys) from an OmegaConf config."""
    paths = set()
    
    if not OmegaConf.is_dict(cfg):
        return paths
    
    for key in cfg.keys():
        current_path = f"{path}.{key}" if path else key
        paths.add(current_path)
        
        value = cfg[key]
        if OmegaConf.is_dict(value):
            paths.update(_get_all_paths(value, current_path))
    
    return paths


class _PathTracker:
    """Tracks which config paths are accessed."""
    def __init__(self, cfg: OmegaConf, path: str = "", accessed: set = None):
        self._cfg = cfg
        self._path = path
        self._accessed = accessed if accessed is not None else set()
    
    def __getattr__(self, key: str):
        current_path = f"{self._path}.{key}" if self._path else key
        self._accessed.add(current_path)
        
        try:
            value = getattr(self._cfg, key)
            if OmegaConf.is_dict(value):
                return _PathTracker(value, current_path, self._accessed)
            return value
        except AttributeError:
            raise AttributeError(f"Config path '{current_path}' not found")


def _build_cfg_dict(yaml_cfg: OmegaConf) -> dict:
    """Build cfg_dict from yaml_cfg. Raises error if any required keys are missing."""
    return {
        "base_model_name": yaml_cfg.model.base_name,
        "output_model_name": yaml_cfg.model.output_name,
        "device": yaml_cfg.model.device,
        "dataset": yaml_cfg.dataset.name,
        "max_examples": yaml_cfg.dataset.max_examples,
        "optimizer_type": yaml_cfg.training.optimizer.type,
        "lr": yaml_cfg.training.optimizer.lr,
        "weight_decay": yaml_cfg.training.optimizer.weight_decay,
        "gamma": yaml_cfg.training.optimizer.gamma,
        "num_epochs": yaml_cfg.training.optimizer.num_epochs,
        "batch_size": yaml_cfg.training.optimizer.batch_size,
        "grad_accumulation_steps": yaml_cfg.training.optimizer.grad_accumulation_steps,
        "dropout": yaml_cfg.training.optimizer.dropout,
        "warmup_steps": yaml_cfg.training.schedule.warmup_steps,
        "warmup_ratio": yaml_cfg.training.schedule.warmup_ratio,
        "scheduler_type": yaml_cfg.training.schedule.scheduler_type,
        "seed": yaml_cfg.training.control.seed,
        "log_every": yaml_cfg.training.control.log_every,
        "max_seq_length": yaml_cfg.training.parameters.max_seq_length,
        "train_on_answers_only": yaml_cfg.training.parameters.train_on_answers_only,
        "output_dir": yaml_cfg.output.dir,
        "save_every": yaml_cfg.output.save_every,
        "wandb_project": yaml_cfg.logging.wandb.project,
        "wandb_name": yaml_cfg.logging.wandb.name,
        "hf_token_env": yaml_cfg.auth.env_vars.hf_token,
        "wandb_token_env": yaml_cfg.auth.env_vars.wandb_token,
    }


def load_config(config_path: str = "train_configs/default.yaml") -> Config:
    """Load configuration from YAML file and convert to Config dataclass."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load default config first
    default_cfg = OmegaConf.load("train_configs/default.yaml")
    
    # Track which paths are actually accessed when building cfg_dict
    accessed_paths = set()
    tracker = _PathTracker(default_cfg, accessed=accessed_paths)
    try:
        _build_cfg_dict(tracker)
        used_paths = accessed_paths
    except Exception as e:
        raise ValueError(
            f"train_configs/default.yaml is missing required keys. "
            f"Error: {e}\n"
            f"All keys must be present in default.yaml"
        ) from e
    
    # Check for unused keys in default.yaml
    default_paths = _get_all_paths(default_cfg)
    unused_in_default = default_paths - used_paths
    if unused_in_default:
        raise ValueError(
            f"train_configs/default.yaml contains unused keys:\n"
            f"  Unused keys: {', '.join(sorted(unused_in_default))}\n"
            f"  All keys in default.yaml must be used"
        )
    
    # Load experiment config
    experiment_cfg = OmegaConf.load(config_path)
    
    # Check for extra keys in experiment config (keys not in default)
    experiment_paths = _get_all_paths(experiment_cfg)
    extra_in_experiment = experiment_paths - default_paths
    if extra_in_experiment:
        raise ValueError(
            f"{config_path} contains keys not in default.yaml:\n"
            f"  Extra keys: {', '.join(sorted(extra_in_experiment))}\n"
            f"  All keys must be defined in train_configs/default.yaml"
        )
    
    # Merge experiment config over default config (experiment takes precedence)
    yaml_cfg = OmegaConf.merge(default_cfg, experiment_cfg)
    
    # Build cfg_dict from merged config
    cfg_dict = _build_cfg_dict(yaml_cfg)
    
    return Config(**cfg_dict)