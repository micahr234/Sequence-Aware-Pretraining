"""
Configuration loading utilities for both training and testing.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model parameters
    base_model_name: str
    output_model_name: str
    device: str
    
    # Dataset parameters
    dataset: str
    max_examples: Optional[int]
    question_template: str  # Template for formatting question (e.g., "{question}")
    answer_template: str  # Template for formatting answer (e.g., "{answer}" or "{reasoning}\n\nThe answer is: {answer}")
    
    # Optimizer settings
    optimizer_type: Optional[str]
    lr: float
    weight_decay: float
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
    
    # Evaluation configuration
    eval_dataset: Optional[str]  # Dataset name for evaluation (e.g., "gsm8k")
    eval_split: Optional[str]  # Split to use (e.g., "test")
    eval_max_examples: Optional[int]  # Max examples to evaluate (None for all)
    eval_interval_steps: Optional[int]  # Evaluate every N steps (None to disable)
    eval_answer_regex: Optional[str]  # Regex pattern to extract answer from model output (None to use default)


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


def _build_train_cfg_dict(yaml_cfg: OmegaConf) -> dict:
    """Build train cfg_dict from yaml_cfg. All parameters must be explicitly defined."""
    eval_section = yaml_cfg.evaluation
    return {
        "base_model_name": yaml_cfg.model.base_name,
        "output_model_name": yaml_cfg.model.output_name,
        "device": yaml_cfg.model.device,
        "dataset": yaml_cfg.dataset.name,
        "max_examples": yaml_cfg.dataset.max_examples,
        "question_template": yaml_cfg.dataset.question_template,
        "answer_template": yaml_cfg.dataset.answer_template,
        "optimizer_type": yaml_cfg.training.optimizer.type,
        "lr": yaml_cfg.training.optimizer.lr,
        "weight_decay": yaml_cfg.training.optimizer.weight_decay,
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
        "eval_dataset": eval_section.dataset,
        "eval_split": eval_section.split,
        "eval_max_examples": eval_section.max_examples,
        "eval_interval_steps": eval_section.interval_steps,
        "eval_answer_regex": eval_section.answer_regex,
    }


def load_train_config(config_path: str = "train_configs/default.yaml") -> TrainConfig:
    """Load training configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load default config first
    default_cfg = OmegaConf.load("train_configs/default.yaml")
    
    # Load experiment config
    experiment_cfg = OmegaConf.load(config_path)
    
    # Check for extra keys in experiment config
    default_paths = _get_all_paths(default_cfg)
    experiment_paths = _get_all_paths(experiment_cfg)
    extra_in_experiment = experiment_paths - default_paths
    if extra_in_experiment:
        raise ValueError(
            f"{config_path} contains keys not in default.yaml:\n"
            f"  Extra keys: {', '.join(sorted(extra_in_experiment))}\n"
            f"  All keys must be defined in train_configs/default.yaml"
        )
    
    # Merge experiment config over default config
    # All parameters must be explicitly defined - no implicit defaults
    # Experiment config values override corresponding default.yaml values
    yaml_cfg = OmegaConf.merge(default_cfg, experiment_cfg)
    
    # Track which paths are actually accessed from the merged config
    accessed_paths = set()
    tracker = _PathTracker(yaml_cfg, accessed=accessed_paths)
    try:
        _build_train_cfg_dict(tracker)
        used_paths = accessed_paths
    except (AttributeError, KeyError) as e:
        raise ValueError(
            f"Config is missing required keys. "
            f"Error: {e}\n"
            f"All parameters must be explicitly defined in default.yaml and/or {config_path}"
        ) from e
    
    # Check for unused keys in default.yaml (all keys must be used)
    unused_in_default = default_paths - used_paths
    if unused_in_default:
        raise ValueError(
            f"train_configs/default.yaml contains unused keys:\n"
            f"  Unused keys: {', '.join(sorted(unused_in_default))}\n"
            f"  All keys in default.yaml must be used"
        )
    
    # Build cfg_dict from merged config - all parameters are explicitly defined (no defaults)
    cfg_dict = _build_train_cfg_dict(yaml_cfg)
    
    # Handle device auto-detection
    if cfg_dict["device"] == "auto":
        cfg_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return TrainConfig(**cfg_dict)


# Backwards compatibility aliases
Config = TrainConfig
load_config = load_train_config

