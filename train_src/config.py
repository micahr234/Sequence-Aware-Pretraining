"""
Configuration loading utilities.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
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
    
    # Output configuration
    output_dir: str
    save_every: int
    
    # Weights & Biases configuration
    wandb_project: Optional[str]
    wandb_name: Optional[str]
    
    # Environment variable names for authentication
    hf_token_env: Optional[str]
    wandb_token_env: Optional[str]




def load_config(config_path: str = "train_configs/default.yaml") -> Config:
    """Load configuration from YAML file and convert to Config dataclass."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load default config first
    default_cfg = OmegaConf.load("train_configs/default.yaml")
    
    # Load experiment config
    experiment_cfg = OmegaConf.load(config_path)
    
    # Merge experiment config over default config (experiment takes precedence)
    yaml_cfg = OmegaConf.merge(default_cfg, experiment_cfg)
    
    # Handle device auto-detection
    if yaml_cfg.hardware.device == "auto":
        yaml_cfg.hardware.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to flat structure for Config dataclass
    cfg_dict = {
        # Model parameters
        "base_model_name": yaml_cfg.model.base_name,
        "output_model_name": yaml_cfg.model.output_name,
        "device": yaml_cfg.hardware.device,
        
        # Dataset parameters
        "dataset": yaml_cfg.dataset.name,
        "max_examples": yaml_cfg.dataset.max_examples,
        
        # Optimizer settings
        "lr": yaml_cfg.training.optimizer.lr,
        "weight_decay": yaml_cfg.training.optimizer.weight_decay,
        "gamma": yaml_cfg.training.optimizer.gamma,
        "dropout": yaml_cfg.training.optimizer.dropout,
        
        # Training schedule
        "num_epochs": yaml_cfg.training.schedule.num_epochs,
        "batch_size": yaml_cfg.training.schedule.batch_size,
        "grad_accumulation_steps": yaml_cfg.training.schedule.grad_accumulation_steps,
        "warmup_steps": yaml_cfg.training.schedule.warmup_steps,
        "warmup_ratio": yaml_cfg.training.schedule.warmup_ratio,
        "scheduler_type": yaml_cfg.training.schedule.scheduler_type,
        
        
        
        # Training control
        "seed": yaml_cfg.training.control.seed,
        "log_every": yaml_cfg.training.control.log_every,
        
        # Training parameters
        "max_seq_length": yaml_cfg.training_params.max_seq_length,
        
        # Output configuration
        "output_dir": yaml_cfg.output.dir,
        "save_every": yaml_cfg.output.save_every,
        
        # Weights & Biases configuration
        "wandb_project": yaml_cfg.wandb.project,
        "wandb_name": yaml_cfg.wandb.name,
        
        # Environment variable names for authentication
        "hf_token_env": yaml_cfg.env_vars.hf_token,
        "wandb_token_env": yaml_cfg.env_vars.wandb_token,
    }
    
    return Config(**cfg_dict)