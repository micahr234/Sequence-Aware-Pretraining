"""
Test configuration loading utilities.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from omegaconf import OmegaConf


@dataclass
class TestConfig:
    # Model parameters
    model_name: str
    device: str
    
    # Generation parameters
    max_new_tokens: int
    num_samples: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    pad_token_id: Optional[int]
    
    # Prompt configuration
    prompt: str
    prompt_file: Optional[str]
    
    # Analysis parameters
    save_logits: bool
    save_probabilities: bool
    top_k_probs: int
    output_dir: str
    save_format: str
    save_visualizations: bool
    
    # Visualization parameters
    plot_top_k: int
    plot_width: int
    plot_height: int
    dpi: int
    colormap: str
    colorbar: bool
    
    # Experiment parameters
    name: str
    description: str
    seed: int
    
    # Environment parameters
    hf_token_env: Optional[str]
    wandb_token_env: Optional[str]
    wandb_project: Optional[str]
    wandb_name: Optional[str]


def load_test_config(config_path: str = "test_configs/default.yaml") -> TestConfig:
    """
    Load test configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        TestConfig object with loaded parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load default config first
    default_cfg = OmegaConf.load("test_configs/default.yaml")
    
    # Load experiment config
    experiment_cfg = OmegaConf.load(config_path)
    
    # Merge experiment config over default config
    yaml_cfg = OmegaConf.merge(default_cfg, experiment_cfg)
    
    # Handle device auto-detection
    if yaml_cfg.model.device == "auto":
        yaml_cfg.model.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg_dict = {
        # Model parameters
        "model_name": yaml_cfg.model.name,
        "device": yaml_cfg.model.device,
        
        # Generation parameters
        "max_new_tokens": yaml_cfg.generation.max_new_tokens,
        "num_samples": yaml_cfg.generation.num_samples,
        "temperature": yaml_cfg.generation.temperature,
        "top_p": yaml_cfg.generation.top_p,
        "top_k": yaml_cfg.generation.top_k,
        "do_sample": yaml_cfg.generation.do_sample,
        "pad_token_id": yaml_cfg.generation.pad_token_id,
        
        # Prompt configuration
        "prompt": yaml_cfg.generation.prompt,
        "prompt_file": yaml_cfg.generation.prompt_file,
        
        # Analysis parameters
        "save_logits": yaml_cfg.analysis.save_logits,
        "save_probabilities": yaml_cfg.analysis.save_probabilities,
        "top_k_probs": yaml_cfg.analysis.top_k_probs,
        "output_dir": yaml_cfg.analysis.output_dir,
        "save_format": yaml_cfg.analysis.save_format,
        "save_visualizations": yaml_cfg.analysis.save_visualizations,
        
        # Visualization parameters
        "plot_top_k": yaml_cfg.visualization.plot_top_k,
        "plot_width": yaml_cfg.visualization.plot_width,
        "plot_height": yaml_cfg.visualization.plot_height,
        "dpi": yaml_cfg.visualization.dpi,
        "colormap": yaml_cfg.visualization.colormap,
        "colorbar": yaml_cfg.visualization.colorbar,
        
        # Experiment parameters
        "name": yaml_cfg.experiment.name,
        "description": yaml_cfg.experiment.description,
        "seed": yaml_cfg.experiment.seed,
        
        # Environment parameters
        "hf_token_env": yaml_cfg.env.hf_token_env,
        "wandb_token_env": yaml_cfg.env.wandb_token_env,
        "wandb_project": yaml_cfg.env.wandb_project,
        "wandb_name": yaml_cfg.env.wandb_name,
    }
    
    return TestConfig(**cfg_dict)
