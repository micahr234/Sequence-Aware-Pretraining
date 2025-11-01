#!/usr/bin/env python3
"""
Training script for Sequence-Aware Pretraining.

Usage:
    python scripts/train.py [config_file]
    
Examples:
    python scripts/train.py                           # Use default config
    python scripts/train.py train_configs/baseline.yaml  # Use custom config
    
Requirements:
    - transformers: pip install transformers
    - datasets: pip install datasets
    - torch: pip install torch
    - CUDA-compatible GPU recommended for best performance

Note: Uses Discounted Log-Suffix SFT training for sequence-aware pretraining.
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*", category=UserWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import train
from config import load_config

def main():
    """Main training function."""
    # Allow config file to be specified as command line argument
    config_path = "train_configs/default.yaml"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    
    # Print config for verification
    print("Configuration:")
    for key, value in cfg.__dict__.items():
        print(f"  {key}: {value}")
    print()
    
    # Start training
    train(cfg)

if __name__ == "__main__":
    main()
