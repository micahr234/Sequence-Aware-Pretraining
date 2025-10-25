"""
Utility functions for testing framework.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_prompts_from_file(file_path: str) -> list:
    """
    Load prompts from a text file (one prompt per line).
    
    Args:
        file_path: Path to the prompts file
        
    Returns:
        List of prompts
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    return prompts


def create_test_prompts_file(output_path: str = "test_prompts.txt"):
    """
    Create a sample prompts file for testing.
    
    Args:
        output_path: Path to save the prompts file
    """
    sample_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where artificial intelligence",
        "Once upon a time, in a distant galaxy",
        "The weather today is sunny and warm",
        "Scientists have discovered a new species",
        "The future of technology looks promising",
        "In the depths of the ocean, there lives",
        "The ancient library contained many secrets",
        "As the sun sets over the mountains",
        "The robot learned to understand human emotions"
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in sample_prompts:
            f.write(prompt + '\n')
    
    print(f"Created test prompts file: {output_path}")


def ensure_output_directory(output_dir: str):
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")


def print_experiment_info(config):
    """Print experiment information."""
    print("🧪 Probability Analysis Test")
    print("=" * 50)
    print(f"📊 Model: {config.model_name}")
    print(f"🎯 Device: {config.device}")
    print(f"📝 Prompt: {config.prompt[:50]}...")
    print(f"🔄 Samples: {config.num_samples}")
    print(f"📏 Max Tokens: {config.max_new_tokens}")
    print(f"🌡️ Temperature: {config.temperature}")
    print(f"🎲 Top-p: {config.top_p}")
    print(f"🔝 Top-k: {config.top_k}")
    print(f"📁 Output: {config.output_dir}")
    print("=" * 50)
