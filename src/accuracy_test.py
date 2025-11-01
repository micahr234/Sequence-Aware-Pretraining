"""
Accuracy evaluation test script.
"""

import sys
import os
from pathlib import Path

from config import AccuracyTestConfig, load_train_config
from evaluate import AccuracyEvaluator


def load_accuracy_test_config(config_path: str) -> AccuracyTestConfig:
    """
    Load accuracy test configuration.
    Can either use a dedicated test config or derive from training config.
    
    Args:
        config_path: Path to test configuration YAML file
        
    Returns:
        AccuracyTestConfig object
    """
    from omegaconf import OmegaConf
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    yaml_cfg = OmegaConf.load(config_path)
    
    # Check if this is a test config with accuracy_test section
    if "accuracy_test" in yaml_cfg:
        # Dedicated accuracy test config
        cfg = yaml_cfg.accuracy_test
        model_hub_id = OmegaConf.select(cfg, "model_hub_id", default=OmegaConf.select(yaml_cfg.model, "name", default=""))
        device = OmegaConf.select(cfg, "device", default="auto")
        dataset = OmegaConf.select(cfg, "dataset", default="gsm8k")
        test_split = OmegaConf.select(cfg, "test_split", default="test")
        max_examples = OmegaConf.select(cfg, "max_examples", default=None)
        join_string = OmegaConf.select(cfg, "join_string", default="\n\nThe answer is: ")
        answer_regex = OmegaConf.select(cfg, "answer_regex", default=None)
        max_new_tokens = OmegaConf.select(cfg, "max_new_tokens", default=200)
        temperature = OmegaConf.select(cfg, "temperature", default=0.0)
        top_p = OmegaConf.select(cfg, "top_p", default=1.0)
        top_k = OmegaConf.select(cfg, "top_k", default=50)
        do_sample = OmegaConf.select(cfg, "do_sample", default=False)
        output_dir = OmegaConf.select(cfg, "output_dir", default="test_results/accuracy")
        seed = OmegaConf.select(cfg, "seed", default=42)
        hf_token_env = OmegaConf.select(cfg, "hf_token_env", default=OmegaConf.select(yaml_cfg.env, "hf_token_env", default=None))
        wandb_token_env = OmegaConf.select(cfg, "wandb_token_env", default=OmegaConf.select(yaml_cfg.env, "wandb_token_env", default=None))
    else:
        # Derive from training config - load the training config to get model name
        train_config_path = config_path.replace("test_configs", "train_configs")
        if os.path.exists(train_config_path):
            train_cfg = load_train_config(train_config_path)
            model_hub_id = train_cfg.output_model_name
            # Construct eval join string from training join strings
            join_string = train_cfg.question_reasoning_join_string + train_cfg.reasoning_answer_join_string
            # Construct default answer regex from reasoning_answer_join_string
            import re
            escaped = re.escape(train_cfg.reasoning_answer_join_string)
            answer_regex = escaped + r'(.+?)(?:\n|$)'
        else:
            # Fallback to test config model name
            model_hub_id = OmegaConf.select(yaml_cfg.model, "name", default="")
            join_string = OmegaConf.select(yaml_cfg, "join_string", default="\n\nThe answer is: ")
            answer_regex = OmegaConf.select(yaml_cfg, "answer_regex", default=None)
        
        # Use test config for other parameters
        device = OmegaConf.select(yaml_cfg.model, "device", default="auto")
        dataset = OmegaConf.select(yaml_cfg, "dataset", default="gsm8k")
        test_split = OmegaConf.select(yaml_cfg, "test_split", default="test")
        max_examples = OmegaConf.select(yaml_cfg, "max_examples", default=None)
        max_new_tokens = OmegaConf.select(yaml_cfg.generation, "max_new_tokens", default=200)
        temperature = OmegaConf.select(yaml_cfg.generation, "temperature", default=0.0)
        top_p = OmegaConf.select(yaml_cfg.generation, "top_p", default=1.0)
        top_k = OmegaConf.select(yaml_cfg.generation, "top_k", default=50)
        do_sample = OmegaConf.select(yaml_cfg.generation, "do_sample", default=False)
        output_dir = OmegaConf.select(yaml_cfg.analysis, "output_dir", default="test_results/accuracy")
        seed = OmegaConf.select(yaml_cfg.experiment, "seed", default=42)
        hf_token_env = OmegaConf.select(yaml_cfg.env, "hf_token_env", default=None)
        wandb_token_env = OmegaConf.select(yaml_cfg.env, "wandb_token_env", default=None)
        # If not set above, set default regex
        if 'answer_regex' not in locals():
            answer_regex = None
    
    # Handle device auto-detection
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return AccuracyTestConfig(
        model_hub_id=model_hub_id,
        device=device,
        dataset=dataset,
        test_split=test_split,
        max_examples=max_examples,
        join_string=join_string,
        answer_regex=answer_regex,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        output_dir=output_dir,
        seed=seed,
        hf_token_env=hf_token_env,
        wandb_token_env=wandb_token_env,
    )


def main():
    """Main function for accuracy evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on test dataset")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--model", help="Override model hub ID")
    parser.add_argument("--max-examples", type=int, help="Override max examples")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_accuracy_test_config(args.config)
    
    # Apply command line overrides
    if args.model:
        config.model_hub_id = args.model
    if args.max_examples:
        config.max_examples = args.max_examples
    
    print("🎯 Starting Accuracy Evaluation")
    print("=" * 60)
    print(f"Model: {config.model_hub_id}")
    print(f"Dataset: {config.dataset} ({config.test_split})")
    print(f"Join String: {repr(config.join_string)}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    # Create evaluator and run evaluation
    try:
        evaluator = AccuracyEvaluator(config)
        results = evaluator.evaluate_dataset()
        
        print("\n✅ Accuracy evaluation completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

