import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import huggingface_hub
import wandb

from config import Config
from data import load_split
from utils import set_seed
from discounted_sft_trainer import DiscountedLogSuffixSFTTrainer
from data_collator import AttentionMaskDataCollator


def train(cfg: Config):
    """
    Main training function using Discounted Log-Suffix SFT trainer.
    
    Args:
        cfg: Configuration object containing all training parameters
    """
    print("🚀 Starting Discounted Log-Suffix SFT Training")
    print("=" * 60)
    
    set_seed(cfg.seed)
    
    # Authenticate with Hugging Face
    hf_token = os.getenv(cfg.hf_token_env) if cfg.hf_token_env else None
    if not hf_token:
        raise ValueError(f"Hugging Face token not found. Please set the environment variable '{cfg.hf_token_env or 'HF_TOKEN'}'")
    huggingface_hub.login(token=hf_token)
    print("🔐 Authenticated with Hugging Face")

    # Authenticate with Weights & Biases
    wandb_token = os.getenv(cfg.wandb_token_env) if cfg.wandb_token_env else None
    if not wandb_token:
        raise ValueError(f"Weights & Biases token not found. Please set the environment variable '{cfg.wandb_token_env or 'WANDB_TOKEN'}'")
    wandb.login(key=wandb_token, verify=True)
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, id=wandb.util.generate_id(), resume="allow")
    print("📊 Authenticated with Weights & Biases")

    # Load dataset
    print(f"\n📊 Loading Dataset: {cfg.dataset}")
    dataset = load_split(cfg.dataset, "train", cfg.max_examples)

    # Handle device auto-detection
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"\n📥 Loading Model: {cfg.base_model_name} on {device}")
    # For single device, use None for device_map and move model manually
    # device_map should only be used for multi-GPU setups or "auto" mode
    # For explicit single devices (cpu, cuda, cuda:0, etc.), load on CPU then move
    is_single_device = device in ["cpu", "cuda"] or (isinstance(device, str) and device.startswith("cuda:"))
    device_map_kwarg = None if is_single_device else device
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        torch_dtype=torch.float32,
        device_map=device_map_kwarg,
    )
    # If device_map was None, move model to the specified device
    if device_map_kwarg is None:
        model = model.to(device)

    if cfg.dropout is not None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = cfg.dropout
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model_max_length - raise error if it fails rather than silently continuing
    if not hasattr(tokenizer, 'model_max_length'):
        raise AttributeError(
            f"Tokenizer '{cfg.base_model_name}' does not have 'model_max_length' attribute. "
            f"This may indicate an incompatible tokenizer type."
        )
    
    if tokenizer.model_max_length != cfg.max_seq_length:
        try:
            tokenizer.model_max_length = cfg.max_seq_length
        except (AttributeError, TypeError) as e:
            raise RuntimeError(
                f"Failed to set tokenizer.model_max_length to {cfg.max_seq_length}. "
                f"Current value: {tokenizer.model_max_length}. "
                f"This may indicate that the tokenizer does not support modifying model_max_length. "
                f"Error: {e}"
            ) from e

    # Setup training arguments
    training_args_kwargs = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accumulation_steps,
        "learning_rate": cfg.lr,
        "num_train_epochs": cfg.num_epochs,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.scheduler_type,
        "logging_steps": cfg.log_every,
        "save_steps": cfg.save_every,
        "fp16": False,
        "bf16": False,
        "report_to": "wandb",
        "remove_unused_columns": False,
        "push_to_hub": True,
        "hub_model_id": cfg.output_model_name,
        "push_to_hub_private_repo": True,
    }
    
    # Handle warmup: if warmup_steps is 0 or None, use warmup_ratio; otherwise use warmup_steps
    # TrainingArguments will ignore warmup_ratio if warmup_steps is provided and > 0
    if cfg.warmup_steps is not None and cfg.warmup_steps > 0:
        training_args_kwargs["warmup_steps"] = cfg.warmup_steps
    elif cfg.warmup_ratio is not None:
        training_args_kwargs["warmup_ratio"] = cfg.warmup_ratio
    
    # Add optimizer type if specified
    if cfg.optimizer_type:
        training_args_kwargs["optim"] = cfg.optimizer_type
    
    training_args = TrainingArguments(**training_args_kwargs)

    # Create data collator and trainer
    # train_on_answers_only: If True, only train on answer portions (zero out loss for text)
    # When enabled, examples without answers are fully masked (no loss computed)
    data_collator = AttentionMaskDataCollator(
        tokenizer=tokenizer, 
        mlm=False,
        train_on_answers_only=cfg.train_on_answers_only
    )
    
    trainer = DiscountedLogSuffixSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
        gamma=cfg.gamma,
    )

    print("\n🎯 Starting Training")
    trainer.train()
    
    print("\n💾 Saving Final Model...")
    trainer.save_model()
