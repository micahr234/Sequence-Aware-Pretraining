import os
import sys

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import huggingface_hub
import wandb

from config import Config
from data import load_split
from utils import set_seed
from discounted_sft_trainer import DiscountedLogSuffixSFTTrainer


def train(cfg: Config):
    """
    Main training function using Discounted Log-Suffix SFT trainer.
    
    Args:
        cfg: Configuration object containing all training parameters
    """
    print("🚀 Starting Discounted Log-Suffix SFT Training")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Login to Hugging Face
    hf_token = os.getenv(cfg.hf_token_env)
    huggingface_hub.login(token=hf_token)
    print("🔐 Authenticated with Hugging Face")

    # Login to Weights & Biases
    wandb_token = os.getenv(cfg.wandb_token_env)
    wandb.login(key=wandb_token, verify=True)
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, id=wandb.util.generate_id(), resume="allow")
    print("📊 Authenticated with Weights & Biases")

    # Load dataset
    print(f"\n📊 Loading Dataset: {cfg.dataset}")
    dataset = load_split(cfg.dataset, "train", cfg.max_examples)

    # Load model and tokenizer
    print(f"\n📥 Loading Model: {cfg.base_model_name} on {cfg.device}")

    print(f"📥 Loading Model: {cfg.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if cfg.device != "cpu" else None,
    )

    # Set dropout if specified
    if cfg.dropout is not None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = cfg.dropout
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set tokenizer max length
    tokenizer.model_max_length = cfg.max_seq_length

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.scheduler_type,
        logging_steps=cfg.log_every,
        save_steps=cfg.save_every,
        fp16=torch.cuda.is_available() and cfg.device != "cpu",
        bf16=False,
        report_to=cfg.wandb_project,
        remove_unused_columns=False,
    )

    # Create trainer with discounted log-suffix weighting
    trainer = DiscountedLogSuffixSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        packing=False,  # Keep sequences un-packed for position-based weighting
        max_seq_length=cfg.max_seq_length,
        gamma=cfg.gamma,  # Discount factor for position weighting
    )

    print(f"\n🎯 Starting Training with {len(dataset)} examples")
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving Final Model...")
    trainer.save_model()
    
    # Save model to Hugging Face if configured
    if cfg.output_model_name:
        print(f"\n💾 Saving Final Model to Hugging Face...")
        print(f"Model name: {cfg.output_model_name}")
        trainer.model.push_to_hub(cfg.output_model_name)
