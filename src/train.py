import os
import re

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
from evaluation_callback import AccuracyEvaluationCallback


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
    is_single_device = device in ["cpu", "cuda"] or (isinstance(device, str) and device.startswith("cuda:"))
    device_map_kwarg = None if is_single_device else device
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        torch_dtype=torch.float32,
        device_map=device_map_kwarg,
    )
    if device_map_kwarg is None:
        model = model.to(device)

    if cfg.dropout is not None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = cfg.dropout
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
        "hub_private_repo": True,
    }
    
    if cfg.warmup_steps is not None:
        training_args_kwargs["warmup_steps"] = cfg.warmup_steps
    if cfg.warmup_ratio is not None:
        training_args_kwargs["warmup_ratio"] = cfg.warmup_ratio
    if cfg.optimizer_type is not None:
        training_args_kwargs["optim"] = cfg.optimizer_type
    
    training_args = TrainingArguments(**training_args_kwargs)

    data_collator = AttentionMaskDataCollator(
        tokenizer=tokenizer, 
        mlm=False,
        train_on_answers_only=cfg.train_on_answers_only,
        question_reasoning_join_string=cfg.question_reasoning_join_string,
        reasoning_answer_join_string=cfg.reasoning_answer_join_string,
    )
    
    trainer = DiscountedLogSuffixSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
        gamma=cfg.gamma,
    )
    
    if cfg.eval_dataset and cfg.eval_interval_steps:
        print(f"\n📊 Setting up accuracy evaluation:")
        print(f"   Dataset: {cfg.eval_dataset} ({cfg.eval_split or 'test'})")
        print(f"   Interval: Every {cfg.eval_interval_steps} steps")
        print(f"   Max examples: {cfg.eval_max_examples or 'all'}")
        
        eval_join_string = cfg.question_reasoning_join_string + cfg.reasoning_answer_join_string
        
        if cfg.eval_answer_regex is not None:
            answer_regex = cfg.eval_answer_regex
        else:
            escaped = re.escape(cfg.reasoning_answer_join_string)
            answer_regex = escaped + r'(.+?)(?:\n|$)'
        
        eval_callback = AccuracyEvaluationCallback(
            eval_dataset_name=cfg.eval_dataset,
            join_string=eval_join_string,
            answer_regex=answer_regex,
            eval_split=cfg.eval_split or "test",
            eval_max_examples=cfg.eval_max_examples,
            max_new_tokens=200,
            eval_interval_steps=cfg.eval_interval_steps,
            device=device,
        )
        trainer.add_callback(eval_callback)

    print("\n🎯 Starting Training")
    trainer.train()
    
    print("\n💾 Saving Final Model...")
    trainer.save_model()
