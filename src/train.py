import os
from typing import List, Dict, Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments
import huggingface_hub
import wandb

from config import Config
from data import load_split
from utils import set_seed
from discounted_sft_trainer import WeightedSFTTrainer
from data_collator import TrainingCollator, EvalDataCollator




def train(cfg: Config):
    """
    Main training function using Weighted SFT trainer.
    
    Args:
        cfg: Configuration object containing all training parameters
    """
    print("🚀 Starting Weighted SFT Training")
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

    # Load train dataset
    print(f"\n📊 Loading Dataset: {cfg.dataset}")
    train_dataset = load_split(cfg.dataset, "train", cfg.max_examples)

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
    
    # Add evaluation settings if evaluation is enabled
    if cfg.eval_dataset and cfg.eval_interval_steps:
        training_args_kwargs["eval_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = cfg.eval_interval_steps
        training_args_kwargs["predict_with_generate"] = True
    
    if cfg.warmup_steps is not None:
        training_args_kwargs["warmup_steps"] = cfg.warmup_steps
    if cfg.warmup_ratio is not None:
        training_args_kwargs["warmup_ratio"] = cfg.warmup_ratio
    if cfg.optimizer_type is not None:
        training_args_kwargs["optim"] = cfg.optimizer_type
    
    # Use Seq2SeqTrainingArguments to support predict_with_generate
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    data_collator = TrainingCollator(
        tokenizer=tokenizer, 
        mlm=False,
        train_on_answers_only=cfg.train_on_answers_only,
        question_template=cfg.question_template,
        answer_template=cfg.answer_template,
    )
    
    # Prepare evaluation dataset if evaluation is enabled
    eval_dataset = None
    eval_data_collator = None
    gold_answers = None
    prompts_list = None
    answer_regex = None
    
    if cfg.eval_dataset and cfg.eval_interval_steps:
        print(f"\n📊 Setting up accuracy evaluation:")
        print(f"   Dataset: {cfg.eval_dataset} ({cfg.eval_split or 'test'})")
        print(f"   Interval: Every {cfg.eval_interval_steps} steps")
        print(f"   Max examples: {cfg.eval_max_examples or 'all'}")
        
        # Load evaluation dataset (keep as raw text like train_dataset)
        print(f"   Loading evaluation dataset...")
        eval_dataset = load_split(
            cfg.eval_dataset,
            cfg.eval_split or "test",
            cfg.eval_max_examples
        )
        print(f"   ✅ Loaded {len(eval_dataset)} examples for evaluation")
        
        # Create eval data collator that will tokenize on-the-fly
        # For decoder-only models, use left padding for generation
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        eval_data_collator = EvalDataCollator(tokenizer, cfg.question_template)
        
        # Extract gold_answers and prompts_list by processing the dataset once
        # This ensures they're available for compute_metrics in the right order
        print(f"   Extracting gold answers and prompts...")
        gold_answers = []
        prompts_list = []
        for i in range(len(eval_dataset)):
            example = eval_dataset[i]
            question = example["question"]
            ground_truth = example["answer"]
            
            # Format prompt using template (same as collator does)
            formatted_question = cfg.question_template.format(question=question)
            
            gold_answers.append(ground_truth)
            prompts_list.append(formatted_question)
        
        print(f"   ✅ Extracted {len(gold_answers)} gold answers and prompts")
        
        # Restore original padding side (eval collator will use left padding internally)
        tokenizer.padding_side = original_padding_side

        if hasattr(model, 'generation_config'):
            model.generation_config.max_new_tokens = 200
            model.generation_config.do_sample = False
            model.generation_config.temperature = 0.0
            if not hasattr(model.generation_config, 'max_length') or model.generation_config.max_length is None:
                model.generation_config.max_length = tokenizer.model_max_length
    
    # Create wrapper data collator that uses different collators for train vs eval
    if eval_data_collator is not None:
        class DualDataCollator:
            def __init__(self, train_collator, eval_collator):
                self.train_collator = train_collator
                self.eval_collator = eval_collator
            
            def __call__(self, features):
                if features and isinstance(features[0], dict):
                    # Check if it's train data (has reasoning field - train dataset specific)
                    if "reasoning" in features[0]:
                        return self.train_collator(features)
                    # Check if it's eval data (has question/answer fields, no input_ids yet)
                    elif ("question" in features[0] and "answer" in features[0]) and "input_ids" not in features[0]:
                        return self.eval_collator(features)
                # Default to train collator if uncertain
                return self.train_collator(features)
        
        data_collator = DualDataCollator(data_collator, eval_data_collator)
    
    trainer = WeightedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        eval_gold_answers=gold_answers,
        eval_prompts_list=prompts_list,
        eval_answer_regex=cfg.eval_answer_regex,
    )

    print("\n🎯 Starting Training")
    trainer.train()
    
    print("\n💾 Saving Final Model...")
    trainer.save_model()
