import os
import re

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding
import huggingface_hub
import wandb
from datasets import load_dataset as hf_load_dataset

from config import Config
from data import load_split
from utils import set_seed
from discounted_sft_trainer import DiscountedLogSuffixSFTTrainer
from data_collator import AttentionMaskDataCollator
from evaluate import extract_answer_from_output, compare_answers


def _load_eval_dataset(dataset_name: str, split: str, max_examples: int = None):
    """
    Load dataset for evaluation. For GSM8K, extracts just the question (not question+reasoning).
    
    Args:
        dataset_name: Dataset name (e.g., "gsm8k")
        split: Dataset split (e.g., "test")
        max_examples: Maximum number of examples to load (None for all)
    
    Returns:
        Dataset with "question" and "answer" fields
    """
    if dataset_name == "gsm8k":
        # Load raw GSM8K dataset to get just questions
        split_str = f"{split}[:{max_examples}]" if max_examples else split
        raw_dataset = hf_load_dataset("gsm8k", "main", split=split_str, streaming=False)
        
        # Extract question and answer
        def extract_question_and_answer(example):
            question = example["question"].strip()
            full_answer = example["answer"].strip()
            # Extract final answer after ####
            marker_match = re.search(r"([\s\S]*?)\s*####\s*(.+)", full_answer)
            if marker_match:
                answer = marker_match.group(2).replace(",", "").strip()
            else:
                answer = ""
            return {"question": question, "answer": answer}
        
        # Map to extract question and answer, keeping both fields
        dataset = raw_dataset.map(extract_question_and_answer)
        return dataset
    else:
        # For other datasets, use standard load_split but we'll use just the question part
        dataset = load_split(dataset_name, split, max_examples)
        # For datasets where text contains question+context, try to extract just question
        # This is a heuristic - may need dataset-specific handling
        return dataset


def _prepare_eval_dataset(eval_dataset, tokenizer, join_string, eval_dataset_name: str):
    """
    Prepare evaluation dataset: format prompts and tokenize.
    
    Args:
        eval_dataset: Dataset with "question" and "answer" fields
        tokenizer: Tokenizer to use
        join_string: Join string to append after question (e.g., question_reasoning_join_string)
        eval_dataset_name: Name of the eval dataset (for dataset-specific handling)
    
    Returns:
        Tuple of (tokenized_dataset, gold_answers_list, prompts_list) where:
        - tokenized_dataset: Dataset with tokenized inputs (input_ids, attention_mask)
        - gold_answers_list: List of gold answer strings in the same order
        - prompts_list: List of original prompt strings (for reconstructing full output)
    
    Raises:
        ValueError: If any example in the dataset is missing an answer
    """
    gold_answers = []
    prompts_list = []
    
    def build_inputs(batch):
        """Format prompts and tokenize, keeping gold answers and prompts."""
        questions = []
        batch_answers = []
        batch_prompts = []
        
        batch_length = len(batch["question"] if "question" in batch else batch["text"])
        
        for i in range(batch_length):
            # Get question
            if "question" in batch:
                question = batch["question"][i]
                ground_truth = batch.get("answer", [""] * batch_length)[i] if batch.get("answer") else ""
            else:
                # Preprocessed format - extract question (first line before reasoning)
                text = batch.get("text", [""])[i]
                # For GSM8K preprocessed: question is before first "\n\n"
                # For other datasets, use full text
                if eval_dataset_name == "gsm8k" and "\n\n" in text:
                    question = text.split("\n\n")[0]
                else:
                    question = text
                ground_truth = batch.get("answer", [""])[i] if batch.get("answer") else ""
            
            # Validate that answer exists - raise error if missing
            if not ground_truth or (isinstance(ground_truth, str) and len(ground_truth.strip()) == 0):
                raise ValueError(
                    f"Evaluation dataset example at index {i} is missing an answer. "
                    f"All examples in the evaluation dataset must have answers."
                )
            
            # Format prompt: question only + join_string (model will generate reasoning + answer)
            prompt = question + join_string
            questions.append(prompt)
            batch_answers.append(ground_truth)
            batch_prompts.append(prompt)
        
        # Tokenize prompts
        enc = tokenizer(
            questions,
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )
        
        # Store gold answers and prompts in closure
        gold_answers.extend(batch_answers)
        prompts_list.extend(batch_prompts)
        
        return enc
    
    # Tokenize the dataset
    # Remove all columns including "answer" since we store it separately in gold_answers
    tokenized_dataset = eval_dataset.map(
        build_inputs,
        batched=True,
        remove_columns=eval_dataset.column_names,  # Remove all original columns
        desc="Preparing eval dataset"
    )
    
    return tokenized_dataset, gold_answers, prompts_list


def _create_compute_metrics(tokenizer, gold_answers, prompts_list, answer_regex):
    """
    Create compute_metrics function that uses generated texts.
    
    Args:
        tokenizer: Tokenizer for decoding
        gold_answers: List of gold answer strings (must match order of eval dataset)
        prompts_list: List of original prompt strings (for reconstructing full output)
        answer_regex: Regex pattern to extract answer from model output
    
    Returns:
        compute_metrics function that can be passed to Trainer
    """
    def compute_metrics(eval_pred):
        """
        Compute accuracy metrics from generated predictions.
        
        Args:
            eval_pred: EvalPrediction object with predictions (generated token IDs only, input was stripped)
        
        Returns:
            Dictionary with accuracy and regex_coverage metrics
        """
        print(f"\n🔍 Debug: compute_metrics called")
        print(f"   eval_pred type: {type(eval_pred)}")
        print(f"   eval_pred attributes: {dir(eval_pred)}")
        
        preds = eval_pred.predictions
        print(f"   preds type: {type(preds)}, shape: {getattr(preds, 'shape', 'no shape')}")
        
        # HF sometimes returns a tuple; make sure we grab the array of IDs
        if isinstance(preds, tuple):
            preds = preds[0]
            print(f"   After tuple unwrap: preds type: {type(preds)}, shape: {getattr(preds, 'shape', 'no shape')}")
        
        # Handle numpy arrays
        if hasattr(preds, 'shape'):
            print(f"   preds shape: {preds.shape}")
            if len(preds.shape) == 1:
                # Single dimension array, wrap in list
                total = 1
                preds = [preds]
            else:
                total = preds.shape[0]
        else:
            total = len(preds) if preds else 0
        
        print(f"   Total examples: {total}")
        
        # preds now contains only the generated tokens (input was stripped in prediction_step)
        correct = 0
        y_pred = []
        
        for i, generated_ids in enumerate(preds):
            # Decode the generated tokens (already extracted from full sequence in prediction_step)
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Reconstruct full output: prompt + generated text
            prompt = prompts_list[i] if i < len(prompts_list) else ""
            full_output = prompt + gen_text
            
            # Extract answer from full output using regex
            predicted_answer = extract_answer_from_output(answer_regex, full_output)
            gold = gold_answers[i] if i < len(gold_answers) else ""
            
            y_pred.append(predicted_answer)
            if predicted_answer is not None and compare_answers(predicted_answer, gold):
                correct += 1
        
        acc = correct / max(1, total)
        # If you want to inspect coverage of the regex:
        coverage = sum(p is not None for p in y_pred) / max(1, total)
        
        # Log accuracy to console
        acc_percent = acc * 100
        print(f"\n📊 Evaluation Results:")
        print(f"   Accuracy: {acc_percent:.2f}% ({correct}/{total})")
        print(f"   Regex Coverage: {coverage*100:.2f}% ({sum(p is not None for p in y_pred)}/{total})")
        
        # Return metrics - Trainer automatically adds "eval/" prefix when logging to wandb
        # So return keys without prefix to avoid "eval/eval/accuracy"
        metrics = {
            "accuracy": acc,
            "accuracy_percent": acc_percent,
            "regex_coverage": coverage,
            "correct": correct,
            "total": total,
        }
        
        print(f"🔍 Debug: Returning metrics: {metrics}")
        
        # Explicitly log to wandb with eval/ prefix to ensure visibility
        # Trainer should do this automatically, but adding explicit logging as backup
        try:
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(eval_metrics)
            print(f"✅ Metrics logged to wandb: {list(eval_metrics.keys())}")
        except Exception as e:
            print(f"⚠️  Failed to log to wandb: {e}")
        
        return metrics
    
    return compute_metrics


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
    
    # Add evaluation settings if evaluation is enabled
    if cfg.eval_dataset and cfg.eval_interval_steps:
        training_args_kwargs["eval_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = cfg.eval_interval_steps
        training_args_kwargs["predict_with_generate"] = True
        # Don't set generation_max_length here - it conflicts with max_new_tokens
        # We'll set max_new_tokens in model.generation_config instead
    
    if cfg.warmup_steps is not None:
        training_args_kwargs["warmup_steps"] = cfg.warmup_steps
    if cfg.warmup_ratio is not None:
        training_args_kwargs["warmup_ratio"] = cfg.warmup_ratio
    if cfg.optimizer_type is not None:
        training_args_kwargs["optim"] = cfg.optimizer_type
    
    # Use Seq2SeqTrainingArguments to support predict_with_generate
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    data_collator = AttentionMaskDataCollator(
        tokenizer=tokenizer, 
        mlm=False,
        train_on_answers_only=cfg.train_on_answers_only,
        question_reasoning_join_string=cfg.question_reasoning_join_string,
        reasoning_answer_join_string=cfg.reasoning_answer_join_string,
    )
    
    # Prepare evaluation dataset if evaluation is enabled
    eval_dataset = None
    compute_metrics_fn = None
    eval_data_collator = None
    
    if cfg.eval_dataset and cfg.eval_interval_steps:
        print(f"\n📊 Setting up accuracy evaluation:")
        print(f"   Dataset: {cfg.eval_dataset} ({cfg.eval_split or 'test'})")
        print(f"   Interval: Every {cfg.eval_interval_steps} steps")
        print(f"   Max examples: {cfg.eval_max_examples or 'all'}")
        
        # Load evaluation dataset
        print(f"   Loading evaluation dataset...")
        raw_eval_dataset = _load_eval_dataset(
            cfg.eval_dataset,
            cfg.eval_split or "test",
            cfg.eval_max_examples
        )
        print(f"   ✅ Loaded {len(raw_eval_dataset)} examples for evaluation")
        
        # Prepare evaluation dataset (format prompts, tokenize)
        print(f"   Preparing evaluation dataset...")
        eval_join_string = cfg.question_reasoning_join_string  # Just question_reasoning_join_string
        
        answer_regex = cfg.eval_answer_regex
        
        eval_dataset, gold_answers, prompts_list = _prepare_eval_dataset(
            raw_eval_dataset,
            tokenizer,
            eval_join_string,
            cfg.eval_dataset
        )
        print(f"   ✅ Prepared evaluation dataset")
        
        # Create compute_metrics function
        compute_metrics_fn = _create_compute_metrics(
            tokenizer,
            gold_answers,
            prompts_list,
            answer_regex
        )
        
        # Create eval data collator - eval dataset is already tokenized, just need padding
        # For decoder-only models, use left padding for generation
        # Set tokenizer to left padding for evaluation (needed for generation)
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        # Create a tokenizer copy for the eval collator to ensure left padding
        eval_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Keep left padding for evaluation (it will be used by Seq2SeqTrainer during eval)
        # We'll restore it after trainer creation if needed, but left padding is correct for generation
    
    # Configure generation parameters for evaluation if enabled
    if cfg.eval_dataset and cfg.eval_interval_steps:
        # Set generation config on model for predict_with_generate
        if hasattr(model, 'generation_config'):
            # Set max_new_tokens for generation (preferred over max_length)
            model.generation_config.max_new_tokens = 200
            model.generation_config.do_sample = False
            model.generation_config.temperature = 0.0
            # Ensure max_length is set to a valid value (not None)
            # Seq2SeqTrainer's prediction_step checks max_length even if max_new_tokens is used
            # Set it to a high value - max_new_tokens will take precedence for actual generation
            if not hasattr(model.generation_config, 'max_length') or model.generation_config.max_length is None:
                # Set to a reasonable default (input_length + max_new_tokens estimate)
                model.generation_config.max_length = tokenizer.model_max_length
    
    # Use eval_data_collator for eval, data_collator for train
    # Seq2SeqTrainer doesn't have eval_data_collator parameter, so we'll create a wrapper
    if eval_data_collator is not None:
        # Create a wrapper data collator that uses different collators for train vs eval
        class DualDataCollator:
            def __init__(self, train_collator, eval_collator):
                self.train_collator = train_collator
                self.eval_collator = eval_collator
                self.is_training = True
            
            def __call__(self, features):
                # Check if features are already tokenized (have input_ids) or raw (have text)
                if features and isinstance(features[0], dict):
                    if "input_ids" in features[0] and "text" not in features[0]:
                        # Tokenized data - use eval collator
                        return self.eval_collator(features)
                    elif "text" in features[0]:
                        # Raw text data - use train collator
                        return self.train_collator(features)
                # Default to eval collator if uncertain (safer for tokenized data)
                return self.eval_collator(features) if features and isinstance(features[0], dict) and "input_ids" in features[0] else self.train_collator(features)
        
        data_collator = DualDataCollator(data_collator, eval_data_collator)
    
    trainer = DiscountedLogSuffixSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        gamma=cfg.gamma,
    )

    print("\n🎯 Starting Training")
    trainer.train()
    
    print("\n💾 Saving Final Model...")
    trainer.save_model()
