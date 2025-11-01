"""
Training callback for accuracy evaluation during training.
"""

import os
from typing import Optional
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb

from data import load_split
from evaluate import extract_answer_from_output, compare_answers
from datasets import load_dataset


class AccuracyEvaluationCallback(TrainerCallback):
    """
    Callback that evaluates model accuracy on a test dataset at specified intervals.
    """
    
    def __init__(
        self,
        eval_dataset_name: str,
        join_string: str,  # Full eval join string (question -> reasoning -> answer)
        answer_regex: str,  # Regex pattern to extract answer from output
        eval_split: str = "test",
        eval_max_examples: Optional[int] = None,
        max_new_tokens: int = 200,
        eval_interval_steps: int = 1000,
        device: str = "auto",
    ):
        """
        Initialize the accuracy evaluation callback.
        
        Args:
            eval_dataset_name: Dataset name for evaluation (e.g., "gsm8k")
            eval_split: Split to use for evaluation (e.g., "test")
            eval_max_examples: Maximum number of examples to evaluate (None for all)
            join_string: Full join string for evaluation prompts (question -> reasoning -> answer)
            answer_regex: Regex pattern to extract answer from model output (should have capture group)
            max_new_tokens: Maximum tokens to generate during evaluation
            eval_interval_steps: Evaluate every N training steps
            device: Device to run evaluation on
        """
        self.eval_dataset_name = eval_dataset_name
        self.eval_split = eval_split
        self.eval_max_examples = eval_max_examples
        self.join_string = join_string  # Full eval join string
        self.answer_regex = answer_regex  # Regex pattern for extracting answer
        self.max_new_tokens = max_new_tokens
        self.eval_interval_steps = eval_interval_steps
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = None  # Will be set when trainer is available
        
        # Load evaluation dataset once
        print(f"\n📊 Loading Evaluation Dataset: {eval_dataset_name} ({eval_split})")
        # For evaluation, load raw dataset to get just questions (not question+reasoning)
        self.eval_dataset = self._load_eval_dataset(eval_dataset_name, eval_split, eval_max_examples)
        print(f"✅ Loaded {len(self.eval_dataset)} examples for evaluation")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Store trainer reference when training begins."""
        self.trainer = kwargs.get("trainer")
    
    def _load_eval_dataset(self, dataset_name: str, split: str, max_examples: Optional[int]):
        """
        Load dataset for evaluation. For GSM8K, extracts just the question (not question+reasoning).
        """
        if dataset_name == "gsm8k":
            # Load raw GSM8K dataset to get just questions
            split_str = f"{split}[:{max_examples}]" if max_examples else split
            raw_dataset = load_dataset("gsm8k", "main", split=split_str, streaming=False)
            
            # Extract question and answer
            def extract_question_and_answer(example):
                question = example["question"].strip()
                full_answer = example["answer"].strip()
                # Extract final answer after ####
                import re
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
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Evaluate accuracy at specified step intervals."""
        # Check if we should evaluate at this step
        if state.global_step % self.eval_interval_steps != 0:
            return
        
        # Don't evaluate at step 0
        if state.global_step == 0:
            return
        
        # Get trainer - prefer from kwargs, fallback to stored reference
        trainer = kwargs.get("trainer") or self.trainer
        if trainer is None:
            print(f"⚠️  Skipping evaluation at step {state.global_step}: trainer not available")
            return
        
        # Get model and tokenizer from trainer
        model = trainer.model
        tokenizer = trainer.tokenizer or trainer.processing_class
        
        if model is None or tokenizer is None:
            print(f"⚠️  Skipping evaluation at step {state.global_step}: model or tokenizer not available")
            return
        
        print(f"\n📊 Evaluating accuracy at step {state.global_step}...")
        
        # Move model to eval mode
        was_training = model.training
        model.eval()
        
        correct = 0
        total = 0
        
        # Evaluate on subset of dataset for speed (sample up to eval_max_examples)
        eval_subset = self.eval_dataset
        if self.eval_max_examples and len(eval_subset) > self.eval_max_examples:
            import random
            random.seed(42)  # Reproducible sampling
            eval_subset = eval_subset.select(random.sample(range(len(eval_subset)), self.eval_max_examples))
        
        with torch.no_grad():
            for i, example in enumerate(eval_subset):
                # For GSM8K, use just the question; for others, extract question from text
                if "question" in example:
                    # Raw dataset format (e.g., GSM8K)
                    question = example["question"]
                    ground_truth = example.get("answer", "")
                else:
                    # Preprocessed format - extract question (first line before reasoning)
                    text = example.get("text", "")
                    # For GSM8K preprocessed: question is before first "\n\n"
                    # For other datasets, use full text
                    if self.eval_dataset_name == "gsm8k" and "\n\n" in text:
                        question = text.split("\n\n")[0]
                    else:
                        question = text
                    ground_truth = example.get("answer", "")
                
                if not ground_truth:
                    continue  # Skip examples without answers
                
                # Format prompt: question only + join_string (model will generate reasoning + answer)
                prompt = question + self.join_string
                
                # Tokenize input
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Generate prediction
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,  # Greedy decoding for accuracy
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the generated part (after input)
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                if generated_text.startswith(input_text):
                    generated_text = generated_text[len(input_text):].strip()
                
                # Extract answer using regex pattern
                full_output = prompt + generated_text
                predicted_answer = extract_answer_from_output(self.answer_regex, full_output)
                
                # Compare answers
                is_correct = compare_answers(predicted_answer, ground_truth)
                
                if is_correct:
                    correct += 1
                total += 1
        
        # Calculate accuracy
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        # Log to wandb
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/correct": correct,
            "eval/total": total,
            "train/global_step": state.global_step,
        })
        
        print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Resume training mode
        if was_training:
            model.train()
        
        return control

