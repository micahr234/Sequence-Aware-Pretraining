"""
Accuracy evaluation module for testing trained models.
"""

import os
import re
import json
from typing import Optional, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub

from config import AccuracyTestConfig
from data import load_split
from utils import set_seed, ensure_output_directory


def extract_answer_from_output(answer_regex: str, output_text: str) -> Optional[str]:
    """
    Extract numerical answer from model output using regex pattern.
    
    Args:
        answer_regex: Regex pattern to extract answer (should have a named group "answer" or capture group)
        output_text: The complete model output text
        
    Returns:
        Extracted answer as string, or None if not found
    """
    match = re.search(answer_regex, output_text, re.DOTALL)
    if match:
        # Prefer named group "answer" if present, otherwise use first capture group, otherwise entire match
        if match.lastindex is not None:
            # Try named group first
            try:
                answer = match.group('answer')
            except (KeyError, IndexError):
                # Fall back to first capture group
                answer = match.group(1)
        else:
            # No capture groups, use entire match
            answer = match.group(0)
        answer = answer.strip()
        # Extract number (handle decimals, remove commas)
        answer_clean = answer.replace(',', '').strip()
        number_match = re.search(r'-?\d+\.?\d*', answer_clean)
        if number_match:
            return number_match.group(0)
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison (remove commas, handle decimals).
    
    Args:
        answer: Answer string to normalize
        
    Returns:
        Normalized answer string
    """
    # Remove commas and whitespace
    normalized = answer.replace(',', '').strip()
    # Try to parse as float to normalize decimals
    try:
        num = float(normalized)
        # Convert to int if it's a whole number, otherwise keep as float string
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return normalized


def compare_answers(predicted: Optional[str], ground_truth: str) -> bool:
    """
    Compare predicted answer with ground truth answer.
    
    Args:
        predicted: Predicted answer (may be None)
        ground_truth: Ground truth answer
        
    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False
    
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    return pred_norm == gt_norm


class AccuracyEvaluator:
    """
    Evaluator for model accuracy on question-answering tasks.
    """
    
    def __init__(self, config: AccuracyTestConfig):
        """
        Initialize the accuracy evaluator.
        
        Args:
            config: Accuracy test configuration
        """
        self.config = config
        self.device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        set_seed(config.seed)
        
        # Authenticate with Hugging Face if needed
        hf_token = os.getenv(config.hf_token_env) if config.hf_token_env else None
        if hf_token:
            huggingface_hub.login(token=hf_token)
            print("🔐 Authenticated with Hugging Face")
        
        # Load model and tokenizer
        print(f"\n📥 Loading Model: {config.model_hub_id} on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_hub_id,
            torch_dtype=torch.float32,
            device_map=self.device if self.device != "cpu" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_hub_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully")
    
    def generate_prediction(self, prompt: str) -> str:
        """
        Generate prediction for a given prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return only the generated part (after the input)
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if generated_text.startswith(input_text):
            generated_text = generated_text[len(input_text):].strip()
        
        return generated_text
    
    def evaluate_dataset(self) -> Dict:
        """
        Evaluate model on test dataset.
        
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n📊 Loading Test Dataset: {self.config.dataset} ({self.config.test_split})")
        
        # Use the same load_split() function as training for consistency
        dataset = load_split(
            self.config.dataset,
            self.config.test_split,
            self.config.max_examples
        )
        
        print(f"📝 Evaluating on {len(dataset)} examples")
        
        results = []
        correct = 0
        total = 0
        
        for i, example in enumerate(dataset):
            # Use just the question (not question + reasoning)
            # load_split() returns "text" and "answer" fields for all datasets
            text = example.get("text", "")
            ground_truth = example.get("answer", "")
            
            # Extract question from text (for GSM8K, it's before first "\n\n")
            if self.config.dataset == "gsm8k" and "\n\n" in text:
                question = text.split("\n\n")[0]
            else:
                question = text
            
            # Format prompt: question only + join_string (model will generate reasoning + answer)
            prompt = question + self.config.join_string
            
            # Generate prediction
            generated = self.generate_prediction(prompt)
            
            # Extract answer using regex pattern
            if self.config.answer_regex:
                answer_regex = self.config.answer_regex
            else:
                # Default: construct regex from reasoning_answer_join_string if available
                # Otherwise use default pattern
                if hasattr(self.config, 'reasoning_answer_join_string') and self.config.reasoning_answer_join_string:
                    escaped = re.escape(self.config.reasoning_answer_join_string)
                    answer_regex = escaped + r'(.+?)(?:\n|$)'
                else:
                    # Fallback to default pattern
                    answer_regex = r'The answer is:\s*(.+?)(?:\n|$)'
            predicted_answer = extract_answer_from_output(answer_regex, prompt + generated)
            
            # Compare answers
            is_correct = compare_answers(predicted_answer, ground_truth)
            
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                "index": i,
                "question": text,
                "ground_truth": ground_truth,
                "predicted": predicted_answer,
                "generated": generated,
                "correct": is_correct
            })
            
            # Print progress
            if (i + 1) % 10 == 0:
                accuracy_so_far = (correct / total) * 100
                print(f"  Processed {i + 1}/{len(dataset)} examples, accuracy: {accuracy_so_far:.2f}%")
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        evaluation_results = {
            "dataset": self.config.dataset,
            "test_split": self.config.test_split,
            "total_examples": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }
        
        # Save results
        ensure_output_directory(self.config.output_dir)
        results_path = os.path.join(self.config.output_dir, "accuracy_results.json")
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n✅ Evaluation Complete")
        print(f"   Total examples: {total}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Results saved to: {results_path}")
        
        return evaluation_results

