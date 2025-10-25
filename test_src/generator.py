"""
Text generation with probability analysis for sequence-aware models.
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class ProbabilityGenerator:
    """
    Generator for text with detailed probability analysis at each timestep.
    """
    
    def __init__(self, model_name: str, device: str = "auto", seed: int = 42):
        """
        Initialize the probability generator.
        
        Args:
            model_name: Name or path to the model
            device: Device to run on
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Load model and tokenizer
        print(f"📥 Loading Model: {model_name} on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_with_probabilities(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_samples: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> Dict:
        """
        Generate text with detailed probability analysis.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            num_samples: Number of generation samples
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            
        Returns:
            Dictionary containing generation results and probability analysis
        """
        print(f"🎯 Generating {num_samples} samples with prompt: '{prompt[:50]}...'")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]
        
        # Storage for results
        all_generated_tokens = []
        all_logits = []
        all_probabilities = []
        all_top_k_tokens = []
        all_top_k_probs = []
        
        # Generate multiple samples
        for sample_idx in tqdm(range(num_samples), desc="Generating samples"):
            # Reset random seed for each sample to ensure diversity
            torch.manual_seed(self.seed + sample_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed + sample_idx)
            
            # Generate with detailed logging
            sample_results = self._generate_single_sample(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=pad_token_id
            )
            
            all_generated_tokens.append(sample_results["generated_tokens"])
            all_logits.append(sample_results["logits"])
            all_probabilities.append(sample_results["probabilities"])
            all_top_k_tokens.append(sample_results["top_k_tokens"])
            all_top_k_probs.append(sample_results["top_k_probs"])
        
        # Calculate probability mass statistics
        prob_stats = self._calculate_probability_statistics(
            all_generated_tokens, all_probabilities, all_top_k_tokens, all_top_k_probs
        )
        
        return {
            "prompt": prompt,
            "input_length": input_length,
            "max_new_tokens": max_new_tokens,
            "num_samples": num_samples,
            "generation_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample
            },
            "all_generated_tokens": all_generated_tokens,
            "all_logits": all_logits,
            "all_probabilities": all_probabilities,
            "all_top_k_tokens": all_top_k_tokens,
            "all_top_k_probs": all_top_k_probs,
            "probability_statistics": prob_stats
        }
    
    def _generate_single_sample(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        pad_token_id: Optional[int]
    ) -> Dict:
        """Generate a single sample with detailed probability tracking."""
        
        generated_tokens = []
        logits_history = []
        probabilities_history = []
        top_k_tokens_history = []
        top_k_probs_history = []
        
        current_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Calculate probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k tokens and probabilities
            top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
            top_k_tokens = top_k_indices.cpu().numpy()
            top_k_probs_values = top_k_probs.cpu().numpy()
            
            # Store results
            logits_history.append(logits.cpu().numpy())
            probabilities_history.append(probs.cpu().numpy())
            top_k_tokens_history.append(top_k_tokens)
            top_k_probs_history.append(top_k_probs_values)
            
            # Sample next token
            if do_sample:
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    probs = probs.masked_fill(indices_to_remove, 0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return {
            "generated_tokens": generated_tokens,
            "logits": logits_history,
            "probabilities": probabilities_history,
            "top_k_tokens": top_k_tokens_history,
            "top_k_probs": top_k_probs_history
        }
    
    def _calculate_probability_statistics(
        self,
        all_generated_tokens: List[List[int]],
        all_probabilities: List[List[np.ndarray]],
        all_top_k_tokens: List[List[np.ndarray]],
        all_top_k_probs: List[List[np.ndarray]]
    ) -> Dict:
        """Calculate probability statistics across all samples."""
        
        # Find maximum sequence length
        max_length = max(len(tokens) for tokens in all_generated_tokens)
        num_samples = len(all_generated_tokens)
        vocab_size = all_probabilities[0][0].shape[-1]
        
        # Initialize statistics arrays
        token_frequencies = np.zeros((max_length, vocab_size))
        probability_means = np.zeros((max_length, vocab_size))
        probability_stds = np.zeros((max_length, vocab_size))
        entropy_means = np.zeros(max_length)
        entropy_stds = np.zeros(max_length)
        
        # Calculate statistics for each timestep
        for timestep in range(max_length):
            timestep_probs = []
            timestep_entropies = []
            
            for sample_idx in range(num_samples):
                if timestep < len(all_generated_tokens[sample_idx]):
                    # Get the actual token generated
                    token_id = all_generated_tokens[sample_idx][timestep]
                    token_frequencies[timestep, token_id] += 1
                    
                    # Get probability distribution
                    if timestep < len(all_probabilities[sample_idx]):
                        prob_dist = all_probabilities[sample_idx][timestep]
                        timestep_probs.append(prob_dist)
                        
                        # Calculate entropy
                        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
                        timestep_entropies.append(entropy)
            
            if timestep_probs:
                # Calculate mean and std of probability distributions
                prob_array = np.array(timestep_probs)
                probability_means[timestep] = np.mean(prob_array, axis=0)
                probability_stds[timestep] = np.std(prob_array, axis=0)
                
                # Calculate entropy statistics
                entropy_means[timestep] = np.mean(timestep_entropies)
                entropy_stds[timestep] = np.std(timestep_entropies)
        
        # Normalize token frequencies to get empirical probabilities
        token_probabilities = token_frequencies / num_samples
        
        return {
            "max_length": max_length,
            "num_samples": num_samples,
            "vocab_size": vocab_size,
            "token_frequencies": token_frequencies,
            "token_probabilities": token_probabilities,
            "probability_means": probability_means,
            "probability_stds": probability_stds,
            "entropy_means": entropy_means,
            "entropy_stds": entropy_stds
        }
    
    def save_results(self, results: Dict, output_dir: str, save_format: str = "json"):
        """Save generation results to disk."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if save_format in ["json", "both"]:
            # Save as JSON (convert numpy arrays to lists)
            json_results = self._convert_to_json_serializable(results)
            with open(os.path.join(output_dir, "generation_results.json"), "w") as f:
                json.dump(json_results, f, indent=2)
        
        if save_format in ["pickle", "both"]:
            # Save as pickle (preserves numpy arrays)
            with open(os.path.join(output_dir, "generation_results.pkl"), "wb") as f:
                pickle.dump(results, f)
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
