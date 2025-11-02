import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing import Optional, List, Dict, Any
import wandb


class DiscountedSFTTrainer(Seq2SeqTrainer):
    """
    A specialized SFTTrainer that implements discounted log-suffix weighting.
    
    This trainer applies position-dependent weights to the standard cross-entropy loss,
    where later tokens in a sequence receive higher weights according to a discount factor gamma.
    The weighting scheme follows: w_k = (1 - gamma^k) for position k.
    - When gamma=0, weights are uniform: w_k = 1 (standard cross-entropy, no discounting).
    - When gamma=1, weights become linear: w_k = k (maximum discounting).
    
    Args:
        gamma (float): Discount. Higher values give more weight to later tokens.
                      gamma=0 gives uniform weights (no discounting), gamma=1 gives linear weights.
        *args: Arguments passed to Seq2SeqTrainer
        **kwargs: Keyword arguments passed to Seq2SeqTrainer
    """
    
    def __init__(
        self, 
        *args, 
        gamma: float = 0.98,
        eval_gold_answers: Optional[List[str]] = None,
        eval_prompts_list: Optional[List[str]] = None,
        eval_answer_regex: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DiscountedSFTTrainer.
        
        Args:
            gamma: Discount factor for position weighting
            eval_gold_answers: List of gold answer strings for evaluation (optional)
            eval_prompts_list: List of prompt strings for evaluation (optional)
            eval_answer_regex: Regex pattern to extract answers from model output (optional)
            *args: Arguments passed to Seq2SeqTrainer
            **kwargs: Keyword arguments passed to Seq2SeqTrainer
        """
        self.gamma = float(gamma)
        
        # Store evaluation info for compute_metrics
        self.eval_gold_answers = eval_gold_answers
        self.eval_prompts_list = eval_prompts_list
        self.eval_answer_regex = eval_answer_regex
        
        # Always use our internal compute_metrics, overriding any provided one
        # Remove compute_metrics from kwargs if present (we'll override it as a method)
        kwargs.pop('compute_metrics', None)
        
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _position_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Build per-token weights w_k(gamma) aligned with label positions.
        
        Args:
            labels: [B, T] tensor of labels. All positions are counted for weighting.
                   Position index k starts from 0 and increments for each position.
        
        Returns:
            weights: [B, T] tensor with position weights for all positions
        """
        device = labels.device
        B, T = labels.shape
        
        # Build cumulative count per row for ALL positions (including invalid ones)
        # k starts at 0 for first position, 1 for second, etc.
        k = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)  # [B, T]

        if abs(self.gamma) < 1e-8:
            # gamma = 0: uniform weights (all positions get weight 1)
            w = torch.ones_like(k)
        elif abs(1.0 - self.gamma) < 1e-8:
            # gamma = 1: linear weights
            w = k  # when gamma=1, w_k = k
        else:
            # 0 < gamma < 1: discounted weights
            # w_k = (1 - gamma^k) for k >= 0
            # Use a more numerically stable computation
            w = (1.0 - torch.pow(self.gamma, k))
        
        return w  # [B, T], weights for all positions

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Replace standard CE with discounted log-suffix weighted CE.
        
        Args:
            model: The model being trained
            inputs: Dictionary containing input tensors including 'labels'
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in the batch (for compatibility)
            
        Returns:
            loss: Weighted cross-entropy loss (scalar tensor for optimization)
            outputs: Model outputs (if return_outputs=True)
        """
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]
        
        # Reshape logits and labels for cross_entropy: [B*T, V] and [B*T]
        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)  # [B*T, V]
        labels_flat = labels.view(-1)  # [B*T]
        
        # Use standard PyTorch cross_entropy with reduction='none' to get per-token losses
        # ignore_index=-100 handles invalid positions automatically
        per_token_loss = F.cross_entropy(
            logits_flat, 
            labels_flat, 
            reduction='none',
            ignore_index=-100
        )  # [B*T]
        
        # Reshape back to [B, T]
        per_token_loss = per_token_loss.view(B, T)  # [B, T]
        
        # Count valid tokens for normalization (ignore_index=-100 already sets invalid positions to loss=0)
        num_valid_tokens = (labels != -100).sum()
        if num_valid_tokens == 0:
            raise ValueError(
                "No valid tokens found in batch (all labels are -100). "
                "This indicates all tokens are marked as ignored, which should not happen. "
                "Check your data collator and preprocessing."
            )
        
        # Build weights for all positions (including invalid ones, but they won't contribute to loss)
        with torch.no_grad():
            weights = self._position_weights(labels)  # [B, T], includes all positions
        
        # Calculate unweighted loss (standard cross-entropy) for logging
        unweighted_loss = per_token_loss.sum() / num_valid_tokens
        
        # Calculate weighted loss: apply position weights to per-token losses
        weighted_loss = (weights * per_token_loss).sum() / num_valid_tokens

        # Store both losses for logging
        if not hasattr(self, '_unweighted_loss'):
            self._unweighted_loss = []
        if not hasattr(self, '_weighted_loss'):
            self._weighted_loss = []
        self._unweighted_loss.append(unweighted_loss.detach().cpu().item())
        self._weighted_loss.append(weighted_loss.detach().cpu().item())

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def get_unweighted_loss(self):
        """
        Get the most recent unweighted loss value.
        
        Returns:
            float: The most recent unweighted loss value, or None if not available
        """
        if hasattr(self, '_unweighted_loss') and self._unweighted_loss:
            return self._unweighted_loss[-1]
        return None
    
    def get_weighted_loss(self):
        """
        Get the most recent weighted loss value.
        
        Returns:
            float: The most recent weighted loss value, or None if not available
        """
        if hasattr(self, '_weighted_loss') and self._weighted_loss:
            return self._weighted_loss[-1]
        return None

    def log(self, logs, start_time=None):
        """
        Override logging to include both weighted and unweighted loss.
        Also ensures eval metrics from compute_metrics are properly included.
        
        Args:
            logs: Dictionary of metrics to log (may include eval metrics from compute_metrics)
            start_time: Optional start time for logging (passed to parent)
        """
        # Add unweighted loss to logs if available
        unweighted_loss = self.get_unweighted_loss()
        if unweighted_loss is not None:
            logs['unweighted_loss'] = unweighted_loss
        
        # Add weighted loss to logs if available (note: 'loss' is already the weighted loss)
        weighted_loss = self.get_weighted_loss()
        if weighted_loss is not None:
            logs['weighted_loss'] = weighted_loss
        
        # Ensure eval metrics are included - Trainer should have added them already
        # but we verify they're present and log them explicitly if needed
        if 'eval' in str(logs) or any(k.startswith('eval') for k in logs.keys()):
            print(f"🔍 Debug: Log method called with eval metrics: {[k for k in logs.keys() if 'eval' in k.lower()]}")
        
        # Call parent logging method (this will log all metrics including eval ones)
        super().log(logs, start_time)
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute accuracy metrics from generated predictions.
        
        This method extracts answers from model outputs using regex patterns,
        compares them to gold answers, and returns accuracy metrics.
        
        Similar structure to compute_loss - overrides the parent class method.
        
        Args:
            eval_pred: EvalPrediction object with predictions (generated token IDs only, input was stripped)
                    Has attributes: predictions (array of token IDs) and label_ids
        
        Returns:
            Dictionary with accuracy and regex_coverage metrics
        """
        # If no eval data provided, return empty dict (metrics won't be computed)
        if self.eval_gold_answers is None or self.eval_prompts_list is None:
            return {}
        
        # Import here to avoid circular imports
        from evaluate import extract_answer_from_output, compare_answers
        
        preds = eval_pred.predictions
        print(f"\n🔍 Debug: compute_metrics called")
        print(f"   eval_pred type: {type(eval_pred)}")
        print(f"   eval_pred attributes: {dir(eval_pred)}")
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
        
        # Get tokenizer from processing_class
        tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else self.processing_class
        
        # preds now contains only the generated tokens (input was stripped in prediction_step)
        correct = 0
        y_pred = []
        
        for i, generated_ids in enumerate(preds):
            # Decode the generated tokens (already extracted from full sequence in prediction_step)
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Reconstruct full output: prompt + generated text
            prompt = self.eval_prompts_list[i] if i < len(self.eval_prompts_list) else ""
            full_output = prompt + gen_text
            
            # Extract answer from full output using regex
            answer_regex = self.eval_answer_regex
            if answer_regex is None:
                # Default regex pattern (can be customized)
                answer_regex = r'The answer is:\s*(.+?)(?:\n|$)'
            
            predicted_answer = extract_answer_from_output(answer_regex, full_output)
            gold = self.eval_gold_answers[i] if i < len(self.eval_gold_answers) else ""
            
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
