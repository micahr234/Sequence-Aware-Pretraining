import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


class DiscountedLogSuffixSFTTrainer(Seq2SeqTrainer):
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
    
    def __init__(self, *args, gamma: float = 0.98, **kwargs):
        self.gamma = float(gamma)
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
