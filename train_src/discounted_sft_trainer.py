import torch
import torch.nn.functional as F
from trl import SFTTrainer


class DiscountedLogSuffixSFTTrainer(SFTTrainer):
    """
    A specialized SFTTrainer that implements discounted log-suffix weighting.
    
    This trainer applies position-dependent weights to the standard cross-entropy loss,
    where later tokens in a sequence receive higher weights according to a discount factor gamma.
    The weighting scheme follows: w_k = (1 - gamma^k) / (1 - gamma) for position k.
    
    Args:
        gamma (float): Discount factor in (0, 1]. Higher values give more weight to later tokens.
                      When gamma=1, weights become linear: w_k = k.
        *args: Arguments passed to SFTTrainer
        **kwargs: Keyword arguments passed to SFTTrainer
    """
    
    def __init__(self, *args, gamma: float = 0.98, **kwargs):
        assert 0.0 < gamma <= 1.0, "gamma must be in (0, 1]"
        self.gamma = float(gamma)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _position_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Build per-token weights w_k(gamma) aligned with label positions.
        
        Args:
            labels: [B, T] with -100 for ignore positions (HF convention).
                   For each row, let valid positions be where labels != -100.
                   Index those valid positions as k=1..L_i and assign weight w_k.
        
        Returns:
            weights: [B, T] tensor with position weights, zeros where labels == -100
        """
        device = labels.device
        B, T = labels.shape
        valid = (labels != -100).to(torch.float32)              # [B, T]
        
        # Build cumulative count per row for valid positions only
        k = valid.cumsum(dim=1) * valid                         # k in {0..L_i} at valid positions
        
        # Ensure k is non-negative and handle edge cases
        k = k.clamp_min(0)

        if abs(1.0 - self.gamma) < 1e-8:
            w = k.to(torch.float32)  # when gamma=1, w_k = k
        else:
            # w_k = (1 - gamma^k) for k > 0, 0 for k = 0
            # Use a more numerically stable computation
            gamma_pow_k = torch.pow(self.gamma, k.clamp_min(0))
            w = (1.0 - gamma_pow_k).to(torch.float32)

        # Zero out invalid spots explicitly
        w = w * valid
        
        return w  # [B, T], zeros where labels == -100

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
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        # log p(w_k) at each labeled position
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Ensure labels are within vocabulary bounds
        vocab_size = logits.size(-1)
        labels_clamped = labels.clamp(0, vocab_size - 1)
        
        # Create mask for valid positions (not -100)
        valid_mask = (labels != -100)
        
        # Gather log probabilities, but only for valid positions
        token_logp = log_probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)  # [B, T]
        
        # Set invalid positions to 0 (they'll be masked out anyway)
        token_logp = token_logp * valid_mask.float()

        # Build weights for labeled positions
        with torch.no_grad():
            weights = self._position_weights(labels)  # [B, T]
            denom = (labels != -100).sum().clamp_min(1)

        # Calculate unweighted loss (standard cross-entropy) for logging
        unweighted_loss = -token_logp.sum() / denom
        
        # Calculate weighted (negative) log-likelihood, averaged over valid tokens
        weighted_loss = -(weights * token_logp).sum() / denom

        # Store unweighted loss for logging (accessible via trainer.log_metrics)
        if not hasattr(self, '_unweighted_loss'):
            self._unweighted_loss = []
        self._unweighted_loss.append(unweighted_loss.detach().cpu().item())

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

    def log(self, logs, start_time=None):
        """
        Override logging to include unweighted loss.
        
        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for logging (passed to parent)
        """
        # Add unweighted loss to logs if available
        unweighted_loss = self.get_unweighted_loss()
        if unweighted_loss is not None:
            logs['unweighted_loss'] = unweighted_loss
        
        # Call parent logging method
        super().log(logs, start_time)
