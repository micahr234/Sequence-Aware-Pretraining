"""
Custom data collator that converts character-level attention masks to token-level masks.
"""

from typing import Dict, List, Any, Optional
import torch
from transformers import DataCollatorForLanguageModeling


class AttentionMaskDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that uses character-level attention masks to create token-level labels.
    Only tokens corresponding to characters with True in the attention mask will have
    labels (not -100), allowing gradients to flow only for those tokens.
    """
    
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tokenizer = tokenizer
    
    def _char_to_token_mask(self, text: str, char_mask: List[bool]) -> List[bool]:
        """
        Convert character-level attention mask to token-level mask.
        
        Args:
            text: The input text
            char_mask: Character-level attention mask (list of bools, one per char)
            
        Returns:
            token_mask: Token-level mask (list of bools, one per token)
        """
        # Tokenize with return_offsets_mapping to get char-to-token mapping
        max_length = getattr(self.tokenizer, 'model_max_length', None)
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        
        offsets = encoded["offset_mapping"]  # List of (start, end) tuples
        token_mask = []
        
        for start, end in offsets:
            # Special tokens (BOS, EOS, etc.) have (0, 0) offset
            if start == 0 and end == 0:
                token_mask.append(False)  # Don't compute loss on special tokens
                continue
            
            # Count how many characters in this token's span are True in char_mask
            # If any character is True, mark the token as True
            if end <= len(char_mask):
                token_char_mask = char_mask[start:end]
                token_mask.append(any(token_char_mask) if token_char_mask else False)
            else:
                # Token extends beyond text (shouldn't happen, but handle gracefully)
                token_mask.append(False)
        
        return token_mask
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples, converting char-level masks to token-level labels.
        
        Args:
            features: List of examples from SFTTrainer. 
                    SFTTrainer applies formatting_func, so features may be:
                    - dict with 'text' and 'attention_mask' (char-level) if formatting_func returns dict
                    - or already processed strings
                    
        Returns:
            Batch dictionary with tokenized inputs and labels where only masked tokens
            have labels (not -100)
        """
        # Extract texts and char-level masks
        texts = []
        char_masks = []
        
        for f in features:
            if isinstance(f, dict):
                # If formatting_func returned dict with text and attention_mask
                text = f.get("text", "")
                char_mask = f.get("attention_mask", None)
            else:
                # If formatting_func returned just text string
                text = str(f)
                char_mask = None
            
            texts.append(text)
            
            # Handle char_mask
            if char_mask is None:
                char_masks.append([True] * len(text))  # Default: all masked (pretraining)
            elif len(char_mask) != len(text):
                # Pad or truncate to match text length
                if len(char_mask) < len(text):
                    char_mask = char_mask + [False] * (len(text) - len(char_mask))
                else:
                    char_mask = char_mask[:len(text)]
                char_masks.append(char_mask)
            else:
                char_masks.append(char_mask)
        
        # Convert char-level masks to token-level masks for each example
        token_masks = []
        for text, char_mask in zip(texts, char_masks):
            token_mask = self._char_to_token_mask(text, char_mask)
            token_masks.append(token_mask)
        
        # Tokenize texts
        max_length = getattr(self.tokenizer, 'model_max_length', None)
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Create labels (copy of input_ids)
        labels = batch["input_ids"].clone()
        
        # Set labels to -100 where token_mask is False (to ignore in loss)
        for i, token_mask in enumerate(token_masks):
            # Ensure token_mask length matches sequence length
            seq_len = labels.shape[1]
            if len(token_mask) < seq_len:
                # Pad with False (don't compute loss)
                token_mask = token_mask + [False] * (seq_len - len(token_mask))
            elif len(token_mask) > seq_len:
                # Truncate to sequence length
                token_mask = token_mask[:seq_len]
            
            # Set labels to -100 where mask is False
            for j, mask_val in enumerate(token_mask):
                if not mask_val:
                    labels[i, j] = -100
        
        batch["labels"] = labels
        return batch

