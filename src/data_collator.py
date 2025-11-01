"""
Custom data collator that converts character-level attention masks to token-level masks.
"""

import warnings
from typing import Dict, List, Any
import torch
from transformers import DataCollatorForLanguageModeling


class AttentionMaskDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that uses character-level attention masks to create token-level labels.
    Only tokens corresponding to characters with True in the attention mask will have
    labels (not -100), allowing gradients to flow only for those tokens.
    """
    
    def __init__(
        self, 
        tokenizer, 
        train_on_answers_only, 
        question_reasoning_join_string: str,
        reasoning_answer_join_string: str,
        *args, 
        **kwargs
    ):
        """
        Args:
            tokenizer: Tokenizer to use
            train_on_answers_only: If True, only compute loss on answer portions.
                                  When an answer exists, text portion loss is zeroed out.
                                  When no answer exists, no loss is computed (all masked).
            question_reasoning_join_string: Join string between question and reasoning (e.g., "\n\n")
            reasoning_answer_join_string: Join string between reasoning and answer (e.g., "\n\nThe answer is: ")
        """
        super().__init__(tokenizer, *args, **kwargs)
        self.tokenizer = tokenizer
        self.train_on_answers_only = train_on_answers_only
        self.question_reasoning_join_string = question_reasoning_join_string
        self.reasoning_answer_join_string = reasoning_answer_join_string
    
    def _char_to_token_mask(self, text: str, char_mask: List[bool]) -> List[bool]:
        """
        Convert character-level attention mask to token-level mask.
        
        Args:
            text: The input text
            char_mask: Character-level attention mask (list of bools, one per char)
            
        Returns:
            token_mask: Token-level mask (list of bools, one per token)
        """
        max_length = getattr(self.tokenizer, 'model_max_length', None)
        
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        
        offsets = encoded["offset_mapping"]
        token_mask = []
        
        for start, end in offsets:
            if start == 0 and end == 0:
                token_mask.append(False)
                continue
            
            if end <= len(char_mask):
                token_char_mask = char_mask[start:end]
                token_mask.append(any(token_char_mask) if token_char_mask else False)
            else:
                token_mask.append(False)
        
        return token_mask
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples, generating masks from text and answer fields.
        
        Args:
            features: List of dicts with 'text' and 'answer' fields from dataset.
                    - If answer exists: compute loss only on answer portion
                    - If answer is None/empty: compute loss on entire text (pretraining mode)
                    
        Returns:
            Batch dictionary with tokenized inputs and labels where only masked tokens
            have labels (not -100)
        """
        texts = []
        char_masks = []
        
        for f in features:
            text = f["text"]
            answer = f.get("answer")
            
            # Validate text - raise error if invalid rather than silently skipping
            if not isinstance(text, str):
                raise ValueError(
                    f"Expected 'text' to be a string, got {type(text).__name__}. "
                    f"Feature keys: {list(f.keys())}"
                )
            if not text or len(text.strip()) == 0:
                raise ValueError(
                    f"Found empty text in feature. This should not happen after preprocessing. "
                    f"Feature keys: {list(f.keys())}"
                )
            
            # If answer exists or train_on_answers_only is enabled
            if answer:
                # Validate answer is a string
                if not isinstance(answer, str):
                    raise ValueError(
                        f"Expected 'answer' to be a string or None, got {type(answer).__name__}. "
                        f"Feature keys: {list(f.keys())}"
                    )
                if not answer.strip():
                    raise ValueError(
                        f"Found empty answer string. Empty answers should be None, not empty strings. "
                        f"Feature keys: {list(f.keys())}"
                    )
                
                full_text = text + self.reasoning_answer_join_string + answer
                texts.append(full_text)
                char_mask = [False] * len(text) + [True] * (len(full_text) - len(text))
                char_masks.append(char_mask)
            elif self.train_on_answers_only:
                texts.append(text)
                char_masks.append([False] * len(text))
            else:
                texts.append(text)
                char_masks.append([True] * len(text))
        
        if not texts:
            raise ValueError(
                "No valid texts found in batch. This should not happen if preprocessing is correct."
            )
        
        token_masks = []
        for text, char_mask in zip(texts, char_masks):
            token_mask = self._char_to_token_mask(text, char_mask)
            token_masks.append(token_mask)
        
        max_length = getattr(self.tokenizer, 'model_max_length', None)
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        
        if max_length is not None and "attention_mask" in batch:
            at_max_length = 0
            attention_mask = batch["attention_mask"]
            for attn_mask in attention_mask:
                actual_length = attn_mask.sum().item()
                if actual_length == max_length:
                    at_max_length += 1
            
            if at_max_length > 0:
                warnings.warn(
                    f"Data truncation detected: {at_max_length} out of {len(texts)} examples "
                    f"are at max_length={max_length} and may have been truncated.",
                    UserWarning,
                    stacklevel=2
                )
        
        labels = batch["input_ids"].clone()
        
        for i, token_mask in enumerate(token_masks):
            seq_len = labels.shape[1]
            
            if len(token_mask) < seq_len:
                token_mask = token_mask + [False] * (seq_len - len(token_mask))
            elif len(token_mask) > seq_len:
                token_mask = token_mask[:seq_len]
            
            for j, mask_val in enumerate(token_mask):
                if not mask_val:
                    labels[i, j] = -100
        
        batch["labels"] = labels
        return batch

