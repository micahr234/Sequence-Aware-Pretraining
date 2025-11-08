"""
Custom data collator that converts character-level attention masks to token-level masks.
"""

import warnings
from typing import Dict, List, Any
import torch
from transformers import DataCollatorForLanguageModeling


class TrainingCollator(DataCollatorForLanguageModeling):
    """
    Data collator that uses character-level attention masks to create token-level labels.
    Only tokens corresponding to characters with True in the attention mask will have
    labels (not -100), allowing gradients to flow only for those tokens.
    """
    
    def __init__(
        self, 
        tokenizer, 
        train_on_answers_only, 
        question_template: str,
        answer_template: str,
        *args, 
        **kwargs
    ):
        """
        Args:
            tokenizer: Tokenizer to use
            train_on_answers_only: If True, only compute loss on answer portions.
                                  When an answer exists, text portion loss is zeroed out.
                                  When no answer exists, no loss is computed (all masked).
            question_template: Template for formatting question (e.g., "{question}")
            answer_template: Template for formatting answer (e.g., "{answer}" or "{reasoning}\n\nThe answer is: {answer}")
        """
        super().__init__(tokenizer, *args, **kwargs)
        self.tokenizer = tokenizer
        self.train_on_answers_only = train_on_answers_only
        self.question_template = question_template
        self.answer_template = answer_template
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples, generating masks from question, reasoning, and answer fields.
        
        Tokenizes question, reasoning, and answer separately for easier mask creation.
        
        Args:
            features: List of dicts with 'question', 'reasoning', and 'answer' fields from dataset.
                    - Format: question + question_reasoning_join_string + reasoning + reasoning_answer_join_string + answer
                    - If answer exists: compute loss only on answer portion (and optionally reasoning)
                    - If answer is None/empty: compute loss on entire text (pretraining mode)
                    
        Returns:
            Batch dictionary with tokenized inputs and labels where only masked tokens
            have labels (not -100)
        """
        # Tokenize each part separately
        all_input_ids = []
        all_attention_masks = []
        all_label_ids = []
        max_seq_len = 0

        for f in features:
            question = f["question"]
            reasoning = f["reasoning"]
            answer = f["answer"]
            
            # Tokenize question separately
            formatted_question = self.question_template.format(question=question)
            question_enc = self.tokenizer(
                formatted_question,
                add_special_tokens=False,
                return_tensors=None,
            )
            question_tokens = question_enc["input_ids"]
            
            # Tokenize reasoning separately
            reasoning_enc = self.tokenizer(
                reasoning,
                add_special_tokens=False,
                return_tensors=None,
            )
            reasoning_tokens = reasoning_enc["input_ids"]
            
            # Tokenize answer separately
            formatted_answer = self.answer_template.format(answer=answer)
            answer_enc = self.tokenizer(
                formatted_answer,
                add_special_tokens=False,
                return_tensors=None,
            )
            answer_tokens = answer_enc["input_ids"]

            input_ids = question_tokens + reasoning_tokens + answer_tokens
            attention_mask = [1] * len(question_tokens) + [1] * len(reasoning_tokens) + [1] * len(answer_tokens)
            
            if self.train_on_answers_only:
                label_ids = [-100] * len(question_tokens) + [-100] * len(reasoning_tokens) + answer_tokens
            else:
                label_ids = question_tokens + reasoning_tokens + answer_tokens
            
            # error if input_ids is longer than the model's max length
            if len(input_ids) > self.tokenizer.model_max_length:
                raise ValueError(f"Input ids are longer than the model's max length: {len(input_ids)} > {self.tokenizer.model_max_length}")

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_label_ids.append(label_ids)
            max_seq_len = max(max_seq_len, len(input_ids))
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        padded_label_ids = []
        for input_ids, attention_mask, label_ids in zip(all_input_ids, all_attention_masks, all_label_ids):
            padding_length = max_seq_len - len(input_ids)
            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + [1] * len(input_ids)
            label_ids = [-100] * padding_length + label_ids
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
            padded_label_ids.append(label_ids)
        
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_label_ids, dtype=torch.long),
        }
        
        return batch


class EvalDataCollator:
    """
    Data collator for evaluation that formats prompts and tokenizes on-the-fly.
    
    Follows the same format as TrainingCollator for consistency.
    Returns answers in the batch for reference during evaluation.
    """
    def __init__(self, tokenizer, question_template: str):
        self.tokenizer = tokenizer
        self.question_template = question_template
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format prompts, tokenize, and include answers in batch.
        
        Args:
            features: List of dicts with 'question' and 'answer' fields
            
        Returns:
            Batch dictionary with tokenized inputs (input_ids, attention_mask) and answers
        """
        # Tokenize each part separately (same format as TrainingCollator)
        all_input_ids = []
        all_attention_masks = []
        all_answers = []
        max_seq_len = 0
        for f in features:
            question = f["question"]
            ground_truth = f["answer"]
            
            # Format prompt using template
            formatted_question = self.question_template.format(question=question)
            question_enc = self.tokenizer(
                formatted_question,
                add_special_tokens=False,
                return_tensors=None,
            )
            question_tokens = question_enc["input_ids"]
            
            input_ids = question_tokens
            attention_mask = [1] * len(question_tokens)
            
            # Error if input_ids is longer than the model's max length
            if len(input_ids) > self.tokenizer.model_max_length:
                raise ValueError(
                    f"Input ids are longer than the model's max length: "
                    f"{len(input_ids)} > {self.tokenizer.model_max_length}"
                )
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            max_seq_len = max(max_seq_len, len(input_ids))
            all_answers.append(ground_truth)
        
        # Pad sequences (left padding for generation)
        padded_input_ids = []
        padded_attention_masks = []
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            padding_length = max_seq_len - len(input_ids)
            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + attention_mask
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
        
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "answer": all_answers,
        }
        
        return batch

