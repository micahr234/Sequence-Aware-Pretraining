"""
Evaluation utility functions for training.
"""

import re
from typing import Optional


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

