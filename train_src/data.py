import re

from datasets import Dataset, load_dataset

def extract_gsm8k_final(answer_text: str) -> str:
    """
    GSM8K answers look like: "... rationale ... #### 42"
    We extract the part after the last '####' token and strip.
    """
    m = re.findall(r"####\s*(.+)", answer_text)
    return m[-1].strip() if m else answer_text.strip()

def load_split(name: str, split: str, max_examples: int = None) -> Dataset:
    """
    Load dataset split with unified format.
    
    Args:
        name: Dataset name ("gsm8k", "squad", "openbookqa")
        split: Split name ("train", "validation", "test")
        max_examples: Maximum number of examples to load (None for all)
        
    Returns:
        Dataset with unified fields: {"question": str, "answer": str}
    """
    name = name.lower()
    try:
        if name in ["gsm8k", "gsm", "gsm-8k"]:
            ds = load_dataset("gsm8k", "main", split=split)
            # unify fields: {'question': str, 'answer': str (final answer only)}
            ds = ds.map(lambda ex: {"answer": extract_gsm8k_final(ex["answer"])})
            # Limit dataset size if specified
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
        elif name in ["squad"]:
            ds = load_dataset("squad", split=split)
            # take first answer
            def to_fields(ex):
                ans = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
                return {"question": ex["question"], "answer": ans}
            ds = ds.map(to_fields, remove_columns=[c for c in ds.column_names if c not in ["question","answer"]])
            # Limit dataset size if specified
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
        elif name in ["openbookqa"]:
            ds = load_dataset("openbookqa", "main", split=split)
            # correct answer letter
            def to_fields(ex):
                # convert multiple-choice to the string of the correct choice
                idx = ord(ex["answerKey"].strip()) - ord("A")
                choice = ex["choices"]["text"][idx]
                return {"question": ex["question_stem"], "answer": choice}
            ds = ds.map(to_fields, remove_columns=[c for c in ds.column_names if c not in ["question","answer"]])
            # Limit dataset size if specified
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
        else:
            raise ValueError(f"Unknown dataset: {name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{name}' split '{split}': {e}")
