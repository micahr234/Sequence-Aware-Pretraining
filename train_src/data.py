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
    Load dataset split with unified format for broad pretraining.
    
    Args:
        name: Dataset name (see supported datasets below)
        split: Split name ("train", "validation", "test")
        max_examples: Maximum number of examples to load (None for all)
        
    Returns:
        Dataset with unified fields: {"text": str} for pretraining
        
    Supported datasets:
    - "c4": Clean Common Crawl dataset (recommended for broad pretraining)
    - "wikipedia": Wikipedia articles
    - "bookcorpus": BookCorpus dataset
    - "openwebtext": OpenWebText dataset
    - "gsm8k": Math word problems (for reasoning tasks)
    - "squad": Reading comprehension
    - "openbookqa": Science questions
    """
    name = name.lower()
    try:
        if name in ["c4", "c4-en"]:
            # Clean Common Crawl - excellent for broad pretraining
            ds = load_dataset("allenai/c4", "en", split=split, streaming=False)
            # Use the text field directly
            ds = ds.map(lambda ex: {"text": ex["text"]})
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
            
        elif name in ["wikipedia", "wiki"]:
            # Wikipedia articles - good for factual knowledge
            ds = load_dataset("wikipedia", "20220301.en", split=split)
            # Combine title and text
            ds = ds.map(lambda ex: {"text": f"Title: {ex['title']}\n\n{ex['text']}"})
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
            
        elif name in ["bookcorpus", "books"]:
            # BookCorpus - good for narrative text
            ds = load_dataset("bookcorpus", split=split)
            ds = ds.map(lambda ex: {"text": ex["text"]})
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
            
        elif name in ["openwebtext", "owt"]:
            # OpenWebText - diverse web text
            ds = load_dataset("openwebtext", split=split)
            ds = ds.map(lambda ex: {"text": ex["text"]})
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
            
        elif name in ["gsm8k", "gsm", "gsm-8k"]:
            # Math reasoning - specialized task
            ds = load_dataset("gsm8k", "main", split=split)
            ds = ds.map(lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {extract_gsm8k_final(ex['answer'])}"})
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
            
        elif name in ["squad"]:
            # Reading comprehension
            ds = load_dataset("squad", split=split)
            def to_fields(ex):
                ans = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
                return {"text": f"Question: {ex['question']}\nAnswer: {ans}"}
            ds = ds.map(to_fields, remove_columns=[c for c in ds.column_names if c not in ["text"]])
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
            
        elif name in ["openbookqa"]:
            # Science questions
            ds = load_dataset("openbookqa", "main", split=split)
            def to_fields(ex):
                idx = ord(ex["answerKey"].strip()) - ord("A")
                choice = ex["choices"]["text"][idx]
                return {"text": f"Question: {ex['question_stem']}\nAnswer: {choice}"}
            ds = ds.map(to_fields, remove_columns=[c for c in ds.column_names if c not in ["text"]])
            if max_examples and max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))
            return ds
            
        else:
            raise ValueError(f"Unknown dataset: {name}. Supported: c4, wikipedia, bookcorpus, openwebtext, gsm8k, squad, openbookqa")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{name}' split '{split}': {e}")
