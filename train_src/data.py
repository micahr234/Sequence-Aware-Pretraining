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
    # Dataset configuration mapping
    DATASET_CONFIGS = {
        "c4": {
            "dataset_name": "allenai/c4",
            "config": "en",
            "text_field": "text",
            "transform": lambda ex: {"text": ex["text"]},
            "streaming": False
        },
        "c4-en": {
            "dataset_name": "allenai/c4", 
            "config": "en",
            "text_field": "text",
            "transform": lambda ex: {"text": ex["text"]},
            "streaming": False
        },
        "wikipedia": {
            "dataset_name": "wikipedia",
            "config": "20220301.en", 
            "text_field": "text",
            "transform": lambda ex: {"text": f"Title: {ex['title']}\n\n{ex['text']}"},
            "streaming": False
        },
        "wiki": {
            "dataset_name": "wikipedia",
            "config": "20220301.en",
            "text_field": "text", 
            "transform": lambda ex: {"text": f"Title: {ex['title']}\n\n{ex['text']}"},
            "streaming": False
        },
        "bookcorpus": {
            "dataset_name": "bookcorpus",
            "config": None,
            "text_field": "text",
            "transform": lambda ex: {"text": ex["text"]},
            "streaming": False
        },
        "books": {
            "dataset_name": "bookcorpus",
            "config": None,
            "text_field": "text",
            "transform": lambda ex: {"text": ex["text"]},
            "streaming": False
        },
        "openwebtext": {
            "dataset_name": "openwebtext",
            "config": None,
            "text_field": "text",
            "transform": lambda ex: {"text": ex["text"]},
            "streaming": False
        },
        "owt": {
            "dataset_name": "openwebtext",
            "config": None,
            "text_field": "text",
            "transform": lambda ex: {"text": ex["text"]},
            "streaming": False
        },
        "gsm8k": {
            "dataset_name": "gsm8k",
            "config": "main",
            "text_field": "text",
            "transform": lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {extract_gsm8k_final(ex['answer'])}"},
            "streaming": False
        },
        "gsm": {
            "dataset_name": "gsm8k",
            "config": "main",
            "text_field": "text",
            "transform": lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {extract_gsm8k_final(ex['answer'])}"},
            "streaming": False
        },
        "gsm-8k": {
            "dataset_name": "gsm8k",
            "config": "main",
            "text_field": "text",
            "transform": lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {extract_gsm8k_final(ex['answer'])}"},
            "streaming": False
        },
        "squad": {
            "dataset_name": "squad",
            "config": None,
            "text_field": "text",
            "transform": lambda ex: {
                "text": f"Question: {ex['question']}\nAnswer: {ex['answers']['text'][0] if ex['answers']['text'] else ''}"
            },
            "streaming": False
        },
        "openbookqa": {
            "dataset_name": "openbookqa",
            "config": "main",
            "text_field": "text",
            "transform": lambda ex: {
                "text": f"Question: {ex['question_stem']}\nAnswer: {ex['choices']['text'][ord(ex['answerKey'].strip()) - ord('A')]}"
            },
            "streaming": False
        }
    }
    
    name = name.lower()
    
    if name not in DATASET_CONFIGS:
        supported = ", ".join(set(config["dataset_name"] for config in DATASET_CONFIGS.values()))
        raise ValueError(f"Unknown dataset: {name}. Supported: {supported}")
    
    try:
        config = DATASET_CONFIGS[name]
        
        # Build split string with max_examples if specified
        split_str = f"{split}[:{max_examples}]" if max_examples and max_examples > 0 else split
        
        # Load dataset
        if config["config"]:
            ds = load_dataset(config["dataset_name"], config["config"], split=split_str, streaming=config["streaming"])
        else:
            ds = load_dataset(config["dataset_name"], split=split_str, streaming=config["streaming"])
        
        # Apply transformation
        if name in ["squad", "openbookqa"]:
            # These need special handling for column removal
            ds = ds.map(config["transform"], remove_columns=[c for c in ds.column_names if c not in ["text"]])
        else:
            ds = ds.map(config["transform"])
        
        return ds
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{name}' split '{split}': {e}")
