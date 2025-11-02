import re

from datasets import Dataset, load_dataset


# ============================================================================
# Dataset-specific preprocessing functions
# ============================================================================

def preprocess_c4(example: dict) -> dict:
    """
    Preprocess C4 (Colossal Clean Crawled Corpus) dataset.
    
    Args:
        example: Raw example with "text" field
        
    Returns:
        Dict with "question", "reasoning", and "answer" fields (for pretraining)
    """
    text = example["text"]
    if not text or len(text) < 10:
        return None
    text = text.strip()
    return {"question": "", "reasoning": "", "answer": text}


def preprocess_wikipedia(example: dict) -> dict:
    """
    Preprocess Wikipedia dataset.
    
    Args:
        example: Raw example with "title" and "text" fields
        
    Returns:
        Dict with "question", "reasoning", and "answer" fields (for pretraining)
    """
    title = example["title"] if "title" in example else None
    text = example["text"]
    
    if not text or len(text) < 50:
        return None
    
    if title:
        formatted_text = f"Title: {title}\n\n{text}"
    else:
        formatted_text = text
    
    formatted_text = formatted_text.strip()
    return {"question": "", "reasoning": "", "answer": formatted_text}


def preprocess_bookcorpus(example: dict) -> dict:
    """
    Preprocess BookCorpus dataset.
    
    Args:
        example: Raw example with "text" field
        
    Returns:
        Dict with "question", "reasoning", and "answer" fields (for pretraining)
    """
    text = example["text"]
    
    if not text or len(text) < 20:
        return None
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return {"question": "", "reasoning": "", "answer": text}


def preprocess_openwebtext(example: dict) -> dict:
    """
    Preprocess OpenWebText dataset.
    
    Args:
        example: Raw example with "text" field
        
    Returns:
        Dict with "question", "reasoning", and "answer" fields (for pretraining)
    """
    text = example["text"]
    
    if not text or len(text) < 20:
        return None
    
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return {"question": "", "reasoning": "", "answer": text}


def preprocess_gsm8k(example: dict) -> dict:
    """
    Preprocess GSM8K (Grade School Math 8K) dataset.
    
    Args:
        example: Raw example with "question" and "answer" fields
        
    Returns:
        Dict with "question", "reasoning", and "answer" fields
    """
    question = example["question"].strip()
    full_answer = example["answer"].strip()
    
    marker_match = re.search(r"([\s\S]*?)\s*####\s*(.+)", full_answer)
    if not marker_match:
        raise ValueError(f"GSM8K example missing '####' marker. Question: {question[:100]}...")
    
    reasoning = marker_match.group(1).strip()
    answer = marker_match.group(2).replace(",", "").strip()
    
    if not answer:
        raise ValueError(f"GSM8K example has empty final answer after '####' marker. Question: {question[:100]}...")
    
    return {"question": question, "reasoning": reasoning, "answer": answer}


def preprocess_squad(example: dict) -> dict:
    """
    Preprocess SQuAD (Stanford Question Answering Dataset).
    
    Args:
        example: Raw example with "question", "context", and "answers" fields
        
    Returns:
        Dict with "question", "reasoning", and "answer" fields
    """
    question = example["question"]
    context = example["context"] if "context" in example else None
    answers = example["answers"] if "answers" in example else None
    
    if not question:
        return None
    
    answer_text = ""
    if answers and "text" in answers and answers["text"]:
        answer_text = answers["text"][0]
    
    # Format question with context if available
    if context:
        formatted_question = f"Question: {question}\nContext: {context}"
    else:
        formatted_question = f"Question: {question}"
    
    formatted_question = formatted_question.strip()
    answer_text = answer_text.strip() if answer_text else None
    return {"question": formatted_question, "reasoning": "", "answer": answer_text}


def preprocess_openbookqa(example: dict) -> dict:
    """
    Preprocess OpenBookQA dataset.
    
    Args:
        example: Raw example with "question_stem", "choices", and "answerKey" fields
        
    Returns:
        Dict with "question", "reasoning", and "answer" fields
    """
    question_stem = example["question_stem"]
    choices = example["choices"]
    answer_key = example["answerKey"].upper() if "answerKey" in example else None
    
    formatted_question = f"Question: {question_stem}"
    formatted_question = formatted_question.strip()
    
    if not question_stem or not choices or not answer_key:
        return {"question": formatted_question, "reasoning": "", "answer": None}
    
    choice_texts = choices["text"] if "text" in choices else None
    if not choice_texts:
        return {"question": formatted_question, "reasoning": "", "answer": None}
    
    if answer_key and len(answer_key) == 1 and answer_key.isalpha():
        choice_idx = ord(answer_key) - ord('A')
        if 0 <= choice_idx < len(choice_texts):
            correct_answer = choice_texts[choice_idx]
            correct_answer = correct_answer.strip() if correct_answer else None
            return {"question": formatted_question, "reasoning": "", "answer": correct_answer}
    
    return {"question": formatted_question, "reasoning": "", "answer": None}


# ============================================================================
# Main dataset loading function
# ============================================================================

def load_split(name: str, split: str, max_examples: int = None) -> Dataset:
    """
    Load dataset split with unified format for broad pretraining.
    
    Each dataset has its own preprocessing function that handles the specific
    structure and formatting requirements.
    
    Args:
        name: Dataset name (see supported datasets below)
        split: Split name ("train", "validation", "test")
        max_examples: Maximum number of examples to load (None for all)
        
    Returns:
        Dataset with unified fields: {"question": str, "reasoning": str, "answer": str or None}
        
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
            "preprocess_fn": preprocess_c4
        },
        "wikipedia": {
            "dataset_name": "wikipedia",
            "config": "20220301.en",
            "preprocess_fn": preprocess_wikipedia
        },
        "bookcorpus": {
            "dataset_name": "bookcorpus",
            "config": None,
            "preprocess_fn": preprocess_bookcorpus
        },
        "openwebtext": {
            "dataset_name": "openwebtext",
            "config": None,
            "preprocess_fn": preprocess_openwebtext
        },
        "gsm8k": {
            "dataset_name": "gsm8k",
            "config": "main",
            "preprocess_fn": preprocess_gsm8k
        },
        "squad": {
            "dataset_name": "squad",
            "config": None,
            "preprocess_fn": preprocess_squad
        },
        "openbookqa": {
            "dataset_name": "openbookqa",
            "config": "main",
            "preprocess_fn": preprocess_openbookqa
        }
    }

    try:
        config = DATASET_CONFIGS[name]
        
        # Use non-streaming mode (load entire dataset into memory)
        # Include max_examples in split string if specified
        if max_examples and max_examples > 0:
            split_str = f"{split}[:{max_examples}]"
        else:
            split_str = split
        
        if config["config"]:
            print(f"Loading dataset {config['dataset_name']} with config {config['config']} and split {split_str}")
            ds = load_dataset(config["dataset_name"], config["config"], split=split_str, streaming=False)
        else:
            print(f"Loading dataset {config['dataset_name']} with split {split_str}")
            ds = load_dataset(config["dataset_name"], split=split_str, streaming=False)
        
        print(f"Applying preprocessing for {name} dataset...")
        
        # Get column names for non-streaming datasets
        if hasattr(ds, 'column_names') and ds.column_names is not None:
            columns_to_remove = [c for c in ds.column_names if c not in ["question", "reasoning", "answer"]]
        else:
            # Fallback: try features if column_names not available
            if hasattr(ds, 'features') and ds.features is not None:
                columns_to_remove = [c for c in ds.features.keys() if c not in ["question", "reasoning", "answer"]]
            else:
                raise RuntimeError(
                    f"Cannot determine column names for dataset '{name}'. "
                    f"Dataset must have either 'column_names' attribute or 'features' attribute. "
                    f"This may indicate a problem with the dataset loading."
                )
        
        # Apply preprocessing with desc parameter (supported for non-streaming datasets)
        ds = ds.map(
            config["preprocess_fn"],
            remove_columns=columns_to_remove,
            desc=f"Preprocessing {name}"
        )
        
        print(f"Successfully loaded and preprocessed dataset {name}")
        
        return ds
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{name}' split '{split}': {e}")
