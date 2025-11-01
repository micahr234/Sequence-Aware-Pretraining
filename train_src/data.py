import re

from datasets import Dataset, load_dataset


# ============================================================================
# Dataset-specific preprocessing functions
# ============================================================================

def preprocess_c4(example: dict) -> dict:
    """
    Preprocess C4 (Colossal Clean Crawled Corpus) dataset.
    
    C4 is already heavily preprocessed and cleaned. The text field contains
    clean web content suitable for direct use in pretraining.
    For pretraining datasets, all tokens are masked (True).
    
    Args:
        example: Raw example with "text" field
        
    Returns:
        Preprocessed example with "text" and "attention_mask" fields
    """
    text = example.get("text", "").strip()
    # C4 is already clean, just ensure we have valid text
    if not text or len(text) < 10:  # Filter very short texts
        return None
    
    # For pretraining, mask all characters (all True)
    attention_mask = [True] * len(text)
    
    return {"text": text, "attention_mask": attention_mask}


def preprocess_wikipedia(example: dict) -> dict:
    """
    Preprocess Wikipedia dataset.
    
    Wikipedia articles have title and text fields. We format them nicely
    for pretraining by including the title as a header.
    For pretraining datasets, all tokens are masked (True).
    
    Args:
        example: Raw example with "title" and "text" fields
        
    Returns:
        Preprocessed example with "text" and "attention_mask" fields
    """
    title = example.get("title", "").strip()
    text = example.get("text", "").strip()
    
    if not text or len(text) < 50:  # Filter very short articles
        return None
    
    # Format with title header for better context
    if title:
        formatted_text = f"Title: {title}\n\n{text}"
    else:
        formatted_text = text
    
    # For pretraining, mask all characters (all True)
    attention_mask = [True] * len(formatted_text)
    
    return {"text": formatted_text, "attention_mask": attention_mask}


def preprocess_bookcorpus(example: dict) -> dict:
    """
    Preprocess BookCorpus dataset.
    
    BookCorpus contains book text that may need basic cleaning.
    The text field contains raw book content.
    For pretraining datasets, all tokens are masked (True).
    
    Args:
        example: Raw example with "text" field
        
    Returns:
        Preprocessed example with "text" and "attention_mask" fields
    """
    text = example.get("text", "").strip()
    
    if not text or len(text) < 20:  # Filter very short passages
        return None
    
    # Basic cleaning: normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # For pretraining, mask all characters (all True)
    attention_mask = [True] * len(text)
    
    return {"text": text, "attention_mask": attention_mask}


def preprocess_openwebtext(example: dict) -> dict:
    """
    Preprocess OpenWebText dataset.
    
    OpenWebText contains web content scraped from Reddit links.
    Basic cleaning is applied to ensure quality text.
    For pretraining datasets, all tokens are masked (True).
    
    Args:
        example: Raw example with "text" field
        
    Returns:
        Preprocessed example with "text" and "attention_mask" fields
    """
    text = example.get("text", "").strip()
    
    if not text or len(text) < 20:  # Filter very short content
        return None
    
    # Basic cleaning: normalize excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces/tabs
    
    # For pretraining, mask all characters (all True)
    attention_mask = [True] * len(text)
    
    return {"text": text, "attention_mask": attention_mask}


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """
    Extract the final numerical answer from GSM8K answer format.
    
    GSM8K answers have format: "...rationale explaining steps... #### 42"
    We extract the numerical answer after the last '####' token.
    
    Args:
        answer_text: Full answer text with rationale and final answer
        
    Returns:
        Final numerical answer as string (with commas removed for formatting)
    """
    m = re.findall(r"####\s*(.+)", answer_text)
    if m:
        return m[-1].strip().replace(",", "")
    return answer_text.strip().replace(",", "")


def preprocess_gsm8k(example: dict) -> dict:
    """
    Preprocess GSM8K (Grade School Math 8K) dataset.
    
    GSM8K contains math word problems with questions and answers.
    Answers include a step-by-step rationale followed by the final answer
    marked with "####".
    
    Args:
        example: Raw example with "question" and "answer" fields
        
    Returns:
        Preprocessed example with "text" and "attention_mask" fields.
        Only the final answer (text after "####") has True in attention_mask.
        The reasoning portion before "####" is marked as False.
    """
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    
    # Format with question and full answer field
    formatted_text = f"{question}\n\n{answer}"
    
    # Create character-level attention mask: only mark final answer (after "####") as True
    # Find "####" marker in the formatted text
    marker_match = re.search(r"####\s*(.+)", formatted_text)
    if not marker_match:
        raise ValueError(f"GSM8K example missing '####' marker. Question: {question[:100]}...")
    
    # Get the final answer text (everything after "####" and whitespace)
    final_answer_text = marker_match.group(1).strip().replace(",", "")
    
    if not final_answer_text:
        raise ValueError(f"GSM8K example has empty final answer after '####' marker. Question: {question[:100]}...")
    
    # Find where the final answer starts in the formatted text
    # The match group 1 captures everything after "####", so we need to find
    # where this text actually appears (after stripping whitespace)
    text_after_marker = formatted_text[marker_match.end():]
    num_leading_spaces = len(text_after_marker) - len(text_after_marker.lstrip())
    final_answer_start_idx = marker_match.end() + num_leading_spaces
    
    # Mask: False for everything except the final answer
    attention_mask = [False] * len(formatted_text)
    # Mark only the final answer portion as True
    for i in range(final_answer_start_idx, min(final_answer_start_idx + len(final_answer_text), len(formatted_text))):
        attention_mask[i] = True
    
    return {"text": formatted_text, "attention_mask": attention_mask}


def preprocess_squad(example: dict) -> dict:
    """
    Preprocess SQuAD (Stanford Question Answering Dataset).
    
    SQuAD contains reading comprehension questions with context passages.
    Each example has a question, context, and answer(s).
    
    Args:
        example: Raw example with "question", "context", and "answers" fields
                where answers is a dict with "text" (list) and "answer_start" (list)
        
    Returns:
        Preprocessed example with "text" and "attention_mask" fields.
        Only answer tokens have True in attention_mask.
    """
    question = example.get("question", "").strip()
    context = example.get("context", "").strip()
    answers = example.get("answers", {})
    
    if not question:
        return None
    
    # Extract the first answer text
    answer_text = ""
    if answers and "text" in answers and answers["text"]:
        answer_text = answers["text"][0].strip()
    
    # Format as question-answer pair
    # Optionally include context for richer training data
    if context and answer_text:
        formatted_text = f"Question: {question}\nContext: {context}\nAnswer: {answer_text}"
        # Find where answer starts (after "Answer: ")
        answer_start_idx = formatted_text.find(answer_text)
        if answer_start_idx >= 0:
            attention_mask = [False] * answer_start_idx + [True] * len(answer_text)
        else:
            attention_mask = [False] * len(formatted_text)
    elif answer_text:
        formatted_text = f"Question: {question}\nAnswer: {answer_text}"
        # Find where answer starts (after "Answer: ")
        answer_start_idx = formatted_text.find(answer_text)
        if answer_start_idx >= 0:
            attention_mask = [False] * answer_start_idx + [True] * len(answer_text)
        else:
            attention_mask = [False] * len(formatted_text)
    else:
        formatted_text = f"Question: {question}"
        # No answer, so all False (we'll filter these out)
        attention_mask = [False] * len(formatted_text)
    
    return {"text": formatted_text, "attention_mask": attention_mask}


def preprocess_openbookqa(example: dict) -> dict:
    """
    Preprocess OpenBookQA dataset.
    
    OpenBookQA contains science questions with multiple choice answers.
    Each example has a question stem, multiple choices, and a correct answer key.
    
    Args:
        example: Raw example with "question_stem", "choices" (dict with "text" and "label"),
                 and "answerKey" fields
        
    Returns:
        Preprocessed example with "text" and "attention_mask" fields.
        Only answer tokens have True in attention_mask.
    """
    question_stem = example.get("question_stem", "").strip()
    choices = example.get("choices", {})
    answer_key = example.get("answerKey", "").strip().upper()
    
    if not question_stem or not choices or not answer_key:
        return None
    
    # Extract choice texts
    choice_texts = choices.get("text", [])
    if not choice_texts:
        return None
    
    # Map answer key (A, B, C, D) to index
    if answer_key and len(answer_key) == 1 and answer_key.isalpha():
        choice_idx = ord(answer_key) - ord('A')
        if 0 <= choice_idx < len(choice_texts):
            correct_answer = choice_texts[choice_idx].strip()
            
            formatted_text = f"Question: {question_stem}\nAnswer: {correct_answer}"
            
            # Create character-level attention mask: False for question, True for answer
            # Find where answer starts (after "Answer: ")
            answer_start_idx = formatted_text.find(correct_answer)
            if answer_start_idx >= 0:
                attention_mask = [False] * answer_start_idx + [True] * len(correct_answer)
            else:
                attention_mask = [False] * len(formatted_text)
            
            return {"text": formatted_text, "attention_mask": attention_mask}
    
    return None


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
            "preprocess_fn": preprocess_c4,
            "streaming": True
        },
        "wikipedia": {
            "dataset_name": "wikipedia",
            "config": "20220301.en",
            "preprocess_fn": preprocess_wikipedia,
            "streaming": False
        },
        "bookcorpus": {
            "dataset_name": "bookcorpus",
            "config": None,
            "preprocess_fn": preprocess_bookcorpus,
            "streaming": False
        },
        "openwebtext": {
            "dataset_name": "openwebtext",
            "config": None,
            "preprocess_fn": preprocess_openwebtext,
            "streaming": False
        },
        "gsm8k": {
            "dataset_name": "gsm8k",
            "config": "main",
            "preprocess_fn": preprocess_gsm8k,
            "streaming": False
        },
        "squad": {
            "dataset_name": "squad",
            "config": None,
            "preprocess_fn": preprocess_squad,
            "streaming": False
        },
        "openbookqa": {
            "dataset_name": "openbookqa",
            "config": "main",
            "preprocess_fn": preprocess_openbookqa,
            "streaming": False
        }
    }
    
    name = name.lower()
    
    if name not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS.keys()))
        raise ValueError(f"Unknown dataset: {name}. Supported: {supported}")
    
    try:
        config = DATASET_CONFIGS[name]
        
        # Handle streaming vs non-streaming mode differently
        if config["streaming"]:
            # For streaming mode, load the full split and then take the required number
            split_str = split
            if config["config"]:
                print(f"Loading dataset {config['dataset_name']} with config {config['config']} and split {split_str} (streaming mode)")
                ds = load_dataset(config["dataset_name"], config["config"], split=split_str, streaming=True)
            else:
                print(f"Loading dataset {config['dataset_name']} with split {split_str} (streaming mode)")
                ds = load_dataset(config["dataset_name"], split=split_str, streaming=True)
            
            # Apply max_examples limit using take() for streaming datasets
            if max_examples and max_examples > 0:
                print(f"Taking first {max_examples} examples from streaming dataset")
                ds = ds.take(max_examples)
        else:
            # For non-streaming mode, use split slicing
            split_str = f"{split}[:{max_examples}]" if max_examples and max_examples > 0 else split
            
            # Load dataset
            if config["config"]:
                print(f"Loading dataset {config['dataset_name']} with config {config['config']} and split {split_str}")
                ds = load_dataset(config["dataset_name"], config["config"], split=split_str, streaming=False)
            else:
                print(f"Loading dataset {config['dataset_name']} with split {split_str}")
                ds = load_dataset(config["dataset_name"], split=split_str, streaming=False)
        
        # Apply dataset-specific preprocessing
        print(f"Applying preprocessing for {name} dataset...")
        
        # Wrap preprocessing function to handle None returns properly
        def preprocess_wrapper(ex):
            result = config["preprocess_fn"](ex)
            if result is None:
                return {"text": "", "attention_mask": []}  # Return empty for filtering
            return result
        
        ds = ds.map(
            preprocess_wrapper,
            remove_columns=[c for c in ds.column_names if c not in ["text", "attention_mask"]],
            desc=f"Preprocessing {name}"
        )
        
        # Filter out invalid examples (empty or None text)
        ds = ds.filter(lambda x: x.get("text") is not None and len(x.get("text", "").strip()) > 0 and len(x.get("attention_mask", [])) > 0)
        
        # For streaming datasets, convert to regular Dataset for easier handling
        if config["streaming"] and max_examples and max_examples > 0:
            # Convert streaming dataset to regular dataset
            examples = list(ds)
            ds = Dataset.from_list(examples)
        
        print(f"Successfully loaded and preprocessed {len(ds)} examples from {name}")
        return ds
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{name}' split '{split}': {e}")
