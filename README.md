# Sequence-Aware Pretraining

A modern implementation of sequence-aware pretraining using discounted log-suffix weighting for improved language model training.

## Overview

This project implements **Discounted Log-Suffix SFT Training**, a novel approach that applies position-dependent weights to cross-entropy loss during supervised fine-tuning. Later tokens in a sequence receive higher weights according to a discount factor γ, encouraging the model to focus more on important, later parts of sequences.

## Key Features

- **🎯 Discounted Log-Suffix Weighting**: Position-based weighting for better sequence learning
- **⚡ Efficient Training**: Uses standard SFT training with custom loss function  
- **🔧 Configurable**: Easy to customize training parameters and datasets
- **📊 Multiple Datasets**: Support for GSM8K, SQuAD, and OpenBookQA
- **🎭 Flexible Formatting**: Customizable prompt templates
- **🚀 Modern Stack**: Built on Hugging Face Transformers and TRL

## Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd Sequence-Aware-Pretraining

# Install dependencies and setup environment
bash install.sh

# Activate virtual environment
source .venv/bin/activate
```

### Manual Installation

```bash
# Install dependencies
pip install -e .

# Or using uv (recommended)
uv pip install -e .
```

## Usage

### Basic Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py train_configs/baseline.yaml
```

### Configuration

Training is highly configurable through YAML files:

```yaml
# train_configs/default.yaml
model:
  base_name: "gpt2"                    # Base model to fine-tune
  output_name: "sequence-aware-model"  # Output model name

dataset:
  name: "c4"                          # Dataset to use (C4 for broad pretraining)
  max_examples: 1000                   # Limit dataset size

training:
  optimizer:
    lr: 2e-5                           # Learning rate
    weight_decay: 0.01                 # Weight decay
  
  schedule:
    num_epochs: 3                      # Number of epochs
    batch_size: 4                      # Batch size
    grad_accumulation_steps: 4         # Gradient accumulation
  
  sft:
    gamma: 0.98                        # Discount factor for position weighting
```

## Supported Datasets

### Broad Pretraining (Recommended)
- **C4**: Clean Common Crawl dataset - excellent for general language understanding
- **Wikipedia**: Wikipedia articles - good for factual knowledge
- **BookCorpus**: BookCorpus dataset - good for narrative text
- **OpenWebText**: Diverse web text - good for conversational understanding

### Specialized Tasks
- **GSM8K**: Math word problems - for reasoning tasks
- **SQuAD**: Reading comprehension - for QA tasks
- **OpenBookQA**: Science questions - for knowledge tasks

## Training Approach

### Discounted Log-Suffix Weighting

The training uses a position-based weighting scheme where:

- Position k in a sequence gets weight: `w_k = (1 - γ^k) / (1 - γ)`
- Higher γ values give more weight to later tokens
- When γ = 1, weights become linear: `w_k = k`
- This encourages the model to focus more on later, more important tokens

### Mathematical Foundation

For a sequence of length L with discount factor γ ∈ (0, 1]:

```
w_k = (1 - γ^k) / (1 - γ)  for k = 1, 2, ..., L
```

When γ = 1, the weights become linear: `w_k = k`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```