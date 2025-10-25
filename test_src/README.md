# Probability Analysis Test Framework

This framework provides tools for analyzing the probability distributions of language models during text generation. It generates multiple samples and calculates probability mass for each token at each timestep.

## Features

- **Multi-sample Generation**: Generate multiple samples with the same prompt to analyze probability distributions
- **Timestep Analysis**: Calculate probability mass for each token at each generation timestep
- **Statistical Analysis**: Compute entropy, token frequencies, and probability statistics
- **Visualization**: Create heatmaps, entropy plots, and frequency distributions
- **Flexible Configuration**: YAML-based configuration for different test scenarios

## Quick Start

### 1. Basic Usage

```bash
# Run with default configuration
python scripts/test.py test_configs/default.yaml

# Run with custom prompt
python scripts/test.py test_configs/baseline.yaml --prompt "The quick brown fox"

# Override number of samples
python scripts/test.py test_configs/baseline.yaml --num-samples 200
```

### 2. Configuration Files

The framework uses YAML configuration files in the `test_configs/` directory:

- `default.yaml`: Basic configuration for testing
- `baseline.yaml`: Configuration for testing the sequence-aware baseline model
- `comparison.yaml`: Configuration for comparing multiple models

### 3. Output

The framework generates:

- **JSON/Pickle files**: Raw generation data and probability statistics
- **Visualizations**: Heatmaps, entropy plots, token frequency plots
- **HTML Report**: Summary report with all visualizations

## Configuration Options

### Model Configuration
```yaml
model:
  name: "gpt2"  # Model name or path
  device: "auto"  # Device (auto, cuda, cpu)
```

### Generation Parameters
```yaml
generation:
  max_new_tokens: 50  # Maximum tokens to generate
  num_samples: 100  # Number of generation samples
  temperature: 1.0  # Sampling temperature
  top_p: 0.9  # Nucleus sampling
  top_k: 50  # Top-k sampling
  do_sample: true  # Use sampling vs greedy
```

### Analysis Options
```yaml
analysis:
  save_logits: true  # Save raw logits
  save_probabilities: true  # Save probability distributions
  top_k_probs: 10  # Number of top probabilities to save
  output_dir: "test_results"  # Output directory
  save_format: "json"  # Output format (json, pickle, both)
  save_visualizations: true  # Generate plots
```

## Generated Visualizations

1. **Probability Heatmap**: Shows probability distributions over time
2. **Entropy Over Time**: Plots uncertainty evolution during generation
3. **Token Frequencies**: Shows how often each token appears at each timestep
4. **Generation Samples**: Displays sample generated sequences

## Example Usage

### Test the Baseline Model
```bash
python scripts/test.py test_configs/baseline.yaml
```

This will:
- Load the trained sequence-aware baseline model
- Generate 200 samples with the configured prompt
- Create probability heatmaps and entropy plots
- Save results to `test_results/baseline/`

### Compare Models
```bash
# Test base GPT-2
python scripts/test.py test_configs/default.yaml --prompt "In a world where AI"

# Test sequence-aware model
python scripts/test.py test_configs/baseline.yaml --prompt "In a world where AI"
```

### Custom Analysis
```bash
python scripts/test.py test_configs/default.yaml \
  --prompt "The future of technology" \
  --num-samples 500 \
  --max-tokens 100
```

## Output Structure

```
test_results/
├── prompt_0/
│   ├── generation_results.json
│   ├── generation_results.pkl
│   ├── probability_heatmap.png
│   ├── entropy_over_time.png
│   ├── token_frequencies.png
│   ├── generation_samples.png
│   └── analysis_report.html
└── ...
```

## Key Metrics

The framework calculates several important metrics:

- **Token Frequencies**: How often each token appears at each timestep
- **Probability Means/Stds**: Average and standard deviation of probability distributions
- **Entropy**: Uncertainty measure at each timestep
- **Top-K Probabilities**: Most likely tokens and their probabilities

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Matplotlib
- Seaborn
- NumPy
- OmegaConf
- tqdm

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `num_samples` or `max_new_tokens`
2. **Model Not Found**: Check the model path in configuration
3. **Empty Results**: Ensure the model can generate text with the given prompt

### Performance Tips

- Use smaller `num_samples` for faster analysis
- Reduce `max_new_tokens` for shorter sequences
- Use `device: "cpu"` if GPU memory is limited
