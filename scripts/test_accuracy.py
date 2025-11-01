#!/usr/bin/env python3
"""
Accuracy evaluation test script for Sequence-Aware Pretraining.

Usage:
    python scripts/test_accuracy.py [config_file]
    
Examples:
    python scripts/test_accuracy.py test_configs/baseline.yaml
    
Note: This script evaluates model accuracy on test datasets by downloading
      trained models from HuggingFace Hub and comparing predictions with ground truth.
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*", category=UserWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from accuracy_test import main

if __name__ == "__main__":
    main()

