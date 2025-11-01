#!/usr/bin/env python3
"""
Test script for probability analysis.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probability_test import ProbabilityGenerator
from test import main

# For accuracy testing, use accuracy_test module
# from accuracy_test import main as accuracy_main

if __name__ == "__main__":
    main()
