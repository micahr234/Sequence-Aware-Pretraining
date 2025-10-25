#!/usr/bin/env python3
"""
Test script for probability analysis.
"""

import sys
import os

# Add the test_src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test_src'))

from test import main

if __name__ == "__main__":
    main()
