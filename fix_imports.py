#!/usr/bin/env python
"""
Fix import issues by adding proper path configuration
Add this to the top of your scripts that have import issues
"""

import sys
import os

# Add the biobank_classification directory to Python path
biobank_dir = r"/Users/george/Documents/PhD/papagei-foundation-model/biobank_classification"
if biobank_dir not in sys.path:
    sys.path.insert(0, biobank_dir)

# Alternative: Add parent directory if running as package
parent_dir = os.path.dirname(biobank_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Python path configured. Biobank modules should now import correctly.")
print(f"Biobank dir: {biobank_dir}")
