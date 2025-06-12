#!/usr/bin/env python
"""
Debug and fix Python import issues for biobank_classification
"""

import os
import sys
import importlib.util

print("=== PYTHON IMPORT DEBUGGER ===\n")

# 1. Check current working directory
print("1. CURRENT DIRECTORY:")
print(f"   Working dir: {os.getcwd()}")
print(f"   Script location: {os.path.abspath(__file__)}")
print(f"   Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# 2. Check Python path
print("\n2. PYTHON PATH:")
for i, path in enumerate(sys.path):
    print(f"   [{i}] {path}")

# 3. Check if biobank_classification directory exists
print("\n3. CHECKING BIOBANK_CLASSIFICATION DIRECTORY:")
biobank_dir = os.path.dirname(os.path.abspath(__file__))
print(f"   Directory: {biobank_dir}")
print(f"   Exists: {os.path.exists(biobank_dir)}")

# 4. List all Python files in directory
print("\n4. PYTHON FILES IN DIRECTORY:")
if os.path.exists(biobank_dir):
    py_files = [f for f in os.listdir(biobank_dir) if f.endswith(".py")]
    for f in sorted(py_files):
        print(f"   - {f}")

# 5. Check for __init__.py
print("\n5. CHECKING FOR __init__.py:")
init_file = os.path.join(biobank_dir, "__init__.py")
print(f"   __init__.py exists: {os.path.exists(init_file)}")
if not os.path.exists(init_file):
    print("   ⚠️  WARNING: No __init__.py file! This might cause import issues.")

# 6. Try importing modules with detailed error info
print("\n6. TESTING IMPORTS:")

modules_to_test = [
    "biobank_experiment_utils",
    "biobank_classification_utils",
    "biobank_experiment_constants",
    "biobank_reporting_utils",
    "biobank_classification_functions",
    "biobank_imbalance_handling",
    "biobank_feature_functions",
]

failed_imports = []

for module_name in modules_to_test:
    try:
        # Try direct import
        module = __import__(module_name)
        print(f"   ✓ {module_name} - imported successfully")
    except ImportError as e:
        print(f"   ✗ {module_name} - FAILED: {str(e)}")
        failed_imports.append((module_name, str(e)))

        # Try to find the file
        module_file = os.path.join(biobank_dir, f"{module_name}.py")
        if os.path.exists(module_file):
            print(f"      File exists at: {module_file}")
        else:
            print(f"      File NOT FOUND at: {module_file}")

# 7. Fix attempts
print("\n7. ATTEMPTING FIXES:")

# Fix 1: Add current directory to Python path
if biobank_dir not in sys.path:
    print(f"   Adding {biobank_dir} to Python path...")
    sys.path.insert(0, biobank_dir)

    # Retry failed imports
    print("   Retrying imports after adding to path...")
    for module_name, _ in failed_imports:
        try:
            module = __import__(module_name)
            print(f"   ✓ {module_name} - NOW WORKS!")
        except ImportError:
            print(f"   ✗ {module_name} - Still failing")

# Fix 2: Create __init__.py if missing
if not os.path.exists(init_file):
    print(f"\n   Creating __init__.py...")
    try:
        with open(init_file, "w") as f:
            f.write("# Auto-generated __init__.py\n")
        print("   ✓ Created __init__.py")
    except Exception as e:
        print(f"   ✗ Failed to create __init__.py: {e}")

# 8. Generate import fix script
print("\n8. GENERATING FIX SCRIPT:")
fix_script = os.path.join(biobank_dir, "fix_imports.py")

fix_code = f'''#!/usr/bin/env python
"""
Fix import issues by adding proper path configuration
Add this to the top of your scripts that have import issues
"""

import sys
import os

# Add the biobank_classification directory to Python path
biobank_dir = r"{biobank_dir}"
if biobank_dir not in sys.path:
    sys.path.insert(0, biobank_dir)

# Alternative: Add parent directory if running as package
parent_dir = os.path.dirname(biobank_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Python path configured. Biobank modules should now import correctly.")
print(f"Biobank dir: {{biobank_dir}}")
'''

with open(fix_script, "w") as f:
    f.write(fix_code)

print(f"   Created: {fix_script}")
print("\n   To fix imports in any script, add this at the top:")
print("   ```python")
print("   import sys")
print(f"   sys.path.insert(0, r'{biobank_dir}')")
print("   ```")

# 9. Test with fixed path
print("\n9. TESTING WITH FIXED PATH:")
sys.path.insert(0, biobank_dir)

try:
    # Try importing a module that uses other modules
    from biobank_experiment_utils import load_yaml_config

    print("   ✓ Complex import successful! Your imports should work now.")
except Exception as e:
    print(f"   ✗ Complex import failed: {e}")

# 10. Virtual environment check
print("\n10. VIRTUAL ENVIRONMENT CHECK:")
print(f"   Python executable: {sys.executable}")
print(
    f"   Virtual env: {hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)}"
)
if "VIRTUAL_ENV" in os.environ:
    print(f"   VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']}")

print("\n=== SUMMARY ===")
print("\nTo fix your imports, do ONE of these:")
print("\n1. Run from the correct directory:")
print(f"   cd {biobank_dir}")
print("   python your_script.py")
print("\n2. Add this to the top of your scripts:")
print("   import sys")
print(f"   sys.path.insert(0, r'{biobank_dir}')")
print("\n3. Run as a module from parent directory:")
print(f"   cd {os.path.dirname(biobank_dir)}")
print("   python -m biobank_classification.your_script")
print("\n4. Install as editable package (best long-term solution):")
print(f"   cd {os.path.dirname(biobank_dir)}")
print("   pip install -e .")
print("   (requires setup.py in parent directory)")
