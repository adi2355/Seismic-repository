#!/usr/bin/env python3
"""
Prepare diagnostic framework files for Google Colab.
This script creates a zip file with all necessary framework files.
"""

import zipfile
import os
from pathlib import Path

def create_colab_package():
    """Create a zip package with all diagnostic framework files."""
    
    # Files to include in the package
    framework_files = [
        'phase2b_diagnostic_experiments.py',
        'run_diagnostic_experiments.py',
        'test_diagnostic_framework.py',
        'PHASE2B_DIAGNOSTIC_README.md'
    ]
    
    # Check if all files exist
    missing_files = []
    for file in framework_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all framework files are present.")
        return False
    
    # Create zip package
    package_name = 'phase2b_diagnostic_framework.zip'
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in framework_files:
            zipf.write(file)
            print(f"✅ Added {file} to package")
    
    print(f"\n📦 Created package: {package_name}")
    print(f"📁 Package size: {os.path.getsize(package_name) / 1024:.1f} KB")
    
    print("\n🚀 Instructions for Google Colab:")
    print("1. Upload the modified notebook: MAIN_898_with_diagnostic_framework.ipynb")
    print("2. Run the first cell to upload diagnostic framework files")
    print("3. Upload this zip file and extract it, or upload files individually")
    print("4. Run the validation cell to ensure everything is working")
    print("5. Start with the quick diagnostic test (5 epochs)")
    
    return True

if __name__ == "__main__":
    print("📋 Preparing Diagnostic Framework for Google Colab")
    print("=" * 50)
    
    success = create_colab_package()
    
    if success:
        print("\n✅ Package ready for upload!")
    else:
        print("\n❌ Package creation failed!") 