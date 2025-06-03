#!/usr/bin/env python3
"""
Test script for FiLM validation - includes all necessary imports
"""

# Essential imports
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add path for any local modules if needed
sys.path.append('.')

print("🔧 STARTING FiLM VALIDATION TEST")
print("="*60)

# Check basic requirements
print("📋 CHECKING BASIC REQUIREMENTS:")
print(f"   Python version: {sys.version}")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Working directory: {os.getcwd()}")

# Check if required files exist
required_files = [
    'main_898of_0_898model_speed_and_structure_starter_notebook.py',
    'complete_sincgat_unet_integration.py',
    'phase2_experimental_framework.py'
]

print("\n📁 CHECKING REQUIRED FILES:")
all_files_exist = True
for file in required_files:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n❌ MISSING REQUIRED FILES - Cannot proceed with validation")
    print("   Please ensure all required files are in the working directory")
    sys.exit(1)

# Try to load the main script
print("\n🔄 LOADING MAIN SCRIPT...")
try:
    # Load the script content
    with open('main_898of_0_898model_speed_and_structure_starter_notebook.py', 'r') as f:
        script_content = f.read()
    
    # Execute the script in the current namespace
    exec(script_content)
    print("   ✅ Main script loaded successfully")
    
    # Check if key functions are defined
    key_functions = [
        'run_corrected_film_validation',
        'train_with_film_awareness', 
        'Stage2ExperimentalFramework'
    ]
    
    print("\n🔍 CHECKING KEY FUNCTIONS:")
    for func_name in key_functions:
        if func_name in globals():
            print(f"   ✅ {func_name}")
        else:
            print(f"   ❌ {func_name}")
    
    # Run basic validation
    print("\n🧪 RUNNING BASIC VALIDATION...")
    if 'run_corrected_film_validation' in globals():
        validation_result = run_corrected_film_validation()
        print(f"\n📊 VALIDATION RESULT: {validation_result.get('status', 'unknown')}")
        
        if validation_result.get('status') == 'success':
            print("🎉 VALIDATION SUCCESSFUL - READY FOR FiLM EXPERIMENTS!")
        else:
            error = validation_result.get('error', 'unknown error')
            print(f"⚠️ VALIDATION ISSUES: {error}")
    else:
        print("❌ Validation function not available")

except Exception as e:
    print(f"❌ ERROR LOADING SCRIPT: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🏁 VALIDATION TEST COMPLETE") 