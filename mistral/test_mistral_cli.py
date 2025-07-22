#!/usr/bin/env python3
"""
Test script for Mistral CLI to verify functionality
"""

import subprocess
import os
import sys

def test_single_text():
    """Test classification of a single text."""
    print("Testing single text classification...")
    
    # Test text from the notebook
    test_text = "Title: Quartette für zwei Violinen, Viola, Violoncell\\nSubjects: String quartets--Scores"
    
    cmd = [
        sys.executable,
        "predict_mistral_classifier.py",
        "--text", test_text,
        "--verbose"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        print("\nOutput:")
        print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def test_help():
    """Test help output."""
    print("\nTesting help output...")
    
    cmd = [
        sys.executable,
        "predict_mistral_classifier.py",
        "--help"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        print("\nHelp output:")
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("MISTRAL_API_KEY"):
        print("WARNING: MISTRAL_API_KEY environment variable not set")
        print("The CLI will fail without a valid API key")
        print("\nTo set it: export MISTRAL_API_KEY='your-api-key'")
    
    print("Running Mistral CLI tests...\n")
    
    # Test help
    if test_help():
        print("✓ Help test passed")
    else:
        print("✗ Help test failed")
    
    # Only test actual classification if API key is available
    if os.environ.get("MISTRAL_API_KEY"):
        print("\n" + "="*50 + "\n")
        if test_single_text():
            print("✓ Single text test passed")
        else:
            print("✗ Single text test failed")
    else:
        print("\nSkipping classification test (no API key)")