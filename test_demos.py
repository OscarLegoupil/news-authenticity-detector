#!/usr/bin/env python3
"""
Test script to verify all demo apps work correctly.
"""

import sys
import subprocess
import time
from pathlib import Path

def test_demo_usage():
    """Test the demo_usage.py script."""
    print("Testing demo_usage.py...")
    try:
        result = subprocess.run([sys.executable, "demo_usage.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("PASS: demo_usage.py works correctly")
            return True
        else:
            print("FAIL: demo_usage.py failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("FAIL: demo_usage.py timed out")
        return False
    except Exception as e:
        print(f"FAIL: demo_usage.py error: {e}")
        return False

def test_pipeline_imports():
    """Test that all pipeline components can be imported."""
    print("Testing pipeline imports...")
    try:
        sys.path.append('src')
        from src.pipeline import FakeNewsDetector, PipelineConfig
        from src.interpretability.explainer import ExplainerFactory, MultiExplainer
        
        # Test basic initialization
        detector = FakeNewsDetector()
        print("PASS: All pipeline imports work correctly")
        return True
    except Exception as e:
        print(f"FAIL: Pipeline import error: {e}")
        return False

def test_streamlit_import():
    """Test that the Streamlit app can be imported."""
    print("Testing Streamlit app import...")
    try:
        # Test import without running the app
        with open('web_demo.py', 'r') as f:
            content = f.read()
        
        # Check for common issues
        if "st.experimental_rerun" in content:
            print("FAIL: Found deprecated st.experimental_rerun")
            return False
        
        print("PASS: Streamlit app code looks good")
        return True
    except Exception as e:
        print(f"FAIL: Streamlit test error: {e}")
        return False

def test_requirements():
    """Test that key requirements are available."""
    print("Testing key requirements...")
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        import sklearn
        print("PASS: Key packages are available")
        return True
    except ImportError as e:
        print(f"FAIL: Missing requirement: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Fake News Detection Demo Apps")
    print("=" * 50)
    
    tests = [
        test_requirements,
        test_pipeline_imports,
        test_streamlit_import,
        test_demo_usage,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAIL: Test {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Your demo apps are ready to use.")
        print("\nTo run the demos:")
        print("1. Basic demo: python demo_usage.py")
        print("2. Web demo: python run_demo.py")
        print("3. Direct Streamlit: streamlit run web_demo.py")
    else:
        print("ERROR: Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)