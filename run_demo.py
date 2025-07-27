#!/usr/bin/env python3
"""
Script to run the Streamlit web demo with proper setup.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit demo."""
    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)
    
    # Check if datasets are extracted
    raw_data_dir = Path("data/raw")
    zip_files = list(raw_data_dir.glob("*.zip"))
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if zip_files and not csv_files:
        print("Extracting datasets...")
        import zipfile
        for zip_file in zip_files:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(raw_data_dir)
                print(f"Extracted {zip_file.name}")
            except Exception as e:
                print(f"Failed to extract {zip_file.name}: {e}")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    
    # Run the demo
    print("Starting Fake News Detection Web Demo...")
    print("Demo will be available at: http://localhost:8501")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "web_demo.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    main()