#!/usr/bin/env python3
"""
Script to run the FastAPI server with proper setup.
"""

import os
import sys
import logging
from pathlib import Path
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the FastAPI server."""
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
    
    # Run the server
    print("Starting Fake News Detection API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Health check endpoint: http://localhost:8000/health")
    
    uvicorn.run(
        "src.deployment.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()