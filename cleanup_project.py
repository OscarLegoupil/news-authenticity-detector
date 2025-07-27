#!/usr/bin/env python3
"""
Project cleanup script to organize files and prepare for git commits.
Run this script to clean up the project structure.
"""

import os
import shutil
import sys
from pathlib import Path

def cleanup_project():
    """Execute project cleanup steps."""
    
    print("Starting project cleanup...")
    
    # 1. Create necessary directories
    print("1. Creating necessary directories...")
    directories = [
        "notebooks/archive",
        "data/raw", 
        "data/processed",
        "models/checkpoints",
        "results/experiments"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    # 2. Move legacy notebooks to archive
    print("\n2. Moving legacy notebooks to archive...")
    notebooks_to_archive = [
        "ProjectFakeNewsDetection.ipynb",
        "Text_ClassificationLeMonde_Legoupil.ipynb"
    ]
    
    for notebook in notebooks_to_archive:
        if os.path.exists(notebook):
            destination = f"notebooks/archive/{notebook}"
            shutil.move(notebook, destination)
            print(f"   Moved: {notebook} -> {destination}")
        else:
            print(f"   Not found: {notebook}")
    
    # 3. Remove obsolete directories and files
    print("\n3. Removing obsolete files and directories...")
    to_remove = [
        "Session2_Embeddings_Exploration",
        "language_model_EN_(1).ipynb"
    ]
    
    for item in to_remove:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"   Removed directory: {item}")
            else:
                os.remove(item)
                print(f"   Removed file: {item}")
        else:
            print(f"   Not found: {item}")
    
    # 4. Move data files to proper location
    print("\n4. Moving data files to data/raw/...")
    data_files = [
        "Fake.csv.zip",
        "True.csv.zip", 
        "fake_or_real_news.csv.zip"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            destination = f"data/raw/{data_file}"
            shutil.move(data_file, destination)
            print(f"   Moved: {data_file} -> {destination}")
        else:
            print(f"   Not found: {data_file}")
    
    # 5. Check file structure
    print("\n5. Final project structure:")
    print_directory_structure(".", max_depth=3)
    
    print("\nProject cleanup completed!")
    print("\nNext steps:")
    print("1. Review the git commit strategy in the console output")
    print("2. Execute the git commands to commit changes in logical groups")
    print("3. Run 'python train_pipeline.py --skip-deberta' to test the pipeline")

def print_directory_structure(path, prefix="", max_depth=2, current_depth=0):
    """Print directory structure with limited depth."""
    if current_depth >= max_depth:
        return
    
    path = Path(path)
    if not path.is_dir():
        return
    
    # Get all items and sort them
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        # Skip hidden files and __pycache__
        if item.name.startswith('.') or item.name == '__pycache__':
            continue
        
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth - 1:
            extension = "    " if is_last else "│   "
            print_directory_structure(item, prefix + extension, max_depth, current_depth + 1)

def show_git_commands():
    """Display the git commands for committing changes."""
    print("\n" + "="*60)
    print("GIT COMMIT STRATEGY")
    print("="*60)
    
    commands = [
        {
            "title": "Commit 1: Add project structure and configuration files",
            "commands": [
                "git add .gitignore requirements.txt configs/ data/ tests/ notebooks/",
                "git commit -m \"Add project structure and configuration files\n\n- Add comprehensive .gitignore for Python ML projects\n- Add requirements.txt with all necessary dependencies\n- Create configs/ directory with default YAML configuration\n- Set up data/ structure with raw/ and processed/ subdirectories\n- Create tests/ structure for unit and integration tests\n- Create notebooks/archive/ for legacy notebook organization\""
            ]
        },
        {
            "title": "Commit 2: Implement core data preprocessing and traditional models", 
            "commands": [
                "git add src/data/ src/models/traditional/ src/__init__.py src/models/__init__.py",
                "git commit -m \"Implement data preprocessing and traditional ML models\n\n- Add TextPreprocessor with traditional and transformer preprocessing\n- Add DatasetLoader for ISOT and Kaggle datasets with label standardization\n- Implement BOWClassifier and TFIDFClassifier with cross-validation\n- Add comprehensive error handling and logging\n- Include model serialization and cross-domain evaluation capabilities\""
            ]
        },
        {
            "title": "Commit 3: Implement DeBERTa-v3 training pipeline",
            "commands": [
                "git add src/models/transformers/",
                "git commit -m \"Implement DeBERTa-v3 fine-tuning pipeline\n\n- Add DeBERTaClassifier with configurable hyperparameters\n- Implement FakeNewsDataset for efficient data loading\n- Add cross-domain evaluation capabilities for transformer models\n- Include model saving and loading functionality\n- Add comprehensive metrics computation during training\""
            ]
        },
        {
            "title": "Commit 4: Add ensemble methods and confidence calibration",
            "commands": [
                "git add src/models/ensemble.py src/models/calibration.py",
                "git commit -m \"Add ensemble methods and confidence calibration\n\n- Implement EnsembleClassifier with weighted voting, stacking, and confidence-based methods\n- Add ModelCalibrator with Platt scaling and isotonic regression\n- Include confidence interval calculation and reliability diagrams\n- Add ConfidenceBasedFilter for production deployment\n- Implement comprehensive calibration evaluation metrics\""
            ]
        },
        {
            "title": "Commit 5: Add evaluation framework and cross-domain metrics",
            "commands": [
                "git add src/evaluation/",
                "git commit -m \"Add comprehensive evaluation framework\n\n- Implement CrossDomainEvaluator with stability and robustness testing\n- Add ModelBenchmark for comprehensive model comparison\n- Include performance visualization and comparison reports\n- Add adversarial robustness evaluation with text perturbations\n- Implement calibration curve plotting and analysis tools\""
            ]
        },
        {
            "title": "Commit 6: Add main pipeline and training scripts",
            "commands": [
                "git add src/pipeline.py train_pipeline.py demo_usage.py",
                "git commit -m \"Add main pipeline orchestration and training scripts\n\n- Implement FakeNewsDetector as main pipeline interface\n- Add comprehensive training pipeline with error handling\n- Include demo script for easy testing and validation\n- Add configuration loading from YAML files\n- Implement end-to-end workflow from data loading to inference\""
            ]
        },
        {
            "title": "Commit 7: Update documentation and project roadmap",
            "commands": [
                "git add README.md",
                "git commit -m \"Update README with implementation status and usage examples\n\n- Update development roadmap with completed phases\n- Add quick start guide with installation and usage instructions\n- Include training pipeline commands and options\n- Document new modular architecture and capabilities\""
            ]
        }
    ]
    
    for i, commit in enumerate(commands, 1):
        print(f"\n{i}. {commit['title']}")
        print("-" * 50)
        for cmd in commit['commands']:
            print(f"   {cmd}")

if __name__ == "__main__":
    cleanup_project()
    show_git_commands()