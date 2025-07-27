#!/usr/bin/env python3
"""
Main training script for fake news detection pipeline.
Demonstrates the complete workflow from data loading to model evaluation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import FakeNewsDetector, PipelineConfig
from src.evaluation.metrics import CrossDomainEvaluator, ModelBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-deberta', action='store_true',
                       help='Skip DeBERTa training (requires GPU)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced data')
    
    args = parser.parse_args()
    
    logger.info("Starting fake news detection training pipeline...")
    
    try:
        # Initialize pipeline
        if os.path.exists(args.config):
            detector = FakeNewsDetector.from_config_file(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            detector = FakeNewsDetector()
            logger.info("Using default configuration")
        
        # Step 1: Load datasets
        logger.info("Step 1: Loading datasets...")
        datasets = detector.load_datasets()
        
        if not datasets:
            logger.error("No datasets loaded. Please check data paths in configuration.")
            return
        
        for name, df in datasets.items():
            logger.info(f"Loaded {name}: {len(df)} articles")
        
        # Step 2: Train traditional models
        logger.info("Step 2: Training traditional ML models...")
        traditional_results = detector.train_traditional_models('isot')
        
        for model_name, results in traditional_results.items():
            logger.info(f"{model_name} CV accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        
        # Step 3: Train DeBERTa model (optional)
        if not args.skip_deberta:
            logger.info("Step 3: Training DeBERTa model...")
            try:
                deberta_results = detector.train_transformer_model('isot')
                logger.info("DeBERTa training completed successfully")
            except Exception as e:
                logger.error(f"DeBERTa training failed: {e}")
                logger.info("Continuing without DeBERTa model...")
        else:
            logger.info("Step 3: Skipping DeBERTa training")
        
        # Step 4: Create ensemble
        logger.info("Step 4: Creating ensemble model...")
        try:
            ensemble_model = detector.create_ensemble()
            logger.info("Ensemble model created successfully")
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            logger.info("Continuing without ensemble...")
        
        # Step 5: Cross-domain evaluation
        if 'isot' in datasets and 'kaggle' in datasets:
            logger.info("Step 5: Running cross-domain evaluation...")
            
            try:
                cross_domain_results = detector.evaluate_cross_domain('isot', 'kaggle')
                
                logger.info("Cross-domain evaluation results:")
                for model_name, results in cross_domain_results.items():
                    if isinstance(results, dict) and 'accuracy' in results:
                        logger.info(f"{model_name}: {results['accuracy']:.4f} accuracy, {results['f1_score']:.4f} F1")
                
            except Exception as e:
                logger.error(f"Cross-domain evaluation failed: {e}")
        else:
            logger.warning("Cannot perform cross-domain evaluation: missing datasets")
        
        # Step 6: Save models
        logger.info("Step 6: Saving trained models...")
        try:
            detector.save_models()
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
        
        # Step 7: Save configuration
        config_save_path = "results/final_config.yaml"
        os.makedirs("results", exist_ok=True)
        detector.save_config(config_save_path)
        
        logger.info("Training pipeline completed successfully!")
        
        # Demo: Make some predictions
        logger.info("Demo: Making sample predictions...")
        sample_texts = [
            "Scientists at Harvard University have discovered a new method for treating cancer using targeted therapy.",
            "BREAKING: Celebrity spotted doing something completely normal, you won't believe what happens next!",
            "The Federal Reserve announced new monetary policy measures to address inflation concerns."
        ]
        
        # Try different models
        for model_type in ['tfidf', 'ensemble']:  # Skip DeBERTa if not trained
            if (model_type == 'ensemble' and hasattr(detector, 'ensemble_model') and detector.ensemble_model is not None) or \
               (model_type in detector.traditional_models):
                try:
                    results = detector.predict(sample_texts, model_type=model_type)
                    logger.info(f"\n{model_type.upper()} predictions:")
                    for i, (text, label, conf) in enumerate(zip(sample_texts, results['labels'], results['confidence'])):
                        logger.info(f"Text {i+1}: {label} (confidence: {conf:.3f})")
                        logger.info(f"Preview: {text[:100]}...")
                except Exception as e:
                    logger.error(f"Prediction failed for {model_type}: {e}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()