"""
Integration tests for the complete fake news detection pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.pipeline import FakeNewsDetector, PipelineConfig


class TestFakeNewsDetectorIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic dataset for testing
        self.fake_texts = [
            "SHOCKING discovery that doctors don't want you to know!",
            "BREAKING: Celebrity caught doing something unbelievable!",
            "You won't believe what happened next in this AMAZING story!",
            "URGENT: Government trying to hide this INCREDIBLE secret!",
            "Scientists HATE this simple trick that will change your life!"
        ]
        
        self.real_texts = [
            "Researchers at Stanford University published findings today.",
            "The Federal Reserve announced new monetary policy measures.",
            "Climate scientists report new data on temperature trends.",
            "Government officials met to discuss infrastructure development.",
            "Medical researchers completed a clinical trial with results."
        ]
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_datasets()
        
        # Create test configuration
        self.test_config = PipelineConfig(
            isot_fake_path=os.path.join(self.temp_dir, "fake.csv"),
            isot_real_path=os.path.join(self.temp_dir, "real.csv"),
            kaggle_path=os.path.join(self.temp_dir, "kaggle.csv"),
            traditional_max_features=50,  # Reduce for faster testing
            models_dir=os.path.join(self.temp_dir, "models"),
            results_dir=os.path.join(self.temp_dir, "results")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_datasets(self):
        """Create test CSV files."""
        # Create fake news CSV
        fake_df = pd.DataFrame({
            'title': [f"Fake Title {i}" for i in range(len(self.fake_texts))],
            'text': self.fake_texts,
            'subject': ['politics'] * len(self.fake_texts),
            'date': ['2023-01-01'] * len(self.fake_texts)
        })
        fake_df.to_csv(os.path.join(self.temp_dir, "fake.csv"), index=False)
        
        # Create real news CSV
        real_df = pd.DataFrame({
            'title': [f"Real Title {i}" for i in range(len(self.real_texts))],
            'text': self.real_texts,
            'subject': ['politics'] * len(self.real_texts),
            'date': ['2023-01-01'] * len(self.real_texts)
        })
        real_df.to_csv(os.path.join(self.temp_dir, "real.csv"), index=False)
        
        # Create Kaggle format CSV
        all_texts = self.fake_texts + self.real_texts
        all_labels = ['FAKE'] * len(self.fake_texts) + ['REAL'] * len(self.real_texts)
        
        kaggle_df = pd.DataFrame({
            'title': [f"Title {i}" for i in range(len(all_texts))],
            'text': all_texts,
            'label': all_labels
        })
        kaggle_df.to_csv(os.path.join(self.temp_dir, "kaggle.csv"), index=False)
    
    def test_complete_pipeline_workflow(self):
        """Test the complete pipeline from data loading to prediction."""
        # Initialize detector
        detector = FakeNewsDetector(config=self.test_config)
        
        # Load datasets
        datasets = detector.load_datasets()
        self.assertIn('isot', datasets)
        self.assertIn('kaggle', datasets)
        
        # Check dataset structure
        isot_df = datasets['isot']
        self.assertIn('text_clean_traditional', isot_df.columns)
        self.assertIn('text_clean_transformer', isot_df.columns)
        self.assertIn('label_binary', isot_df.columns)
        
        # Train traditional models
        traditional_results = detector.train_traditional_models('isot')
        self.assertIn('bow', traditional_results)
        self.assertIn('tfidf', traditional_results)
        
        # Check that models are trained
        self.assertIn('bow', detector.traditional_models)
        self.assertIn('tfidf', detector.traditional_models)
        
        # Test predictions
        test_texts = [
            "Breaking scientific research reveals important findings",
            "SHOCKING secret that will AMAZE you - click here NOW!"
        ]
        
        # Test BOW predictions
        bow_results = detector.predict(test_texts, model_type='bow')
        self.assertIn('predictions', bow_results)
        self.assertIn('labels', bow_results)
        self.assertIn('probabilities', bow_results)
        self.assertIn('confidence', bow_results)
        
        # Test TF-IDF predictions
        tfidf_results = detector.predict(test_texts, model_type='tfidf')
        self.assertEqual(len(tfidf_results['predictions']), 2)
        self.assertEqual(len(tfidf_results['labels']), 2)
        
        # Create ensemble
        ensemble_model = detector.create_ensemble()
        self.assertIsNotNone(ensemble_model)
        
        # Test ensemble predictions
        ensemble_results = detector.predict(test_texts, model_type='ensemble')
        self.assertEqual(len(ensemble_results['predictions']), 2)
    
    def test_cross_domain_evaluation(self):
        """Test cross-domain evaluation functionality."""
        detector = FakeNewsDetector(config=self.test_config)
        datasets = detector.load_datasets()
        
        # Train models
        detector.train_traditional_models('isot')
        
        # Perform cross-domain evaluation
        cross_domain_results = detector.evaluate_cross_domain('isot', 'kaggle')
        
        # Check results structure
        self.assertIn('bow', cross_domain_results)
        self.assertIn('tfidf', cross_domain_results)
        
        # Check metrics in results
        bow_results = cross_domain_results['bow']
        self.assertIn('accuracy', bow_results)
        self.assertIn('f1_score', bow_results)
        self.assertIn('classification_report', bow_results)
        
        # Verify metrics are in reasonable range
        self.assertGreater(bow_results['accuracy'], 0.0)
        self.assertLessEqual(bow_results['accuracy'], 1.0)
        self.assertGreater(bow_results['f1_score'], 0.0)
        self.assertLessEqual(bow_results['f1_score'], 1.0)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        detector = FakeNewsDetector(config=self.test_config)
        datasets = detector.load_datasets()
        
        # Train models
        detector.train_traditional_models('isot')
        detector.create_ensemble()
        
        # Save models
        detector.save_models()
        
        # Check that model files exist
        models_dir = self.test_config.models_dir
        self.assertTrue(os.path.exists(os.path.join(models_dir, "bow_model.joblib")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "tfidf_model.joblib")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "ensemble_model.joblib")))
        
        # Create new detector and load models
        new_detector = FakeNewsDetector(config=self.test_config)
        new_detector.load_models()
        
        # Test that loaded models work
        test_texts = ["This is a test sentence"]
        
        original_pred = detector.predict(test_texts, model_type='bow')
        loaded_pred = new_detector.predict(test_texts, model_type='bow')
        
        np.testing.assert_array_equal(
            original_pred['predictions'],
            loaded_pred['predictions']
        )
    
    def test_configuration_management(self):
        """Test configuration saving and loading."""
        detector = FakeNewsDetector(config=self.test_config)
        
        # Save configuration
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        detector.save_config(config_path)
        
        # Check that config file exists
        self.assertTrue(os.path.exists(config_path))
        
        # Load configuration
        new_detector = FakeNewsDetector.from_config_file(config_path)
        
        # Check that configuration is loaded correctly
        self.assertEqual(new_detector.config.traditional_max_features, 50)
        self.assertEqual(new_detector.config.isot_fake_path, self.test_config.isot_fake_path)
    
    def test_error_handling_missing_datasets(self):
        """Test error handling when datasets are missing."""
        # Create config with non-existent files
        bad_config = PipelineConfig(
            isot_fake_path="nonexistent_fake.csv",
            isot_real_path="nonexistent_real.csv",
            kaggle_path="nonexistent_kaggle.csv"
        )
        
        detector = FakeNewsDetector(config=bad_config)
        datasets = detector.load_datasets()
        
        # Should return empty dict when files don't exist
        self.assertEqual(len(datasets), 0)
    
    def test_prediction_with_different_input_types(self):
        """Test predictions with different input types."""
        detector = FakeNewsDetector(config=self.test_config)
        detector.load_datasets()
        detector.train_traditional_models('isot')
        
        test_texts = ["Test sentence one", "Test sentence two"]
        
        # Test with list
        list_results = detector.predict(test_texts, model_type='bow')
        
        # Test with pandas Series
        series_results = detector.predict(pd.Series(test_texts), model_type='bow')
        
        # Results should be the same
        np.testing.assert_array_equal(
            list_results['predictions'],
            series_results['predictions']
        )
    
    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises appropriate error."""
        detector = FakeNewsDetector(config=self.test_config)
        detector.load_datasets()
        detector.train_traditional_models('isot')
        
        test_texts = ["Test sentence"]
        
        with self.assertRaises(ValueError):
            detector.predict(test_texts, model_type='nonexistent_model')
    
    def test_prediction_before_training_raises_error(self):
        """Test that prediction before training raises appropriate error."""
        detector = FakeNewsDetector(config=self.test_config)
        test_texts = ["Test sentence"]
        
        with self.assertRaises(ValueError):
            detector.predict(test_texts, model_type='bow')


class TestPipelineConfigValidation(unittest.TestCase):
    """Test pipeline configuration validation."""
    
    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = PipelineConfig()
        
        # Check default values
        self.assertEqual(config.traditional_max_features, 1000)
        self.assertEqual(config.traditional_min_df, 2)
        self.assertEqual(config.deberta_model_name, "microsoft/deberta-v3-base")
        self.assertEqual(config.ensemble_method, "weighted_voting")
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            traditional_max_features=500,
            deberta_batch_size=32,
            ensemble_method="stacking"
        )
        
        self.assertEqual(config.traditional_max_features, 500)
        self.assertEqual(config.deberta_batch_size, 32)
        self.assertEqual(config.ensemble_method, "stacking")


if __name__ == '__main__':
    unittest.main()