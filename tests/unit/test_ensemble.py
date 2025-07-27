"""
Unit tests for ensemble methods.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path
import tempfile
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.models.ensemble import EnsembleClassifier


class MockModel:
    """Mock model for testing ensemble functionality."""
    
    def __init__(self, predictions=None, probabilities=None):
        self.predictions = predictions if predictions is not None else [1, 0, 1, 0, 1, 0, 1, 0]
        self.probabilities = probabilities if probabilities is not None else np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
        self.is_fitted = False
        self.classes_ = np.array([0, 1])  # Required for sklearn compatibility
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (required for sklearn compatibility)."""
        return {
            'predictions': self.predictions,
            'probabilities': self.probabilities
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (required for sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def fit(self, X, y):
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return np.array(self.predictions[:len(X)])
    
    def predict_proba(self, X):
        return self.probabilities[:len(X)]


class TestEnsembleClassifier(unittest.TestCase):
    """Test cases for EnsembleClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "This is a positive example",
            "This is a negative example", 
            "Another positive sample",
            "Another negative sample",
            "Third positive example",
            "Third negative example",
            "Fourth positive example",
            "Fourth negative example"
        ]
        self.sample_labels = [1, 0, 1, 0, 1, 0, 1, 0]
        
        # Create mock models
        self.model1 = MockModel(
            predictions=[1, 0, 1, 0, 1, 0, 1, 0],
            probabilities=np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
        )
        self.model2 = MockModel(
            predictions=[1, 1, 0, 0, 1, 1, 0, 0],
            probabilities=np.array([[0.3, 0.7], [0.4, 0.6], [0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2], [0.9, 0.1]])
        )
        
        self.models = [self.model1, self.model2]
        self.model_names = ['model1', 'model2']
    
    def test_initialization(self):
        """Test ensemble classifier initialization."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='weighted_voting'
        )
        
        self.assertEqual(ensemble.models, self.models)
        self.assertEqual(ensemble.model_names, self.model_names)
        self.assertEqual(ensemble.combination_method, 'weighted_voting')
        self.assertFalse(ensemble.is_fitted)
    
    def test_initialization_without_model_names(self):
        """Test initialization without providing model names."""
        ensemble = EnsembleClassifier(models=self.models)
        
        expected_names = ['Model_0', 'Model_1']
        self.assertEqual(ensemble.model_names, expected_names)
    
    def test_weighted_voting_fit(self):
        """Test weighted voting ensemble fitting."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='weighted_voting'
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        
        self.assertTrue(ensemble.is_fitted)
        self.assertIsNotNone(ensemble.weights)
        self.assertEqual(len(ensemble.weights), 2)
        
        # Weights should sum to 1
        self.assertAlmostEqual(sum(ensemble.weights), 1.0, places=5)
    
    def test_weighted_voting_predict(self):
        """Test weighted voting prediction."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='weighted_voting'
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        predictions = ensemble.predict(self.sample_texts[:2])
        
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_weighted_voting_predict_proba(self):
        """Test weighted voting probability prediction."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='weighted_voting'
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        probabilities = ensemble.predict_proba(self.sample_texts[:2])
        
        self.assertEqual(probabilities.shape, (2, 2))
        # Probabilities should sum to 1 for each sample
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
    
    def test_stacking_fit(self):
        """Test stacking ensemble fitting."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='stacking'
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        
        self.assertTrue(ensemble.is_fitted)
        self.assertIsNotNone(ensemble.meta_classifier)
    
    def test_stacking_predict(self):
        """Test stacking prediction."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='stacking'
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        predictions = ensemble.predict(self.sample_texts[:2])
        
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_confidence_based_fit(self):
        """Test confidence-based ensemble fitting."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='confidence_based'
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        
        self.assertTrue(ensemble.is_fitted)
    
    def test_confidence_based_predict(self):
        """Test confidence-based prediction."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names,
            combination_method='confidence_based'
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        predictions = ensemble.predict(self.sample_texts[:2])
        
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_invalid_combination_method(self):
        """Test that invalid combination method raises error."""
        ensemble = EnsembleClassifier(
            models=self.models,
            combination_method='invalid_method'
        )
        
        with self.assertRaises(ValueError):
            ensemble.fit(self.sample_texts, self.sample_labels)
    
    def test_predict_before_fit_raises_error(self):
        """Test that prediction before fitting raises error."""
        ensemble = EnsembleClassifier(models=self.models)
        
        with self.assertRaises(ValueError):
            ensemble.predict(self.sample_texts)
        
        with self.assertRaises(ValueError):
            ensemble.predict_proba(self.sample_texts)
    
    def test_custom_weights(self):
        """Test ensemble with custom weights."""
        custom_weights = [0.7, 0.3]
        ensemble = EnsembleClassifier(
            models=self.models,
            combination_method='weighted_voting',
            weights=custom_weights
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        
        self.assertEqual(ensemble.weights, custom_weights)
    
    def test_get_model_contributions(self):
        """Test getting individual model contributions."""
        ensemble = EnsembleClassifier(
            models=self.models,
            model_names=self.model_names
        )
        
        ensemble.fit(self.sample_texts, self.sample_labels)
        contributions = ensemble.get_model_contributions(self.sample_texts[:2])
        
        self.assertIn('model1', contributions)
        self.assertIn('model2', contributions)
        
        # Each model should contribute probabilities for 2 samples
        self.assertEqual(contributions['model1'].shape, (2, 2))
        self.assertEqual(contributions['model2'].shape, (2, 2))
    
    def test_save_and_load_ensemble(self):
        """Test ensemble saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ensemble_path = os.path.join(temp_dir, 'test_ensemble.joblib')
            
            # Train and save ensemble
            ensemble = EnsembleClassifier(
                models=self.models,
                model_names=self.model_names,
                combination_method='weighted_voting'
            )
            ensemble.fit(self.sample_texts, self.sample_labels)
            ensemble.save_ensemble(ensemble_path)
            
            # Load ensemble
            loaded_ensemble = EnsembleClassifier.load_ensemble(ensemble_path)
            
            # Test that loaded ensemble works
            self.assertTrue(loaded_ensemble.is_fitted)
            self.assertEqual(loaded_ensemble.model_names, self.model_names)
            self.assertEqual(loaded_ensemble.combination_method, 'weighted_voting')
            
            # Compare predictions
            original_pred = ensemble.predict(self.sample_texts[:2])
            loaded_pred = loaded_ensemble.predict(self.sample_texts[:2])
            
            np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_input_validation(self):
        """Test input validation for different input types."""
        ensemble = EnsembleClassifier(models=self.models)
        ensemble.fit(self.sample_texts, self.sample_labels)
        
        # Test with pandas Series
        series_input = pd.Series(self.sample_texts[:2])
        list_pred = ensemble.predict(self.sample_texts[:2])
        series_pred = ensemble.predict(series_input)
        
        np.testing.assert_array_equal(list_pred, series_pred)
        
        # Test invalid input
        with self.assertRaises(ValueError):
            ensemble.predict("not a list or series")


class TestEnsembleEdgeCases(unittest.TestCase):
    """Test edge cases for ensemble classifier."""
    
    def test_single_model_ensemble(self):
        """Test ensemble with single model."""
        model = MockModel()
        ensemble = EnsembleClassifier(models=[model])
        
        # This should still work
        ensemble.fit(['text1', 'text2'], [1, 0])
        predictions = ensemble.predict(['text1'])
        
        self.assertEqual(len(predictions), 1)
    
    def test_models_without_predict_proba(self):
        """Test ensemble with models that don't have predict_proba."""
        class ModelWithoutProba:
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return np.array([1] * len(X))
        
        model = ModelWithoutProba()
        ensemble = EnsembleClassifier(models=[model])
        
        ensemble.fit(['text1', 'text2'], [1, 0])
        predictions = ensemble.predict(['text1'])
        probabilities = ensemble.predict_proba(['text1'])
        
        self.assertEqual(len(predictions), 1)
        self.assertEqual(probabilities.shape, (1, 2))
    
    def test_empty_models_list(self):
        """Test ensemble with empty models list."""
        ensemble = EnsembleClassifier(models=[])
        
        # Empty models should work but return default behavior
        ensemble.fit(['text1', 'text2'], [1, 0])
        predictions = ensemble.predict(['text1'])
        
        # Should return something (even if it's just a default)
        self.assertIsInstance(predictions, (int, np.integer, np.ndarray))
        if isinstance(predictions, np.ndarray):
            self.assertEqual(len(predictions), 1)
        else:
            # Single prediction returned
            self.assertIn(predictions, [0, 1])


if __name__ == '__main__':
    unittest.main()