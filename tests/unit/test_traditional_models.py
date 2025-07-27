"""
Unit tests for traditional ML models.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import tempfile
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.models.traditional.bow_tfidf import BOWClassifier, TFIDFClassifier, TraditionalMLClassifier


class TestTraditionalMLClassifier(unittest.TestCase):
    """Test cases for TraditionalMLClassifier base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "This is a positive example with good words",
            "This is a negative example with bad words",
            "Another positive sample with excellent content",
            "Another negative sample with terrible content"
        ]
        self.sample_labels = [1, 0, 1, 0]
    
    def test_initialization_bow(self):
        """Test BOW classifier initialization."""
        classifier = BOWClassifier(max_features=100, min_df=1)
        
        self.assertEqual(classifier.vectorizer_type, 'bow')
        self.assertEqual(classifier.max_features, 100)
        self.assertEqual(classifier.min_df, 1)
        self.assertFalse(classifier.is_fitted)
    
    def test_initialization_tfidf(self):
        """Test TF-IDF classifier initialization."""
        classifier = TFIDFClassifier(max_features=200, min_df=2)
        
        self.assertEqual(classifier.vectorizer_type, 'tfidf')
        self.assertEqual(classifier.max_features, 200)
        self.assertEqual(classifier.min_df, 2)
        self.assertFalse(classifier.is_fitted)
    
    def test_fit_and_predict_bow(self):
        """Test BOW classifier fitting and prediction."""
        classifier = BOWClassifier(max_features=100, min_df=1)
        
        # Fit the model
        classifier.fit(self.sample_texts, self.sample_labels)
        
        # Check if fitted
        self.assertTrue(classifier.is_fitted)
        self.assertIsNotNone(classifier.feature_names_)
        
        # Make predictions
        predictions = classifier.predict(self.sample_texts[:2])
        probabilities = classifier.predict_proba(self.sample_texts[:2])
        
        # Check predictions
        self.assertEqual(len(predictions), 2)
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_fit_and_predict_tfidf(self):
        """Test TF-IDF classifier fitting and prediction."""
        classifier = TFIDFClassifier(max_features=100, min_df=1)
        
        # Fit the model
        classifier.fit(self.sample_texts, self.sample_labels)
        
        # Check if fitted
        self.assertTrue(classifier.is_fitted)
        
        # Make predictions
        predictions = classifier.predict(self.sample_texts[:2])
        probabilities = classifier.predict_proba(self.sample_texts[:2])
        
        # Check predictions
        self.assertEqual(len(predictions), 2)
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predict_before_fit_raises_error(self):
        """Test that prediction before fitting raises error."""
        classifier = BOWClassifier()
        
        with self.assertRaises(ValueError):
            classifier.predict(self.sample_texts)
        
        with self.assertRaises(ValueError):
            classifier.predict_proba(self.sample_texts)
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        classifier = TFIDFClassifier(max_features=50, min_df=1)
        
        results = classifier.cross_validate(self.sample_texts, self.sample_labels, cv=2)
        
        # Check results structure
        self.assertIn('mean_accuracy', results)
        self.assertIn('std_accuracy', results)
        self.assertIn('scores', results)
        
        # Check results values
        self.assertIsInstance(results['mean_accuracy'], float)
        self.assertIsInstance(results['std_accuracy'], float)
        self.assertEqual(len(results['scores']), 2)
    
    def test_evaluate_cross_domain(self):
        """Test cross-domain evaluation."""
        classifier = BOWClassifier(max_features=50, min_df=1)
        
        # Create slightly different domain data
        train_texts = self.sample_texts
        train_labels = self.sample_labels
        test_texts = [
            "This is good quality content",
            "This is bad quality content"
        ]
        test_labels = [1, 0]
        
        results = classifier.evaluate_cross_domain(
            train_texts, train_labels, test_texts, test_labels
        )
        
        # Check results structure
        self.assertIn('accuracy', results)
        self.assertIn('f1_score', results)
        self.assertIn('classification_report', results)
        
        # Check results values
        self.assertIsInstance(results['accuracy'], float)
        self.assertIsInstance(results['f1_score'], float)
        self.assertIsInstance(results['classification_report'], str)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        classifier = TFIDFClassifier(max_features=50, min_df=1)
        classifier.fit(self.sample_texts, self.sample_labels)
        
        importance = classifier.get_feature_importance(top_k=5)
        
        # Check structure
        self.assertIn('fake', importance)
        self.assertIn('real', importance)
        
        # Check that we get the requested number of features
        self.assertEqual(len(importance['fake']), 5)
        self.assertEqual(len(importance['real']), 5)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.joblib')
            
            # Train and save model
            classifier = BOWClassifier(max_features=50, min_df=1)
            classifier.fit(self.sample_texts, self.sample_labels)
            classifier.save_model(model_path)
            
            # Load model
            loaded_classifier = BOWClassifier.load_model(model_path)
            
            # Test that loaded model works
            self.assertTrue(loaded_classifier.is_fitted)
            
            # Compare predictions
            original_pred = classifier.predict(self.sample_texts[:2])
            loaded_pred = loaded_classifier.predict(self.sample_texts[:2])
            
            np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_input_validation(self):
        """Test input validation for different input types."""
        classifier = BOWClassifier(max_features=50, min_df=1)
        
        # Test with pandas Series
        series_input = pd.Series(self.sample_texts)
        classifier.fit(series_input, self.sample_labels)
        
        # Test prediction with different input types
        list_pred = classifier.predict(self.sample_texts[:2])
        series_pred = classifier.predict(pd.Series(self.sample_texts[:2]))
        
        np.testing.assert_array_equal(list_pred, series_pred)
        
        # Test invalid input
        with self.assertRaises(ValueError):
            classifier.predict("not a list or series")
    
    def test_invalid_vectorizer_type(self):
        """Test that invalid vectorizer type raises error."""
        with self.assertRaises(ValueError):
            TraditionalMLClassifier(vectorizer_type='invalid')
    
    def test_feature_importance_without_linear_model(self):
        """Test feature importance with non-linear model raises error."""
        from sklearn.ensemble import RandomForestClassifier
        
        classifier = TraditionalMLClassifier(
            vectorizer_type='tfidf',
            classifier_params={'max_iter': 100}
        )
        
        # Replace with non-linear classifier
        classifier.classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier.fit(self.sample_texts, self.sample_labels)
        
        # This should work for RandomForest as it has feature_importances_
        # But our implementation specifically checks for coef_, so it should raise an error
        with self.assertRaises(ValueError):
            classifier.get_feature_importance()


class TestBOWClassifier(unittest.TestCase):
    """Test cases specific to BOW classifier."""
    
    def test_bow_specific_functionality(self):
        """Test BOW-specific functionality."""
        classifier = BOWClassifier(max_features=100, min_df=1)
        
        # Check that it's using the correct vectorizer
        self.assertEqual(classifier.vectorizer_type, 'bow')
        
        # The vectorizer should be CountVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        self.assertIsInstance(classifier.vectorizer, CountVectorizer)


class TestTFIDFClassifier(unittest.TestCase):
    """Test cases specific to TF-IDF classifier."""
    
    def test_tfidf_specific_functionality(self):
        """Test TF-IDF-specific functionality."""
        classifier = TFIDFClassifier(max_features=100, min_df=1)
        
        # Check that it's using the correct vectorizer
        self.assertEqual(classifier.vectorizer_type, 'tfidf')
        
        # The vectorizer should be TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.assertIsInstance(classifier.vectorizer, TfidfVectorizer)


if __name__ == '__main__':
    unittest.main()