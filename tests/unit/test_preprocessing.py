"""
Unit tests for text preprocessing functionality.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.data.preprocessing import TextPreprocessor, DatasetLoader, standardize_labels


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        self.sample_texts = [
            "Breaking: Scientists at MIT have discovered AMAZING new technology! http://fake-news.com",
            "The Federal Reserve announced new policy measures today.",
            "SHOCKING: You won't believe this INCREDIBLE secret! @username",
            ""
        ]
    
    def test_clean_text_traditional(self):
        """Test traditional text cleaning."""
        text = "Breaking: Scientists at MIT have discovered AMAZING new technology! http://fake-news.com"
        cleaned = self.preprocessor.clean_text_traditional(text)
        
        # Should be lowercase
        self.assertEqual(cleaned, cleaned.lower())
        
        # Should not contain URLs
        self.assertNotIn("http", cleaned)
        
        # Should not contain punctuation
        self.assertNotIn(":", cleaned)
        self.assertNotIn("!", cleaned)
        
        # Should contain relevant words
        self.assertIn("scientist", cleaned)
        self.assertIn("technology", cleaned)
    
    def test_clean_text_transformer(self):
        """Test transformer text cleaning."""
        text = "Breaking: Scientists at MIT have discovered AMAZING new technology! http://fake-news.com @username"
        cleaned = self.preprocessor.clean_text_transformer(text)
        
        # Should be lowercase
        self.assertEqual(cleaned, cleaned.lower())
        
        # Should not contain URLs or mentions
        self.assertNotIn("http", cleaned)
        self.assertNotIn("@username", cleaned)
        
        # Should preserve basic punctuation and structure
        self.assertIn(":", cleaned)
        self.assertIn("!", cleaned)
        
        # Should contain relevant words
        self.assertIn("scientists", cleaned)
        self.assertIn("technology", cleaned)
    
    def test_empty_text_handling(self):
        """Test handling of empty or None text."""
        self.assertEqual(self.preprocessor.clean_text_traditional(""), "")
        self.assertEqual(self.preprocessor.clean_text_traditional(None), "")
        self.assertEqual(self.preprocessor.clean_text_transformer(""), "")
        self.assertEqual(self.preprocessor.clean_text_transformer(None), "")
    
    def test_preprocess_dataset_traditional(self):
        """Test dataset preprocessing with traditional method."""
        df = pd.DataFrame({'text': self.sample_texts[:3]})
        result = self.preprocessor.preprocess_dataset(df, preprocessing_type='traditional')
        
        self.assertIn('text_clean_traditional', result.columns)
        self.assertEqual(len(result), 3)
        
        # Check that all texts are processed
        self.assertTrue(all(isinstance(text, str) for text in result['text_clean_traditional']))
    
    def test_preprocess_dataset_transformer(self):
        """Test dataset preprocessing with transformer method."""
        df = pd.DataFrame({'text': self.sample_texts[:3]})
        result = self.preprocessor.preprocess_dataset(df, preprocessing_type='transformer')
        
        self.assertIn('text_clean_transformer', result.columns)
        self.assertEqual(len(result), 3)
        
        # Check that all texts are processed
        self.assertTrue(all(isinstance(text, str) for text in result['text_clean_transformer']))
    
    def test_invalid_preprocessing_type(self):
        """Test invalid preprocessing type raises error."""
        df = pd.DataFrame({'text': self.sample_texts[:3]})
        
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_dataset(df, preprocessing_type='invalid')


class TestStandardizeLabels(unittest.TestCase):
    """Test cases for label standardization."""
    
    def test_standardize_labels_default_mapping(self):
        """Test label standardization with default mapping."""
        df = pd.DataFrame({
            'label': ['FAKE', 'REAL', 'Fake', 'Real', 'fake', 'real']
        })
        
        result = standardize_labels(df)
        expected = [0, 1, 0, 1, 0, 1]
        
        self.assertIn('label_binary', result.columns)
        self.assertEqual(result['label_binary'].tolist(), expected)
    
    def test_standardize_labels_custom_mapping(self):
        """Test label standardization with custom mapping."""
        df = pd.DataFrame({
            'label': ['true', 'false', 'true', 'false']
        })
        
        custom_mapping = {'true': 1, 'false': 0}
        result = standardize_labels(df, label_mapping=custom_mapping)
        expected = [1, 0, 1, 0]
        
        self.assertEqual(result['label_binary'].tolist(), expected)
    
    def test_unmapped_labels_warning(self):
        """Test that unmapped labels generate warnings."""
        df = pd.DataFrame({
            'label': ['FAKE', 'REAL', 'UNKNOWN']
        })
        
        with patch('src.data.preprocessing.logger') as mock_logger:
            result = standardize_labels(df)
            mock_logger.warning.assert_called_once()
        
        # Should have one NaN value for unmapped label
        self.assertEqual(result['label_binary'].isna().sum(), 1)


class TestDatasetLoader(unittest.TestCase):
    """Test cases for DatasetLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DatasetLoader()
    
    @patch('pandas.read_csv')
    def test_load_isot_dataset_success(self, mock_read_csv):
        """Test successful ISOT dataset loading."""
        # Mock CSV data
        fake_data = pd.DataFrame({
            'title': ['Fake News 1', 'Fake News 2'],
            'text': ['Fake content 1', 'Fake content 2']
        })
        real_data = pd.DataFrame({
            'title': ['Real News 1', 'Real News 2'],
            'text': ['Real content 1', 'Real content 2']
        })
        
        mock_read_csv.side_effect = [fake_data, real_data]
        
        result = self.loader.load_isot_dataset('fake.csv', 'real.csv')
        
        # Check structure
        self.assertEqual(len(result), 4)
        self.assertIn('label', result.columns)
        self.assertIn('label_binary', result.columns)
        
        # Check labels
        fake_count = (result['label'] == 'Fake').sum()
        real_count = (result['label'] == 'Real').sum()
        self.assertEqual(fake_count, 2)
        self.assertEqual(real_count, 2)
    
    @patch('pandas.read_csv')
    def test_load_isot_dataset_file_error(self, mock_read_csv):
        """Test ISOT dataset loading with file error."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        with self.assertRaises(FileNotFoundError):
            self.loader.load_isot_dataset('nonexistent.csv', 'nonexistent.csv')
    
    @patch('pandas.read_csv')
    def test_load_kaggle_dataset_success(self, mock_read_csv):
        """Test successful Kaggle dataset loading."""
        # Mock CSV data
        kaggle_data = pd.DataFrame({
            'title': ['News 1', 'News 2'],
            'text': ['Content 1', 'Content 2'],
            'label': ['REAL', 'FAKE']
        })
        
        mock_read_csv.return_value = kaggle_data
        
        result = self.loader.load_kaggle_dataset('kaggle.csv')
        
        # Check structure
        self.assertEqual(len(result), 2)
        self.assertIn('label_binary', result.columns)
    
    @patch('pandas.read_csv')
    def test_load_kaggle_dataset_file_error(self, mock_read_csv):
        """Test Kaggle dataset loading with file error."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        with self.assertRaises(FileNotFoundError):
            self.loader.load_kaggle_dataset('nonexistent.csv')


if __name__ == '__main__':
    unittest.main()