"""
Text preprocessing utilities for fake news detection.
Extracted from notebook implementations and enhanced for production use.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing pipeline with configurable options."""
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 apply_lemmatization: bool = True,
                 min_length: int = 3):
        """
        Initialize preprocessor with configuration options.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            apply_lemmatization: Whether to apply lemmatization
            min_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.apply_lemmatization = apply_lemmatization
        self.min_length = min_length
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize components
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        if self.apply_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
    
    def _download_nltk_data(self):
        """Download required NLTK data silently."""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    def clean_text_traditional(self, text: str) -> str:
        """
        Preprocessing for traditional ML models (BoW, TF-IDF).
        Aggressive cleaning including stopword removal and lemmatization.
        Removes source attribution patterns to prevent data leakage.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove source attribution patterns that cause leakage
        text = self._remove_source_patterns(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and Twitter handles
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        
        # Remove punctuation, digits, and other non-alphabetic characters
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize by splitting on whitespace
        tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords]
        
        # Filter by minimum length
        tokens = [word for word in tokens if len(word) >= self.min_length]
        
        # Lemmatize each token
        if self.apply_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def _remove_source_patterns(self, text: str) -> str:
        """
        Remove source attribution patterns that cause data leakage.
        
        Args:
            text: Raw text
            
        Returns:
            Text with source patterns removed
        """
        # Patterns that strongly indicate source and cause leakage
        source_patterns = [
            # News agency attribution
            r'\b(?:reuters|associated press|ap)\b',
            r'\([^)]*reuters[^)]*\)',
            r'\([^)]*ap[^)]*\)',
            
            # Location-based patterns that correlate with source
            r'\b(?:washington|london|new york|paris)\s*\([^)]*(?:reuters|ap)[^)]*\)',
            r'^\s*(?:washington|london|new york|paris)\s*[-–]',
            
            # Common news agency intro patterns
            r'^\s*\([^)]*\)\s*[-–]',
            r'^\s*[A-Z][a-z]+\s*\([^)]*\)\s*[-–]',
            
            # Copyright and attribution lines
            r'©\s*\d{4}.*(?:reuters|associated press|ap)',
            r'copyright.*(?:reuters|associated press|ap)',
            
            # Byline patterns
            r'by\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?',
            
            # Editor notes and source identifiers
            r'\(editing by[^)]*\)',
            r'\(reporting by[^)]*\)',
            r'\(additional reporting[^)]*\)',
        ]
        
        # Apply all source removal patterns
        for pattern in source_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_text_transformer(self, text: str) -> str:
        """
        Lighter preprocessing for transformer models.
        Preserves context while removing noise.
        Removes source attribution patterns to prevent data leakage.
        
        Args:
            text: Input text to clean
            
        Returns:
            Lightly cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove source attribution patterns that cause leakage
        text = self._remove_source_patterns(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and Twitter handles
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_dataset(self, 
                          df: pd.DataFrame, 
                          text_column: str = 'text',
                          preprocessing_type: str = 'traditional') -> pd.DataFrame:
        """
        Apply preprocessing to an entire dataset.
        
        Args:
            df: Input dataframe
            text_column: Name of text column to preprocess
            preprocessing_type: 'traditional' or 'transformer'
            
        Returns:
            Dataframe with added cleaned text column
        """
        df = df.copy()
        
        if preprocessing_type == 'traditional':
            clean_column = f"{text_column}_clean_traditional"
            df[clean_column] = df[text_column].astype(str).apply(self.clean_text_traditional)
        elif preprocessing_type == 'transformer':
            clean_column = f"{text_column}_clean_transformer"
            df[clean_column] = df[text_column].astype(str).apply(self.clean_text_transformer)
        else:
            raise ValueError("preprocessing_type must be 'traditional' or 'transformer'")
        
        logger.info(f"Preprocessed {len(df)} documents with {preprocessing_type} preprocessing")
        return df

def standardize_labels(df: pd.DataFrame, 
                      label_column: str = 'label',
                      label_mapping: Optional[dict] = None) -> pd.DataFrame:
    """
    Standardize label formats across different datasets.
    
    Args:
        df: Input dataframe
        label_column: Name of label column
        label_mapping: Custom mapping dict, defaults to common mappings
        
    Returns:
        Dataframe with standardized binary labels (0=Fake, 1=Real)
    """
    df = df.copy()
    
    if label_mapping is None:
        # Common label mappings
        label_mapping = {
            'FAKE': 0, 'REAL': 1,
            'Fake': 0, 'Real': 1, 
            'fake': 0, 'real': 1,
            0: 0, 1: 1
        }
    
    # Apply mapping
    df[f'{label_column}_binary'] = df[label_column].map(label_mapping)
    
    # Check for unmapped values
    unmapped = df[f'{label_column}_binary'].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} labels could not be mapped: {df[label_column].value_counts()}")
    
    return df

class DatasetLoader:
    """Unified dataset loading interface for different fake news datasets."""
    
    @staticmethod
    def load_isot_dataset(fake_path: str, real_path: str) -> pd.DataFrame:
        """
        Load ISOT fake news dataset from separate CSV files.
        
        Args:
            fake_path: Path to fake news CSV
            real_path: Path to real news CSV
            
        Returns:
            Combined dataframe with standardized format
        """
        try:
            df_fake = pd.read_csv(fake_path)
            df_real = pd.read_csv(real_path)
            
            # Add labels
            df_fake['label'] = 'Fake'
            df_real['label'] = 'Real'
            
            # Combine datasets
            df_combined = pd.concat([df_fake, df_real], ignore_index=True)
            
            # Standardize labels
            df_combined = standardize_labels(df_combined)
            
            logger.info(f"Loaded ISOT dataset: {len(df_fake)} fake, {len(df_real)} real articles")
            return df_combined
            
        except Exception as e:
            logger.error(f"Error loading ISOT dataset: {e}")
            raise
    
    @staticmethod
    def load_kaggle_dataset(path: str) -> pd.DataFrame:
        """
        Load Kaggle fake news dataset.
        
        Args:
            path: Path to Kaggle CSV file
            
        Returns:
            Dataframe with standardized format
        """
        try:
            df = pd.read_csv(path)
            
            # Standardize labels
            df = standardize_labels(df)
            
            logger.info(f"Loaded Kaggle dataset: {len(df)} articles")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {e}")
            raise