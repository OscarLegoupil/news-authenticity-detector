"""
Traditional ML models using Bag-of-Words and TF-IDF representations.
Extracted from notebook implementations and enhanced for production use.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalMLClassifier(BaseEstimator, ClassifierMixin):
    """Base class for traditional ML classifiers with feature extraction."""
    
    def __init__(self, 
                 vectorizer_type: str = 'tfidf',
                 max_features: int = 1000,
                 min_df: int = 2,
                 classifier_params: Optional[Dict] = None):
        """
        Initialize traditional ML classifier.
        
        Args:
            vectorizer_type: 'bow' or 'tfidf'
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for terms
            classifier_params: Parameters for LogisticRegression
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.min_df = min_df
        self.classifier_params = classifier_params or {}
        
        # Initialize components
        self._setup_vectorizer()
        self._setup_classifier()
        
        # State tracking
        self.is_fitted = False
        self.feature_names_ = None
        
    def _setup_vectorizer(self):
        """Initialize the appropriate vectorizer."""
        common_params = {
            'max_features': self.max_features,
            'min_df': self.min_df
        }
        
        if self.vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer(**common_params)
        elif self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(**common_params)
        else:
            raise ValueError("vectorizer_type must be 'bow' or 'tfidf'")
    
    def _setup_classifier(self):
        """Initialize the classifier with default parameters."""
        default_params = {
            'max_iter': 500,
            'solver': 'liblinear',
            'random_state': 42
        }
        default_params.update(self.classifier_params)
        self.classifier = LogisticRegression(**default_params)
    
    def fit(self, X: Union[List[str], pd.Series], y: Union[List, np.ndarray, pd.Series]):
        """
        Fit the vectorizer and classifier.
        
        Args:
            X: Text documents
            y: Labels
        """
        # Convert inputs to appropriate format
        X = self._validate_text_input(X)
        y = np.array(y)
        
        # Fit vectorizer and transform text
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Fit classifier
        self.classifier.fit(X_vectorized, y)
        
        # Store feature names
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        
        logger.info(f"Fitted {self.vectorizer_type} classifier with {X_vectorized.shape[1]} features")
        return self
    
    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Predict labels for new documents.
        
        Args:
            X: Text documents
            
        Returns:
            Predicted labels
        """
        self._check_fitted()
        X = self._validate_text_input(X)
        X_vectorized = self.vectorizer.transform(X)
        return self.classifier.predict(X_vectorized)
    
    def predict_proba(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Predict class probabilities for new documents.
        
        Args:
            X: Text documents
            
        Returns:
            Class probabilities
        """
        self._check_fitted()
        X = self._validate_text_input(X)
        X_vectorized = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vectorized)
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, np.ndarray]:
        """
        Get most important features for each class.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            Dictionary with 'fake' and 'real' top features
        """
        self._check_fitted()
        
        if not hasattr(self.classifier, 'coef_'):
            raise ValueError("Feature importance only available for linear models")
        
        coefficients = self.classifier.coef_[0]
        feature_names = np.array(self.feature_names_)
        
        # Top features for fake (negative coefficients)
        fake_indices = np.argsort(coefficients)[:top_k]
        fake_features = feature_names[fake_indices]
        
        # Top features for real (positive coefficients)  
        real_indices = np.argsort(coefficients)[-top_k:]
        real_features = feature_names[real_indices]
        
        return {
            'fake': fake_features,
            'real': real_features
        }
    
    def cross_validate(self, 
                      X: Union[List[str], pd.Series], 
                      y: Union[List, np.ndarray, pd.Series],
                      cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation evaluation with proper data isolation.
        
        Args:
            X: Text documents
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with mean and std of accuracy scores
        """
        X = self._validate_text_input(X)
        y = np.array(y)
        
        # Perform stratified cross-validation with proper vectorizer fitting
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            # Split data
            X_train_fold = [X[i] for i in train_idx]
            X_val_fold = [X[i] for i in val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Create fresh vectorizer for this fold
            if self.vectorizer_type == 'bow':
                fold_vectorizer = CountVectorizer(
                    max_features=self.vectorizer.max_features,
                    min_df=self.vectorizer.min_df,
                    stop_words='english'
                )
            else:  # tfidf
                fold_vectorizer = TfidfVectorizer(
                    max_features=self.vectorizer.max_features,
                    min_df=self.vectorizer.min_df,
                    stop_words='english'
                )
            
            # Fit vectorizer ONLY on training fold
            X_train_vec = fold_vectorizer.fit_transform(X_train_fold)
            X_val_vec = fold_vectorizer.transform(X_val_fold)
            
            # Train classifier on this fold
            fold_classifier = LogisticRegression(random_state=42, max_iter=1000)
            fold_classifier.fit(X_train_vec, y_train_fold)
            
            # Evaluate on validation fold
            fold_score = fold_classifier.score(X_val_vec, y_val_fold)
            scores.append(fold_score)
        
        scores = np.array(scores)
        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }
        
        logger.info(f"Cross-validation results (NO LEAKAGE): {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        return results
    
    def evaluate_cross_domain(self, 
                             X_train: Union[List[str], pd.Series],
                             y_train: Union[List, np.ndarray, pd.Series],
                             X_test: Union[List[str], pd.Series],
                             y_test: Union[List, np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate cross-domain generalization.
        
        Args:
            X_train: Training text documents
            y_train: Training labels
            X_test: Test text documents from different domain
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Fit on training domain
        self.fit(X_train, y_train)
        
        # Predict on test domain
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        logger.info(f"Cross-domain evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results
    
    def save_model(self, filepath: str):
        """Save the fitted model to disk."""
        self._check_fitted()
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'feature_names': self.feature_names_,
            'config': {
                'vectorizer_type': self.vectorizer_type,
                'max_features': self.max_features,
                'min_df': self.min_df,
                'classifier_params': self.classifier_params
            }
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a fitted model from disk."""
        model_data = joblib.load(filepath)
        
        # Create instance with saved config
        instance = cls(**model_data['config'])
        
        # Restore fitted components
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier'] 
        instance.feature_names_ = model_data['feature_names']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def _validate_text_input(self, X: Union[List[str], pd.Series]) -> List[str]:
        """Validate and convert text input to list of strings."""
        if isinstance(X, pd.Series):
            X = X.astype(str).tolist()
        elif isinstance(X, list):
            X = [str(text) for text in X]
        else:
            raise ValueError("X must be a list of strings or pandas Series")
        return X
    
    def _check_fitted(self):
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

class BOWClassifier(TraditionalMLClassifier):
    """Bag-of-Words classifier with LogisticRegression."""
    
    def __init__(self, **kwargs):
        kwargs['vectorizer_type'] = 'bow'
        super().__init__(**kwargs)

class TFIDFClassifier(TraditionalMLClassifier):
    """TF-IDF classifier with LogisticRegression."""
    
    def __init__(self, **kwargs):
        kwargs['vectorizer_type'] = 'tfidf'
        super().__init__(**kwargs)