"""
Ensemble methods combining traditional ML and transformer models.
Implements weighted voting, stacking, and confidence-based combination strategies.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Base ensemble classifier for combining multiple models."""
    
    def __init__(self, 
                 models: List[Any],
                 model_names: Optional[List[str]] = None,
                 combination_method: str = 'weighted_voting',
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of fitted models
            model_names: Names for models (for logging)
            combination_method: 'weighted_voting', 'stacking', or 'confidence_based'
            weights: Weights for weighted voting (auto-computed if None)
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.combination_method = combination_method
        self.weights = weights
        
        self.meta_classifier = None
        self.is_fitted = False
        
    def fit(self, 
            X: Union[List[str], pd.Series], 
            y: Union[List, np.ndarray, pd.Series],
            calibrate: bool = True):
        """
        Fit the ensemble classifier.
        
        Args:
            X: Training text data
            y: Training labels
            calibrate: Whether to apply probability calibration
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        if self.combination_method == 'weighted_voting':
            self._fit_weighted_voting(X, y, calibrate)
        elif self.combination_method == 'stacking':
            self._fit_stacking(X, y, calibrate)
        elif self.combination_method == 'confidence_based':
            self._fit_confidence_based(X, y, calibrate)
        else:
            raise ValueError("combination_method must be 'weighted_voting', 'stacking', or 'confidence_based'")
        
        self.is_fitted = True
        logger.info(f"Ensemble fitted with {self.combination_method} method")
        return self
    
    def _fit_weighted_voting(self, X: List[str], y: np.ndarray, calibrate: bool):
        """Fit weighted voting ensemble."""
        if self.weights is None:
            # Compute weights based on cross-validation performance
            self.weights = self._compute_model_weights(X, y)
        
        # Store models (skip calibration for now to avoid sklearn compatibility issues)
        self.calibrated_models = self.models
    
    def _fit_stacking(self, X: List[str], y: np.ndarray, calibrate: bool):
        """Fit stacking ensemble with meta-classifier."""
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Train meta-classifier
        self.meta_classifier = LogisticRegression(random_state=42)
        self.meta_classifier.fit(meta_features, y)
        
        # Calibrate if requested
        if calibrate:
            self.meta_classifier = CalibratedClassifierCV(self.meta_classifier, cv=3)
            self.meta_classifier.fit(meta_features, y)
    
    def _fit_confidence_based(self, X: List[str], y: np.ndarray, calibrate: bool):
        """Fit confidence-based ensemble."""
        # This method uses model confidence to weight predictions
        # Implementation focuses on prediction time weighting
        self._fit_weighted_voting(X, y, calibrate)
        
    def _compute_model_weights(self, X: List[str], y: np.ndarray) -> List[float]:
        """Compute model weights based on cross-validation performance."""
        weights = []
        
        for i, model in enumerate(self.models):
            try:
                # Get cross-validated predictions
                if hasattr(model, 'predict'):
                    cv_predictions = cross_val_predict(model, X, y, cv=3)
                    score = f1_score(y, cv_predictions, average='weighted')
                else:
                    # For models that need special handling
                    score = 0.5  # Default weight
                
                weights.append(score)
                logger.info(f"{self.model_names[i]} CV F1: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not compute weight for {self.model_names[i]}: {e}")
                weights.append(0.5)  # Default weight
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        logger.info(f"Model weights: {dict(zip(self.model_names, weights))}")
        return weights
    
    def _generate_meta_features(self, X: List[str], y: np.ndarray) -> np.ndarray:
        """Generate meta-features for stacking."""
        meta_features = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                # Use probabilities as meta-features
                cv_proba = cross_val_predict(model, X, y, cv=3, method='predict_proba')
                meta_features.append(cv_proba)
            else:
                # Use predictions as meta-features (one-hot encoded)
                cv_pred = cross_val_predict(model, X, y, cv=3)
                one_hot = np.zeros((len(cv_pred), 2))
                one_hot[np.arange(len(cv_pred)), cv_pred] = 1
                meta_features.append(one_hot)
        
        return np.hstack(meta_features)
    
    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict using ensemble method."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = self._validate_input(X)
        
        if self.combination_method == 'weighted_voting':
            return self._predict_weighted_voting(X)
        elif self.combination_method == 'stacking':
            return self._predict_stacking(X)
        elif self.combination_method == 'confidence_based':
            return self._predict_confidence_based(X)
    
    def predict_proba(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict probabilities using ensemble method."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = self._validate_input(X)
        
        if self.combination_method == 'weighted_voting':
            return self._predict_proba_weighted_voting(X)
        elif self.combination_method == 'stacking':
            return self._predict_proba_stacking(X)
        elif self.combination_method == 'confidence_based':
            return self._predict_proba_confidence_based(X)
    
    def _predict_weighted_voting(self, X: List[str]) -> np.ndarray:
        """Predict using weighted voting."""
        predictions = []
        
        for model, weight in zip(self.calibrated_models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        # Combine weighted predictions
        combined = np.sum(predictions, axis=0)
        return (combined > 0.5).astype(int)
    
    def _predict_proba_weighted_voting(self, X: List[str]) -> np.ndarray:
        """Predict probabilities using weighted voting."""
        probabilities = []
        
        for model, weight in zip(self.calibrated_models, self.weights):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # Convert predictions to probabilities
                pred = model.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[np.arange(len(pred)), pred] = 1.0
            
            probabilities.append(proba * weight)
        
        # Combine weighted probabilities
        combined_proba = np.sum(probabilities, axis=0)
        return combined_proba
    
    def _predict_stacking(self, X: List[str]) -> np.ndarray:
        """Predict using stacking."""
        # Generate meta-features
        meta_features = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                pred = model.predict(X)
                one_hot = np.zeros((len(pred), 2))
                one_hot[np.arange(len(pred)), pred] = 1
                meta_features.append(one_hot)
        
        meta_features = np.hstack(meta_features)
        
        # Predict using meta-classifier
        return self.meta_classifier.predict(meta_features)
    
    def _predict_proba_stacking(self, X: List[str]) -> np.ndarray:
        """Predict probabilities using stacking."""
        # Generate meta-features
        meta_features = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                pred = model.predict(X)
                one_hot = np.zeros((len(pred), 2))
                one_hot[np.arange(len(pred)), pred] = 1
                meta_features.append(one_hot)
        
        meta_features = np.hstack(meta_features)
        
        # Predict using meta-classifier
        return self.meta_classifier.predict_proba(meta_features)
    
    def _predict_confidence_based(self, X: List[str]) -> np.ndarray:
        """Predict using confidence-based weighting."""
        predictions = []
        confidences = []
        
        for model in self.calibrated_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                pred = np.argmax(proba, axis=1)
                conf = np.max(proba, axis=1)  # Confidence = max probability
            else:
                pred = model.predict(X)
                conf = np.ones(len(pred)) * 0.5  # Default confidence
            
            predictions.append(pred)
            confidences.append(conf)
        
        # Weight predictions by confidence
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Normalize confidences for each sample
        normalized_conf = confidences / np.sum(confidences, axis=0)
        
        # Weighted average
        weighted_pred = np.sum(predictions * normalized_conf, axis=0)
        return (weighted_pred > 0.5).astype(int)
    
    def _predict_proba_confidence_based(self, X: List[str]) -> np.ndarray:
        """Predict probabilities using confidence-based weighting."""
        probabilities = []
        confidences = []
        
        for model in self.calibrated_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                conf = np.max(proba, axis=1)
            else:
                pred = model.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[np.arange(len(pred)), pred] = 1.0
                conf = np.ones(len(pred)) * 0.5
            
            probabilities.append(proba)
            confidences.append(conf)
        
        # Weight probabilities by confidence
        probabilities = np.array(probabilities)
        confidences = np.array(confidences)
        
        # Normalize confidences for each sample
        normalized_conf = confidences / np.sum(confidences, axis=0)
        
        # Weighted average (broadcasting)
        weighted_proba = np.sum(
            probabilities * normalized_conf[:, :, np.newaxis], 
            axis=0
        )
        
        return weighted_proba
    
    def get_model_contributions(self, X: Union[List[str], pd.Series]) -> Dict[str, np.ndarray]:
        """Get individual model predictions for analysis."""
        X = self._validate_input(X)
        contributions = {}
        
        for model, name in zip(self.models, self.model_names):
            if hasattr(model, 'predict_proba'):
                contributions[name] = model.predict_proba(X)
            else:
                pred = model.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[np.arange(len(pred)), pred] = 1.0
                contributions[name] = proba
        
        return contributions
    
    def save_ensemble(self, filepath: str):
        """Save the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving")
        
        ensemble_data = {
            'models': self.models,
            'model_names': self.model_names,
            'combination_method': self.combination_method,
            'weights': self.weights,
            'meta_classifier': self.meta_classifier,
            'calibrated_models': getattr(self, 'calibrated_models', None)
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str):
        """Load a saved ensemble model."""
        ensemble_data = joblib.load(filepath)
        
        instance = cls(
            models=ensemble_data['models'],
            model_names=ensemble_data['model_names'],
            combination_method=ensemble_data['combination_method'],
            weights=ensemble_data['weights']
        )
        
        instance.meta_classifier = ensemble_data['meta_classifier']
        instance.calibrated_models = ensemble_data['calibrated_models']
        instance.is_fitted = True
        
        logger.info(f"Ensemble loaded from {filepath}")
        return instance
    
    def _validate_input(self, X: Union[List[str], pd.Series]) -> List[str]:
        """Validate and convert input to list of strings."""
        if isinstance(X, pd.Series):
            return X.astype(str).tolist()
        elif isinstance(X, list):
            return [str(text) for text in X]
        else:
            raise ValueError("X must be a list of strings or pandas Series")