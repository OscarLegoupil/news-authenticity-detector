"""
Confidence calibration utilities for fake news detection models.
Implements Platt scaling and isotonic regression for probability calibration.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCalibrator:
    """Calibrate model probabilities for better confidence estimates."""
    
    def __init__(self, method: str = 'platt', cv: int = 3):
        """
        Initialize model calibrator.
        
        Args:
            method: Calibration method ('platt' or 'isotonic')
            cv: Number of cross-validation folds for calibration
        """
        self.method = method
        self.cv = cv
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, 
            models: Dict[str, Any], 
            X: Union[List[str], pd.Series], 
            y: Union[List, np.ndarray]):
        """
        Fit calibration on multiple models.
        
        Args:
            models: Dictionary of {model_name: model} to calibrate
            X: Training data
            y: Training labels
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        logger.info(f"Fitting {self.method} calibration on {len(models)} models...")
        
        for model_name, model in models.items():
            logger.info(f"Calibrating {model_name}...")
            
            # Create calibrated classifier
            calibrated_clf = CalibratedClassifierCV(
                model, method=self.method, cv=self.cv
            )
            
            # Fit calibrator
            calibrated_clf.fit(X, y)
            
            self.calibrators[model_name] = calibrated_clf
        
        self.is_fitted = True
        logger.info("Calibration fitting completed")
    
    def predict_proba(self, model_name: str, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Get calibrated probabilities for a specific model.
        
        Args:
            model_name: Name of the model
            X: Input data
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        if model_name not in self.calibrators:
            raise ValueError(f"Model {model_name} not found in calibrators")
        
        X = self._validate_input(X)
        return self.calibrators[model_name].predict_proba(X)
    
    def predict(self, model_name: str, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Get calibrated predictions for a specific model.
        
        Args:
            model_name: Name of the model
            X: Input data
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        X = self._validate_input(X)
        return self.calibrators[model_name].predict(X)
    
    def evaluate_calibration(self, 
                           models: Dict[str, Any],
                           X: Union[List[str], pd.Series],
                           y: Union[List, np.ndarray],
                           n_bins: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Evaluate calibration quality of models before and after calibration.
        
        Args:
            models: Original models to evaluate
            X: Test data
            y: Test labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            Calibration metrics for each model
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating calibration for {model_name}...")
            
            # Get original probabilities
            if hasattr(model, 'predict_proba'):
                orig_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            else:
                # For models without predict_proba, use predictions
                orig_pred = model.predict(X)
                orig_proba = orig_pred.astype(float)
            
            # Get calibrated probabilities
            if model_name in self.calibrators:
                cal_proba = self.calibrators[model_name].predict_proba(X)[:, 1]
            else:
                cal_proba = orig_proba  # No calibration available
            
            # Calculate calibration metrics
            orig_brier = brier_score_loss(y, orig_proba)
            cal_brier = brier_score_loss(y, cal_proba)
            
            try:
                orig_logloss = log_loss(y, orig_proba)
                cal_logloss = log_loss(y, cal_proba)
            except ValueError:
                # Handle case where probabilities are 0 or 1
                orig_proba_clipped = np.clip(orig_proba, 1e-7, 1-1e-7)
                cal_proba_clipped = np.clip(cal_proba, 1e-7, 1-1e-7)
                orig_logloss = log_loss(y, orig_proba_clipped)
                cal_logloss = log_loss(y, cal_proba_clipped)
            
            # Calculate calibration curve
            orig_fraction_pos, orig_mean_pred = calibration_curve(
                y, orig_proba, n_bins=n_bins
            )
            cal_fraction_pos, cal_mean_pred = calibration_curve(
                y, cal_proba, n_bins=n_bins
            )
            
            # Calculate Expected Calibration Error (ECE)
            orig_ece = self._calculate_ece(y, orig_proba, n_bins)
            cal_ece = self._calculate_ece(y, cal_proba, n_bins)
            
            results[model_name] = {
                'original': {
                    'brier_score': orig_brier,
                    'log_loss': orig_logloss,
                    'ece': orig_ece,
                    'calibration_curve': (orig_fraction_pos, orig_mean_pred)
                },
                'calibrated': {
                    'brier_score': cal_brier,
                    'log_loss': cal_logloss,
                    'ece': cal_ece,
                    'calibration_curve': (cal_fraction_pos, cal_mean_pred)
                }
            }
            
            logger.info(f"{model_name} - Original ECE: {orig_ece:.4f}, Calibrated ECE: {cal_ece:.4f}")
        
        return results
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            ECE score
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine which samples fall into the bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_calibration_curves(self, 
                               evaluation_results: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)):
        """
        Plot calibration curves for all models.
        
        Args:
            evaluation_results: Results from evaluate_calibration
            save_path: Path to save the plot
            figsize: Figure size
        """
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        axes = axes.flatten() if n_models > 2 else axes
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            ax = axes[i] if n_models > 1 else axes
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            # Plot original calibration
            orig_frac, orig_mean = results['original']['calibration_curve']
            ax.plot(orig_mean, orig_frac, 'o-', label=f'Original (ECE: {results["original"]["ece"]:.3f})')
            
            # Plot calibrated
            cal_frac, cal_mean = results['calibrated']['calibration_curve']
            ax.plot(cal_mean, cal_frac, 's-', label=f'Calibrated (ECE: {results["calibrated"]["ece"]:.3f})')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'{model_name} Calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        
        plt.show()
    
    def plot_reliability_diagram(self,
                                y_true: np.ndarray,
                                y_prob_dict: Dict[str, np.ndarray],
                                n_bins: int = 10,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8)):
        """
        Plot reliability diagram comparing multiple models.
        
        Args:
            y_true: True labels
            y_prob_dict: Dictionary of {model_name: probabilities}
            n_bins: Number of bins
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Colors for different models
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_prob_dict)))
        
        for (model_name, y_prob), color in zip(y_prob_dict.items(), colors):
            # Calculate calibration curve
            fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
            
            # Plot reliability diagram
            ax1.plot(mean_pred, fraction_pos, 'o-', color=color, label=model_name)
            
            # Plot histogram of predicted probabilities
            ax2.hist(y_prob, bins=n_bins, alpha=0.7, color=color, label=model_name, density=True)
        
        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {save_path}")
        
        plt.show()
    
    def get_confidence_intervals(self,
                               model_name: str,
                               X: Union[List[str], pd.Series],
                               confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Get confidence intervals for predictions.
        
        Args:
            model_name: Name of the model
            X: Input data
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with predictions, probabilities, and confidence intervals
        """
        if not self.is_fitted or model_name not in self.calibrators:
            raise ValueError(f"Calibrator for {model_name} not fitted")
        
        X = self._validate_input(X)
        
        # Get calibrated probabilities
        proba = self.calibrators[model_name].predict_proba(X)
        predictions = self.calibrators[model_name].predict(X)
        
        # Calculate confidence intervals (using normal approximation)
        alpha = 1 - confidence_level
        z_score = 1.96  # For 95% confidence interval
        
        # For binary classification, use binomial confidence intervals
        n_samples = len(X)
        p = proba[:, 1]  # Probability of positive class
        
        # Wilson score interval (better for probabilities near 0 or 1)
        denominator = 1 + z_score**2 / n_samples
        center = (p + z_score**2 / (2 * n_samples)) / denominator
        half_width = z_score * np.sqrt((p * (1 - p) + z_score**2 / (4 * n_samples)) / n_samples) / denominator
        
        lower_bound = center - half_width
        upper_bound = center + half_width
        
        return {
            'predictions': predictions,
            'probabilities': proba,
            'confidence_intervals': {
                'lower': lower_bound,
                'upper': upper_bound,
                'center': center
            }
        }
    
    def _validate_input(self, X: Union[List[str], pd.Series]) -> List[str]:
        """Validate and convert input to list of strings."""
        if isinstance(X, pd.Series):
            return X.astype(str).tolist()
        elif isinstance(X, list):
            return [str(text) for text in X]
        else:
            raise ValueError("X must be a list of strings or pandas Series")

class ConfidenceBasedFilter:
    """Filter predictions based on confidence thresholds."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.8,
                 use_calibrated: bool = True):
        """
        Initialize confidence-based filter.
        
        Args:
            confidence_threshold: Minimum confidence for accepting predictions
            use_calibrated: Whether to use calibrated probabilities
        """
        self.confidence_threshold = confidence_threshold
        self.use_calibrated = use_calibrated
    
    def filter_predictions(self,
                          predictions: np.ndarray,
                          probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Filter predictions based on confidence.
        
        Args:
            predictions: Model predictions
            probabilities: Model probabilities
            
        Returns:
            Dictionary with high/low confidence predictions and indices
        """
        # Calculate confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        # Create masks
        high_confidence_mask = confidence >= self.confidence_threshold
        low_confidence_mask = ~high_confidence_mask
        
        return {
            'high_confidence': {
                'predictions': predictions[high_confidence_mask],
                'probabilities': probabilities[high_confidence_mask],
                'confidence': confidence[high_confidence_mask],
                'indices': np.where(high_confidence_mask)[0]
            },
            'low_confidence': {
                'predictions': predictions[low_confidence_mask],
                'probabilities': probabilities[low_confidence_mask],
                'confidence': confidence[low_confidence_mask],
                'indices': np.where(low_confidence_mask)[0]
            },
            'stats': {
                'total_samples': len(predictions),
                'high_confidence_count': np.sum(high_confidence_mask),
                'high_confidence_ratio': np.mean(high_confidence_mask),
                'average_confidence': np.mean(confidence)
            }
        }
    
    def optimize_threshold(self,
                          y_true: np.ndarray,
                          probabilities: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
        """
        Optimize confidence threshold based on a metric.
        
        Args:
            y_true: True labels
            probabilities: Model probabilities
            metric: Metric to optimize ('accuracy', 'f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold and performance metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        
        thresholds = np.arange(0.5, 1.0, 0.05)
        best_threshold = 0.5
        best_score = 0
        results = []
        
        for threshold in thresholds:
            mask = confidence >= threshold
            
            if np.sum(mask) == 0:  # No samples above threshold
                continue
            
            y_filtered = y_true[mask]
            pred_filtered = predictions[mask]
            
            if metric == 'accuracy':
                score = accuracy_score(y_filtered, pred_filtered)
            elif metric == 'f1':
                score = f1_score(y_filtered, pred_filtered, average='weighted')
            elif metric == 'precision':
                score = precision_score(y_filtered, pred_filtered, average='weighted')
            elif metric == 'recall':
                score = recall_score(y_filtered, pred_filtered, average='weighted')
            else:
                raise ValueError("metric must be 'accuracy', 'f1', 'precision', or 'recall'")
            
            results.append({
                'threshold': threshold,
                'score': score,
                'coverage': np.mean(mask)
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.confidence_threshold = best_threshold
        
        return best_threshold, {
            'best_score': best_score,
            'optimization_results': results
        }