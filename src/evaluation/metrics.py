"""
Comprehensive evaluation metrics for cross-domain fake news detection.
Implements domain adaptation metrics, robustness testing, and interpretability analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    
    def to_dict(self) -> Dict[str, Union[float, str, np.ndarray]]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'roc_auc': self.roc_auc,
            'confusion_matrix': self.confusion_matrix,
            'classification_report': self.classification_report
        }

class CrossDomainEvaluator:
    """Comprehensive evaluator for cross-domain performance."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize evaluator.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.evaluation_history = []
    
    def evaluate_single_domain(self,
                              model: Any,
                              X: Union[List[str], pd.Series],
                              y: Union[List, np.ndarray],
                              model_name: str = "Model") -> EvaluationResults:
        """
        Evaluate model performance on a single domain.
        
        Args:
            model: Fitted model with predict and predict_proba methods
            X: Input texts
            y: True labels
            model_name: Name for logging
            
        Returns:
            Evaluation results
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            if y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
        else:
            y_proba_pos = y_pred.astype(float)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        
        try:
            roc_auc = roc_auc_score(y, y_proba_pos)
        except ValueError:
            roc_auc = 0.5  # Default for constant predictions
        
        cm = confusion_matrix(y, y_pred)
        cr = classification_report(y, y_pred)
        
        results = EvaluationResults(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            classification_report=cr
        )
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        return results
    
    def evaluate_cross_domain(self,
                             model: Any,
                             X_source: Union[List[str], pd.Series],
                             y_source: Union[List, np.ndarray],
                             X_target: Union[List[str], pd.Series],
                             y_target: Union[List, np.ndarray],
                             model_name: str = "Model") -> Dict[str, EvaluationResults]:
        """
        Evaluate cross-domain performance.
        
        Args:
            model: Model to evaluate
            X_source: Source domain texts
            y_source: Source domain labels
            X_target: Target domain texts
            y_target: Target domain labels
            model_name: Name for logging
            
        Returns:
            Dictionary with source and target domain results
        """
        logger.info(f"Evaluating cross-domain performance for {model_name}...")
        
        # Train on source domain
        model.fit(X_source, y_source)
        
        # Evaluate on source domain
        source_results = self.evaluate_single_domain(model, X_source, y_source, f"{model_name}_source")
        
        # Evaluate on target domain
        target_results = self.evaluate_single_domain(model, X_target, y_target, f"{model_name}_target")
        
        # Calculate domain gap
        domain_gap = source_results.accuracy - target_results.accuracy
        logger.info(f"{model_name} domain gap: {domain_gap:.4f}")
        
        results = {
            'source': source_results,
            'target': target_results,
            'domain_gap': domain_gap
        }
        
        # Store in history
        self.evaluation_history.append({
            'model_name': model_name,
            'results': results
        })
        
        return results
    
    def evaluate_multiple_models(self,
                                models: Dict[str, Any],
                                X_source: Union[List[str], pd.Series],
                                y_source: Union[List, np.ndarray],
                                X_target: Union[List[str], pd.Series],
                                y_target: Union[List, np.ndarray]) -> Dict[str, Dict[str, EvaluationResults]]:
        """
        Evaluate multiple models for cross-domain performance.
        
        Args:
            models: Dictionary of {model_name: model}
            X_source: Source domain texts
            y_source: Source domain labels
            X_target: Target domain texts
            y_target: Target domain labels
            
        Returns:
            Dictionary of cross-domain results for each model
        """
        all_results = {}
        
        for model_name, model in models.items():
            try:
                results = self.evaluate_cross_domain(
                    model, X_source, y_source, X_target, y_target, model_name
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return all_results
    
    def create_comparison_report(self,
                               results: Dict[str, Dict[str, EvaluationResults]],
                               save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a comparison report of all models.
        
        Args:
            results: Results from evaluate_multiple_models
            save_path: Path to save the report
            
        Returns:
            DataFrame with comparison metrics
        """
        report_data = []
        
        for model_name, model_results in results.items():
            source_res = model_results['source']
            target_res = model_results['target']
            domain_gap = model_results['domain_gap']
            
            report_data.append({
                'Model': model_name,
                'Source_Accuracy': source_res.accuracy,
                'Source_F1': source_res.f1_score,
                'Source_AUC': source_res.roc_auc,
                'Target_Accuracy': target_res.accuracy,
                'Target_F1': target_res.f1_score,
                'Target_AUC': target_res.roc_auc,
                'Domain_Gap': domain_gap,
                'Generalization_Score': target_res.f1_score  # Target F1 as generalization metric
            })
        
        df = pd.DataFrame(report_data)
        df = df.sort_values('Generalization_Score', ascending=False)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Comparison report saved to {save_path}")
        
        return df
    
    def plot_performance_comparison(self,
                                  results: Dict[str, Dict[str, EvaluationResults]],
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 8)):
        """
        Plot performance comparison across models and domains.
        
        Args:
            results: Results from evaluate_multiple_models
            save_path: Path to save the plot
            figsize: Figure size
        """
        # Prepare data for plotting
        model_names = list(results.keys())
        source_acc = [results[name]['source'].accuracy for name in model_names]
        target_acc = [results[name]['target'].accuracy for name in model_names]
        source_f1 = [results[name]['source'].f1_score for name in model_names]
        target_f1 = [results[name]['target'].f1_score for name in model_names]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy comparison
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, source_acc, width, label='Source Domain', alpha=0.8)
        ax1.bar(x + width/2, target_acc, width, label='Target Domain', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 comparison
        ax2.bar(x - width/2, source_f1, width, label='Source Domain', alpha=0.8)
        ax2.bar(x + width/2, target_f1, width, label='Target Domain', alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Domain gap
        domain_gaps = [results[name]['domain_gap'] for name in model_names]
        ax3.bar(model_names, domain_gaps, alpha=0.8, color='orange')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Domain Gap (Source - Target Accuracy)')
        ax3.set_title('Domain Gap Analysis')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot: Source vs Target performance
        ax4.scatter(source_acc, target_acc, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax4.annotate(name, (source_acc[i], target_acc[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Perfect generalization line
        min_acc = min(min(source_acc), min(target_acc))
        max_acc = max(max(source_acc), max(target_acc))
        ax4.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5, label='Perfect Generalization')
        
        ax4.set_xlabel('Source Domain Accuracy')
        ax4.set_ylabel('Target Domain Accuracy')
        ax4.set_title('Source vs Target Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self,
                               results: Dict[str, Dict[str, EvaluationResults]],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)):
        """
        Plot confusion matrices for all models.
        
        Args:
            results: Results from evaluate_multiple_models
            save_path: Path to save the plot
            figsize: Figure size
        """
        n_models = len(results)
        fig, axes = plt.subplots(2, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (model_name, model_results) in enumerate(results.items()):
            # Source domain confusion matrix
            sns.heatmap(model_results['source'].confusion_matrix, 
                       annot=True, fmt='d', cmap='Blues',
                       ax=axes[0, i], cbar=False)
            axes[0, i].set_title(f'{model_name}\nSource Domain')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')
            
            # Target domain confusion matrix
            sns.heatmap(model_results['target'].confusion_matrix, 
                       annot=True, fmt='d', cmap='Reds',
                       ax=axes[1, i], cbar=False)
            axes[1, i].set_title(f'{model_name}\nTarget Domain')
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def calculate_stability_metrics(self,
                                  model: Any,
                                  X: Union[List[str], pd.Series],
                                  y: Union[List, np.ndarray],
                                  n_trials: int = 10) -> Dict[str, float]:
        """
        Calculate model stability across multiple runs.
        
        Args:
            model: Model to evaluate
            X: Input texts
            y: True labels
            n_trials: Number of evaluation trials
            
        Returns:
            Stability metrics
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        scores = []
        
        for trial in range(n_trials):
            # Use cross-validation for stability assessment
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring='f1_weighted',
                random_state=self.random_state + trial
            )
            scores.append(cv_scores.mean())
        
        scores = np.array(scores)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'coefficient_of_variation': scores.std() / scores.mean() if scores.mean() > 0 else float('inf')
        }
    
    def evaluate_robustness(self,
                          model: Any,
                          X: Union[List[str], pd.Series],
                          y: Union[List, np.ndarray],
                          perturbation_methods: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model robustness to text perturbations.
        
        Args:
            model: Model to evaluate
            X: Input texts
            y: True labels
            perturbation_methods: List of perturbation methods to apply
            
        Returns:
            Robustness metrics
        """
        if perturbation_methods is None:
            perturbation_methods = ['char_swap', 'word_drop', 'synonym_replace']
        
        X = self._validate_input(X)
        y = np.array(y)
        
        # Original performance
        original_preds = model.predict(X)
        original_accuracy = accuracy_score(y, original_preds)
        
        robustness_results = {'original_accuracy': original_accuracy}
        
        for method in perturbation_methods:
            try:
                # Apply perturbation (simplified implementation)
                X_perturbed = self._apply_perturbation(X, method)
                
                # Evaluate on perturbed data
                perturbed_preds = model.predict(X_perturbed)
                perturbed_accuracy = accuracy_score(y, perturbed_preds)
                
                # Calculate robustness score
                robustness_score = perturbed_accuracy / original_accuracy if original_accuracy > 0 else 0
                
                robustness_results[f'{method}_accuracy'] = perturbed_accuracy
                robustness_results[f'{method}_robustness'] = robustness_score
                
            except Exception as e:
                logger.warning(f"Could not apply perturbation {method}: {e}")
                continue
        
        return robustness_results
    
    def _apply_perturbation(self, texts: List[str], method: str) -> List[str]:
        """
        Apply simple text perturbations.
        
        Args:
            texts: List of texts to perturb
            method: Perturbation method
            
        Returns:
            List of perturbed texts
        """
        import random
        random.seed(self.random_state)
        
        perturbed = []
        
        for text in texts:
            if method == 'char_swap':
                # Randomly swap adjacent characters
                if len(text) > 1:
                    chars = list(text)
                    for _ in range(max(1, len(chars) // 20)):  # Swap ~5% of characters
                        i = random.randint(0, len(chars) - 2)
                        chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    text = ''.join(chars)
            
            elif method == 'word_drop':
                # Randomly drop words
                words = text.split()
                if len(words) > 1:
                    n_drop = max(1, len(words) // 10)  # Drop ~10% of words
                    for _ in range(n_drop):
                        if words:  # Check if words list is not empty
                            words.pop(random.randint(0, len(words) - 1))
                    text = ' '.join(words)
            
            elif method == 'synonym_replace':
                # Simple character substitution (simplified synonym replacement)
                substitutions = {'good': 'great', 'bad': 'terrible', 'big': 'large', 'small': 'tiny'}
                for original, replacement in substitutions.items():
                    text = text.replace(original, replacement)
            
            perturbed.append(text)
        
        return perturbed
    
    def _validate_input(self, X: Union[List[str], pd.Series]) -> List[str]:
        """Validate and convert input to list of strings."""
        if isinstance(X, pd.Series):
            return X.astype(str).tolist()
        elif isinstance(X, list):
            return [str(text) for text in X]
        else:
            raise ValueError("X must be a list of strings or pandas Series")

class ModelBenchmark:
    """Benchmarking suite for fake news detection models."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.evaluator = CrossDomainEvaluator()
        self.benchmark_results = {}
    
    def run_benchmark(self,
                     models: Dict[str, Any],
                     datasets: Dict[str, Tuple[List[str], List[int]]],
                     cross_domain_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across models and datasets.
        
        Args:
            models: Dictionary of {model_name: model}
            datasets: Dictionary of {dataset_name: (texts, labels)}
            cross_domain_pairs: List of (source_dataset, target_dataset) pairs
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting comprehensive benchmark...")
        
        results = {
            'single_domain': {},
            'cross_domain': {},
            'stability': {},
            'robustness': {}
        }
        
        # Single domain evaluation
        for dataset_name, (X, y) in datasets.items():
            logger.info(f"Evaluating on {dataset_name} dataset...")
            results['single_domain'][dataset_name] = {}
            
            for model_name, model in models.items():
                try:
                    model.fit(X, y)
                    eval_results = self.evaluator.evaluate_single_domain(model, X, y, model_name)
                    results['single_domain'][dataset_name][model_name] = eval_results
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
        
        # Cross-domain evaluation
        for source_dataset, target_dataset in cross_domain_pairs:
            if source_dataset in datasets and target_dataset in datasets:
                logger.info(f"Cross-domain: {source_dataset} -> {target_dataset}")
                
                X_source, y_source = datasets[source_dataset]
                X_target, y_target = datasets[target_dataset]
                
                pair_key = f"{source_dataset}_to_{target_dataset}"
                results['cross_domain'][pair_key] = {}
                
                for model_name, model in models.items():
                    try:
                        cross_results = self.evaluator.evaluate_cross_domain(
                            model, X_source, y_source, X_target, y_target, model_name
                        )
                        results['cross_domain'][pair_key][model_name] = cross_results
                    except Exception as e:
                        logger.error(f"Error in cross-domain evaluation for {model_name}: {e}")
        
        # Stability evaluation (using first dataset)
        if datasets:
            first_dataset = list(datasets.keys())[0]
            X, y = datasets[first_dataset]
            logger.info(f"Evaluating stability on {first_dataset}...")
            
            for model_name, model in models.items():
                try:
                    stability_results = self.evaluator.calculate_stability_metrics(model, X, y)
                    results['stability'][model_name] = stability_results
                except Exception as e:
                    logger.error(f"Error evaluating stability for {model_name}: {e}")
        
        # Robustness evaluation
        if datasets:
            first_dataset = list(datasets.keys())[0]
            X, y = datasets[first_dataset]
            logger.info(f"Evaluating robustness on {first_dataset}...")
            
            for model_name, model in models.items():
                try:
                    model.fit(X, y)
                    robustness_results = self.evaluator.evaluate_robustness(model, X, y)
                    results['robustness'][model_name] = robustness_results
                except Exception as e:
                    logger.error(f"Error evaluating robustness for {model_name}: {e}")
        
        self.benchmark_results = results
        logger.info("Benchmark completed")
        
        return results
    
    def generate_benchmark_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if not self.benchmark_results:
            raise ValueError("No benchmark results available. Run benchmark first.")
        
        report_lines = ["# Fake News Detection Model Benchmark Report\n"]
        
        # Single domain results
        if 'single_domain' in self.benchmark_results:
            report_lines.append("## Single Domain Performance\n")
            for dataset, models in self.benchmark_results['single_domain'].items():
                report_lines.append(f"### {dataset} Dataset\n")
                report_lines.append("| Model | Accuracy | F1 Score | Precision | Recall | AUC |")
                report_lines.append("|-------|----------|----------|-----------|--------|-----|")
                
                for model_name, results in models.items():
                    report_lines.append(
                        f"| {model_name} | {results.accuracy:.4f} | {results.f1_score:.4f} | "
                        f"{results.precision:.4f} | {results.recall:.4f} | {results.roc_auc:.4f} |"
                    )
                report_lines.append("")
        
        # Cross-domain results
        if 'cross_domain' in self.benchmark_results:
            report_lines.append("## Cross-Domain Performance\n")
            for pair, models in self.benchmark_results['cross_domain'].items():
                report_lines.append(f"### {pair}\n")
                report_lines.append("| Model | Source Acc | Target Acc | Domain Gap | Source F1 | Target F1 |")
                report_lines.append("|-------|------------|------------|------------|-----------|-----------|")
                
                for model_name, results in models.items():
                    source_acc = results['source'].accuracy
                    target_acc = results['target'].accuracy
                    domain_gap = results['domain_gap']
                    source_f1 = results['source'].f1_score
                    target_f1 = results['target'].f1_score
                    
                    report_lines.append(
                        f"| {model_name} | {source_acc:.4f} | {target_acc:.4f} | "
                        f"{domain_gap:.4f} | {source_f1:.4f} | {target_f1:.4f} |"
                    )
                report_lines.append("")
        
        # Stability results
        if 'stability' in self.benchmark_results:
            report_lines.append("## Model Stability\n")
            report_lines.append("| Model | Mean Score | Std Dev | Coefficient of Variation |")
            report_lines.append("|-------|------------|---------|-------------------------|")
            
            for model_name, results in self.benchmark_results['stability'].items():
                report_lines.append(
                    f"| {model_name} | {results['mean_score']:.4f} | "
                    f"{results['std_score']:.4f} | {results['coefficient_of_variation']:.4f} |"
                )
            report_lines.append("")
        
        # Robustness results
        if 'robustness' in self.benchmark_results:
            report_lines.append("## Model Robustness\n")
            report_lines.append("| Model | Original Acc | Char Swap | Word Drop | Synonym Replace |")
            report_lines.append("|-------|--------------|-----------|-----------|-----------------|")
            
            for model_name, results in self.benchmark_results['robustness'].items():
                char_swap = results.get('char_swap_robustness', 'N/A')
                word_drop = results.get('word_drop_robustness', 'N/A')
                synonym_replace = results.get('synonym_replace_robustness', 'N/A')
                
                report_lines.append(
                    f"| {model_name} | {results['original_accuracy']:.4f} | "
                    f"{char_swap if char_swap == 'N/A' else f'{char_swap:.4f}'} | "
                    f"{word_drop if word_drop == 'N/A' else f'{word_drop:.4f}'} | "
                    f"{synonym_replace if synonym_replace == 'N/A' else f'{synonym_replace:.4f}'} |"
                )
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Benchmark report saved to {save_path}")
        
        return report