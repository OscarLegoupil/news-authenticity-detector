"""
Main pipeline orchestration for fake news detection.
Provides unified interface for training, evaluation, and inference.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import yaml

from src.data.preprocessing import TextPreprocessor, DatasetLoader
from src.models.traditional.bow_tfidf import BOWClassifier, TFIDFClassifier
from src.models.transformers.deberta_classifier import TransformerClassifier as DeBERTaClassifier, ModelConfig
from src.models.ensemble import EnsembleClassifier
from src.evaluation.metrics import CrossDomainEvaluator
try:
    from src.training.trainer import TrainingPipeline
except ImportError:
    # Define a minimal TrainingPipeline for basic functionality
    class TrainingPipeline:
        def __init__(self):
            pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the fake news detection pipeline."""
    # Data paths
    isot_fake_path: str = "data/raw/Fake.csv"
    isot_real_path: str = "data/raw/True.csv"
    kaggle_path: str = "data/raw/fake_or_real_news.csv"
    
    # Model configurations
    traditional_max_features: int = 1000
    traditional_min_df: int = 2
    
    # DeBERTa configuration
    deberta_model_name: str = "microsoft/deberta-v3-base"
    deberta_max_length: int = 512
    deberta_batch_size: int = 16
    deberta_epochs: int = 3
    deberta_learning_rate: float = 2e-5
    
    # Ensemble configuration
    ensemble_method: str = "weighted_voting"  # weighted_voting, stacking, confidence_based
    
    # Output paths
    models_dir: str = "models/checkpoints"
    results_dir: str = "results/experiments"
    
    # Training options
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42

class FakeNewsDetector:
    """Main pipeline class for fake news detection."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the fake news detection pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.preprocessor = TextPreprocessor()
        self.loader = DatasetLoader()
        
        # Model storage
        self.traditional_models = {}
        self.transformer_model = None
        self.ensemble_model = None
        
        # Data storage
        self.datasets = {}
        
        # Create output directories
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        logger.info("FakeNewsDetector initialized")
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess all datasets.
        
        Returns:
            Dictionary of loaded datasets
        """
        logger.info("Loading datasets...")
        
        # Load ISOT dataset
        try:
            isot_df = self.loader.load_isot_dataset(
                self.config.isot_fake_path,
                self.config.isot_real_path
            )
            self.datasets['isot'] = isot_df
            logger.info(f"Loaded ISOT dataset: {len(isot_df)} articles")
        except Exception as e:
            logger.warning(f"Could not load ISOT dataset: {e}")
        
        # Load Kaggle dataset
        try:
            kaggle_df = self.loader.load_kaggle_dataset(self.config.kaggle_path)
            self.datasets['kaggle'] = kaggle_df
            logger.info(f"Loaded Kaggle dataset: {len(kaggle_df)} articles")
        except Exception as e:
            logger.warning(f"Could not load Kaggle dataset: {e}")
        
        # Apply preprocessing
        for name, df in self.datasets.items():
            # Traditional preprocessing
            df = self.preprocessor.preprocess_dataset(
                df, preprocessing_type='traditional'
            )
            # Transformer preprocessing
            df = self.preprocessor.preprocess_dataset(
                df, preprocessing_type='transformer'  
            )
            self.datasets[name] = df
        
        return self.datasets
    
    def train_traditional_models(self, dataset_name: str = 'isot') -> Dict[str, Any]:
        """
        Train traditional ML models (BoW, TF-IDF).
        
        Args:
            dataset_name: Name of dataset to train on
            
        Returns:
            Training results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        df = self.datasets[dataset_name]
        X = df['text_clean_traditional']
        y = df['label_binary']
        
        results = {}
        
        # Train BoW classifier
        logger.info("Training BoW classifier...")
        bow_classifier = BOWClassifier(
            max_features=self.config.traditional_max_features,
            min_df=self.config.traditional_min_df
        )
        bow_results = bow_classifier.cross_validate(X, y, cv=self.config.cross_validation_folds)
        bow_classifier.fit(X, y)  # Fit on full dataset
        
        self.traditional_models['bow'] = bow_classifier
        results['bow'] = bow_results
        
        # Train TF-IDF classifier
        logger.info("Training TF-IDF classifier...")
        tfidf_classifier = TFIDFClassifier(
            max_features=self.config.traditional_max_features,
            min_df=self.config.traditional_min_df
        )
        tfidf_results = tfidf_classifier.cross_validate(X, y, cv=self.config.cross_validation_folds)
        tfidf_classifier.fit(X, y)  # Fit on full dataset
        
        self.traditional_models['tfidf'] = tfidf_classifier
        results['tfidf'] = tfidf_results
        
        logger.info("Traditional models training completed")
        return results
    
    def train_transformer_model(self, dataset_name: str = 'isot') -> Dict[str, Any]:
        """
        Train DeBERTa transformer model.
        
        Args:
            dataset_name: Name of dataset to train on
            
        Returns:
            Training results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        df = self.datasets[dataset_name]
        X = df['text_clean_transformer'].tolist()
        y = df['label_binary'].tolist()
        
        # Create DeBERTa configuration
        deberta_config = ModelConfig(
            model_name=self.config.deberta_model_name,
            max_length=self.config.deberta_max_length,
            batch_size=self.config.deberta_batch_size,
            num_epochs=self.config.deberta_epochs,
            learning_rate=self.config.deberta_learning_rate
        )
        
        # Initialize and train model
        logger.info("Training DeBERTa model...")
        self.transformer_model = DeBERTaClassifier(deberta_config)
        
        # Split data for training/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        # Train model
        output_dir = os.path.join(self.config.results_dir, "deberta_training")
        training_history = self.transformer_model.fit(
            X_train, y_train, X_val, y_val, output_dir=output_dir
        )
        
        logger.info("DeBERTa training completed")
        return {'training_history': training_history}
    
    def create_ensemble(self, models_to_ensemble: Optional[List[str]] = None) -> EnsembleClassifier:
        """
        Create ensemble model combining multiple trained models.
        
        Args:
            models_to_ensemble: List of model names to include in ensemble
            
        Returns:
            Trained ensemble classifier
        """
        if models_to_ensemble is None:
            models_to_ensemble = ['bow', 'tfidf']
            if self.transformer_model is not None:
                models_to_ensemble.append('deberta')
        
        # Collect models
        models = []
        model_names = []
        
        for model_name in models_to_ensemble:
            if model_name in self.traditional_models:
                models.append(self.traditional_models[model_name])
                model_names.append(model_name)
            elif model_name == 'deberta' and self.transformer_model is not None:
                models.append(self.transformer_model)
                model_names.append('deberta')
            else:
                logger.warning(f"Model {model_name} not available for ensemble")
        
        if len(models) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        # Create and fit ensemble
        logger.info(f"Creating ensemble with models: {model_names}")
        self.ensemble_model = EnsembleClassifier(
            models=models,
            model_names=model_names,
            combination_method=self.config.ensemble_method
        )
        
        # Fit ensemble on ISOT dataset
        if 'isot' in self.datasets:
            df = self.datasets['isot']
            X = df['text_clean_traditional'].tolist()  # Use traditional preprocessing for compatibility
            y = df['label_binary'].tolist()
            
            self.ensemble_model.fit(X, y, calibrate=True)
            logger.info("Ensemble model fitted")
        
        return self.ensemble_model
    
    def evaluate_cross_domain(self, 
                             source_dataset: str = 'isot',
                             target_dataset: str = 'kaggle') -> Dict[str, Dict[str, float]]:
        """
        Evaluate cross-domain performance of all models.
        
        Args:
            source_dataset: Dataset to train on
            target_dataset: Dataset to test on
            
        Returns:
            Cross-domain evaluation results
        """
        if source_dataset not in self.datasets or target_dataset not in self.datasets:
            raise ValueError("Both datasets must be loaded")
        
        source_df = self.datasets[source_dataset]
        target_df = self.datasets[target_dataset]
        
        results = {}
        
        # Evaluate traditional models
        for model_name, model in self.traditional_models.items():
            logger.info(f"Evaluating {model_name} cross-domain...")
            
            X_train = source_df['text_clean_traditional']
            y_train = source_df['label_binary']
            X_test = target_df['text_clean_traditional']
            y_test = target_df['label_binary']
            
            model_results = model.evaluate_cross_domain(X_train, y_train, X_test, y_test)
            results[model_name] = model_results
        
        # Evaluate transformer model
        if self.transformer_model is not None:
            logger.info("Evaluating DeBERTa cross-domain...")
            
            X_train = source_df['text_clean_transformer'].tolist()
            y_train = source_df['label_binary'].tolist()
            X_test = target_df['text_clean_transformer'].tolist()
            y_test = target_df['label_binary'].tolist()
            
            deberta_results = self.transformer_model.evaluate_cross_domain(
                X_train, y_train, X_test, y_test,
                output_dir=os.path.join(self.config.results_dir, "cross_domain_deberta")
            )
            results['deberta'] = deberta_results
        
        # Evaluate ensemble model
        if self.ensemble_model is not None:
            logger.info("Evaluating ensemble cross-domain...")
            
            # Use traditional preprocessing for ensemble compatibility
            X_test = target_df['text_clean_traditional'].tolist()
            y_test = target_df['label_binary'].tolist()
            
            y_pred = self.ensemble_model.predict(X_test)
            
            from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            ensemble_results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            results['ensemble'] = ensemble_results
        
        # Log summary
        logger.info("Cross-domain evaluation results:")
        for model_name, model_results in results.items():
            logger.info(f"{model_name}: Accuracy={model_results['accuracy']:.4f}, F1={model_results['f1_score']:.4f}")
        
        return results
    
    def predict(self, texts: List[str], model_type: str = 'ensemble') -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
            model_type: Model to use ('bow', 'tfidf', 'deberta', 'ensemble')
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess texts
        if model_type in ['bow', 'tfidf', 'ensemble']:
            clean_texts = [self.preprocessor.clean_text_traditional(text) for text in texts]
        else:
            clean_texts = [self.preprocessor.clean_text_transformer(text) for text in texts]
        
        # Select model
        if model_type == 'ensemble' and self.ensemble_model is not None:
            model = self.ensemble_model
        elif model_type in self.traditional_models:
            model = self.traditional_models[model_type]
        elif model_type == 'deberta' and self.transformer_model is not None:
            model = self.transformer_model
        else:
            raise ValueError(f"Model {model_type} not available")
        
        # Make predictions
        predictions = model.predict(clean_texts)
        probabilities = model.predict_proba(clean_texts)
        
        # Convert to human-readable labels
        labels = ['Fake' if pred == 0 else 'Real' for pred in predictions]
        
        return {
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
            'confidence': np.max(probabilities, axis=1)
        }
    
    def save_models(self):
        """Save all trained models."""
        # Save traditional models
        for name, model in self.traditional_models.items():
            filepath = os.path.join(self.config.models_dir, f"{name}_model.joblib")
            model.save_model(filepath)
        
        # Save transformer model
        if self.transformer_model is not None:
            save_dir = os.path.join(self.config.models_dir, "deberta_model")
            self.transformer_model.save_model(save_dir)
        
        # Save ensemble model
        if self.ensemble_model is not None:
            filepath = os.path.join(self.config.models_dir, "ensemble_model.joblib")
            self.ensemble_model.save_ensemble(filepath)
        
        logger.info("All models saved")
    
    def load_models(self):
        """Load all saved models."""
        # Load traditional models
        for model_type in ['bow', 'tfidf']:
            filepath = os.path.join(self.config.models_dir, f"{model_type}_model.joblib")
            if os.path.exists(filepath):
                if model_type == 'bow':
                    self.traditional_models[model_type] = BOWClassifier.load_model(filepath)
                else:
                    self.traditional_models[model_type] = TFIDFClassifier.load_model(filepath)
        
        # Load transformer model
        deberta_dir = os.path.join(self.config.models_dir, "deberta_model")
        if os.path.exists(deberta_dir):
            self.transformer_model = DeBERTaClassifier.load_model(deberta_dir)
        
        # Load ensemble model
        ensemble_path = os.path.join(self.config.models_dir, "ensemble_model.joblib")
        if os.path.exists(ensemble_path):
            self.ensemble_model = EnsembleClassifier.load_ensemble(ensemble_path)
        
        logger.info("Models loaded")
    
    def save_config(self, filepath: str):
        """Save pipeline configuration to YAML file."""
        config_dict = {
            'data_paths': {
                'isot_fake_path': self.config.isot_fake_path,
                'isot_real_path': self.config.isot_real_path,
                'kaggle_path': self.config.kaggle_path
            },
            'traditional_models': {
                'max_features': self.config.traditional_max_features,
                'min_df': self.config.traditional_min_df
            },
            'deberta': {
                'model_name': self.config.deberta_model_name,
                'max_length': self.config.deberta_max_length,
                'batch_size': self.config.deberta_batch_size,
                'epochs': self.config.deberta_epochs,
                'learning_rate': self.config.deberta_learning_rate
            },
            'ensemble': {
                'method': self.config.ensemble_method
            },
            'training': {
                'cross_validation_folds': self.config.cross_validation_folds,
                'test_size': self.config.test_size,
                'validation_size': self.config.validation_size,
                'random_state': self.config.random_state
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_config_file(cls, filepath: str):
        """Create pipeline from configuration file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert config dict to PipelineConfig
        config = PipelineConfig()
        
        # Update config attributes from file
        if 'data_paths' in config_dict:
            config.isot_fake_path = config_dict['data_paths'].get('isot_fake_path', config.isot_fake_path)
            config.isot_real_path = config_dict['data_paths'].get('isot_real_path', config.isot_real_path)
            config.kaggle_path = config_dict['data_paths'].get('kaggle_path', config.kaggle_path)
        
        if 'traditional_models' in config_dict:
            config.traditional_max_features = config_dict['traditional_models'].get('max_features', config.traditional_max_features)
            config.traditional_min_df = config_dict['traditional_models'].get('min_df', config.traditional_min_df)
        
        if 'deberta' in config_dict:
            config.deberta_model_name = config_dict['deberta'].get('model_name', config.deberta_model_name)
            config.deberta_max_length = config_dict['deberta'].get('max_length', config.deberta_max_length)
            config.deberta_batch_size = config_dict['deberta'].get('batch_size', config.deberta_batch_size)
            config.deberta_epochs = config_dict['deberta'].get('epochs', config.deberta_epochs)
            config.deberta_learning_rate = config_dict['deberta'].get('learning_rate', config.deberta_learning_rate)
        
        return cls(config)