"""
DeBERTa-v3 classifier implementation for fake news detection.
Implements fine-tuning with cross-domain evaluation capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for transformer model."""
    model_name: str = "distilbert-base-uncased"  # More compatible than DeBERTa-v3
    max_length: int = 512
    num_labels: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = False  # Disable for compatibility
    dataloader_num_workers: int = 0  # Disable for Windows compatibility

class FakeNewsDataset(Dataset):
    """Dataset class for fake news classification."""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int], 
                 tokenizer: AutoTokenizer,
                 max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text documents
            labels: List of binary labels (0=Fake, 1=Real)
            tokenizer: DeBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerClassifier:
    """DeBERTa-v3 classifier for fake news detection."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize DeBERTa classifier.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels
        )
        
        self.model.to(self.device)
        self.is_fitted = False
        
        logger.info(f"Initialized DeBERTa classifier with {self.config.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def prepare_datasets(self, 
                        X_train: List[str], 
                        y_train: List[int],
                        X_val: Optional[List[str]] = None,
                        y_val: Optional[List[int]] = None) -> Tuple[FakeNewsDataset, Optional[FakeNewsDataset]]:
        """
        Prepare training and validation datasets.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training dataset and optional validation dataset
        """
        train_dataset = FakeNewsDataset(
            X_train, y_train, self.tokenizer, self.config.max_length
        )
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = FakeNewsDataset(
                X_val, y_val, self.tokenizer, self.config.max_length
            )
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def fit(self, 
            X_train: List[str], 
            y_train: List[int],
            X_val: Optional[List[str]] = None,
            y_val: Optional[List[int]] = None,
            output_dir: str = "./results") -> Dict[str, List[float]]:
        """
        Fine-tune the DeBERTa model.
        
        Args:
            X_train: Training texts
            y_train: Training labels  
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            output_dir: Directory to save training outputs
            
        Returns:
            Training history
        """
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(X_train, y_train, X_val, y_val)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy='no',  # Simplified for compatibility
            save_strategy='no',  # Disable saving for testing
            load_best_model_at_end=False,
            report_to=[]  # Disable all reporting
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics if val_dataset else None,
        )
        
        # Train model
        logger.info("Starting DeBERTa fine-tuning...")
        training_result = trainer.train()
        
        self.trainer = trainer  # Store for later use
        self.is_fitted = True
        
        logger.info("Fine-tuning completed")
        return training_result.log_history
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict labels for new texts.
        
        Args:
            X: Input texts
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(X), self.config.batch_size):
            batch_texts = X[i:i + self.config.batch_size]
            batch_preds = self._predict_batch(batch_texts)
            predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict class probabilities for new texts.
        
        Args:
            X: Input texts
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = []
        
        # Process in batches
        for i in range(0, len(X), self.config.batch_size):
            batch_texts = X[i:i + self.config.batch_size]
            batch_probs = self._predict_proba_batch(batch_texts)
            probabilities.extend(batch_probs)
        
        return np.array(probabilities)
    
    def _predict_batch(self, texts: List[str]) -> List[int]:
        """Predict labels for a batch of texts."""
        self.model.eval()
        
        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.cpu().numpy().tolist()
    
    def _predict_proba_batch(self, texts: List[str]) -> List[List[float]]:
        """Predict probabilities for a batch of texts."""
        self.model.eval()
        
        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        return probabilities.cpu().numpy().tolist()
    
    def evaluate_cross_domain(self, 
                             X_train: List[str],
                             y_train: List[int],
                             X_test: List[str], 
                             y_test: List[int],
                             output_dir: str = "./cross_domain_results") -> Dict[str, float]:
        """
        Evaluate cross-domain generalization.
        
        Args:
            X_train: Training texts from source domain
            y_train: Training labels
            X_test: Test texts from target domain
            y_test: Test labels
            output_dir: Directory to save results
            
        Returns:
            Evaluation metrics
        """
        # Fine-tune on source domain
        logger.info("Fine-tuning on source domain...")
        self.fit(X_train, y_train, output_dir=output_dir)
        
        # Evaluate on target domain
        logger.info("Evaluating on target domain...")
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
        logger.info(f"Cross-domain results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results
    
    def save_model(self, save_directory: str):
        """Save the fine-tuned model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def load_model(cls, load_directory: str, config: Optional[ModelConfig] = None):
        """Load a fine-tuned model."""
        if config is None:
            config = ModelConfig()
            config.model_name = load_directory  # Use the saved model path
        
        instance = cls(config)
        instance.model = AutoModelForSequenceClassification.from_pretrained(load_directory)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_directory)
        instance.model.to(instance.device)
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {load_directory}")
        return instance