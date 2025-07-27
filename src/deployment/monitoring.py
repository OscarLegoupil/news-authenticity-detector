"""
Model monitoring and drift detection for production deployment.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for a single prediction."""
    timestamp: float
    model_type: str
    confidence: float
    processing_time: float
    prediction: Optional[str] = None


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize model monitor.
        
        Args:
            max_history: Maximum number of predictions to keep in memory
        """
        self.max_history = max_history
        self.predictions: deque = deque(maxlen=max_history)
        self.stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'confidence_sum': 0.0,
            'confidence_values': deque(maxlen=1000)
        })
        self.drift_detector = DriftDetector()
        
        # Alert thresholds
        self.confidence_threshold = 0.5
        self.processing_time_threshold = 5.0  # seconds
        self.drift_threshold = 0.05
        
        logger.info("ModelMonitor initialized")
    
    def record_prediction(self, model_type: str, confidence: float, processing_time: float, prediction: Optional[str] = None):
        """
        Record a new prediction for monitoring.
        
        Args:
            model_type: Type of model used
            confidence: Prediction confidence
            processing_time: Time taken for prediction
            prediction: The actual prediction (optional)
        """
        timestamp = time.time()
        
        # Create metrics object
        metrics = PredictionMetrics(
            timestamp=timestamp,
            model_type=model_type,
            confidence=confidence,
            processing_time=processing_time,
            prediction=prediction
        )
        
        # Store in history
        self.predictions.append(metrics)
        
        # Update statistics
        stats = self.stats[model_type]
        stats['count'] += 1
        stats['total_time'] += processing_time
        stats['confidence_sum'] += confidence
        stats['confidence_values'].append(confidence)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update drift detection
        self.drift_detector.add_prediction(confidence, model_type)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        total_predictions = sum(stats['count'] for stats in self.stats.values())
        
        predictions_by_model = {
            model: stats['count'] for model, stats in self.stats.items()
        }
        
        # Calculate average processing time
        total_time = sum(stats['total_time'] for stats in self.stats.values())
        avg_processing_time = total_time / total_predictions if total_predictions > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'predictions_by_model': predictions_by_model,
            'avg_processing_time': avg_processing_time,
            'monitoring_window_hours': self._get_monitoring_window_hours(),
            'drift_status': self.drift_detector.get_drift_status()
        }
    
    def get_model_stats(self, model_type: str) -> Dict[str, Any]:
        """Get statistics for a specific model."""
        if model_type not in self.stats:
            return {'error': f'No data for model {model_type}'}
        
        stats = self.stats[model_type]
        
        # Calculate metrics
        avg_confidence = stats['confidence_sum'] / stats['count'] if stats['count'] > 0 else 0
        avg_processing_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
        
        # Confidence distribution
        confidence_values = list(stats['confidence_values'])
        confidence_std = np.std(confidence_values) if confidence_values else 0
        
        return {
            'model_type': model_type,
            'total_predictions': stats['count'],
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'avg_processing_time': avg_processing_time,
            'min_confidence': min(confidence_values) if confidence_values else 0,
            'max_confidence': max(confidence_values) if confidence_values else 0
        }
    
    def get_recent_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent prediction metrics."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_predictions = [
            asdict(pred) for pred in self.predictions 
            if pred.timestamp >= cutoff_time
        ]
        
        return recent_predictions
    
    def detect_anomalies(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect anomalies in predictions."""
        anomalies = []
        
        # Filter predictions by model if specified
        predictions = self.predictions
        if model_type:
            predictions = [p for p in predictions if p.model_type == model_type]
        
        # Detect confidence anomalies
        confidences = [p.confidence for p in predictions]
        if confidences:
            confidence_mean = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            for pred in predictions:
                z_score = abs(pred.confidence - confidence_mean) / confidence_std if confidence_std > 0 else 0
                if z_score > 3:  # 3 standard deviations
                    anomalies.append({
                        'type': 'confidence_anomaly',
                        'timestamp': pred.timestamp,
                        'model_type': pred.model_type,
                        'confidence': pred.confidence,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 4 else 'medium'
                    })
        
        # Detect processing time anomalies
        processing_times = [p.processing_time for p in predictions]
        if processing_times:
            time_mean = np.mean(processing_times)
            time_std = np.std(processing_times)
            
            for pred in predictions:
                if pred.processing_time > time_mean + 3 * time_std:
                    anomalies.append({
                        'type': 'processing_time_anomaly',
                        'timestamp': pred.timestamp,
                        'model_type': pred.model_type,
                        'processing_time': pred.processing_time,
                        'threshold': time_mean + 3 * time_std,
                        'severity': 'high' if pred.processing_time > self.processing_time_threshold else 'medium'
                    })
        
        return anomalies
    
    def _check_alerts(self, metrics: PredictionMetrics):
        """Check if any alerts should be triggered."""
        # Low confidence alert
        if metrics.confidence < self.confidence_threshold:
            logger.warning(
                f"Low confidence prediction: {metrics.confidence:.3f} "
                f"for model {metrics.model_type}"
            )
        
        # High processing time alert
        if metrics.processing_time > self.processing_time_threshold:
            logger.warning(
                f"High processing time: {metrics.processing_time:.3f}s "
                f"for model {metrics.model_type}"
            )
    
    def _get_monitoring_window_hours(self) -> float:
        """Get the time window covered by current monitoring data."""
        if not self.predictions:
            return 0
        
        oldest_timestamp = min(p.timestamp for p in self.predictions)
        newest_timestamp = max(p.timestamp for p in self.predictions)
        
        return (newest_timestamp - oldest_timestamp) / 3600


class DriftDetector:
    """Detect model drift using statistical methods."""
    
    def __init__(self, window_size: int = 1000, reference_size: int = 1000):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of current window for drift detection
            reference_size: Size of reference window
        """
        self.window_size = window_size
        self.reference_size = reference_size
        self.reference_data = defaultdict(lambda: deque(maxlen=reference_size))
        self.current_data = defaultdict(lambda: deque(maxlen=window_size))
        self.drift_scores = defaultdict(list)
        
        logger.info("DriftDetector initialized")
    
    def add_prediction(self, confidence: float, model_type: str):
        """Add a new prediction for drift monitoring."""
        # Add to current window
        self.current_data[model_type].append(confidence)
        
        # If we have enough data, check for drift
        if len(self.current_data[model_type]) >= self.window_size:
            drift_score = self._calculate_drift_score(model_type)
            self.drift_scores[model_type].append({
                'timestamp': time.time(),
                'score': drift_score
            })
            
            # Move current window to reference (sliding window approach)
            self.reference_data[model_type].extend(
                list(self.current_data[model_type])[:self.window_size // 2]
            )
            
            # Clear half of current window
            new_current = list(self.current_data[model_type])[self.window_size // 2:]
            self.current_data[model_type].clear()
            self.current_data[model_type].extend(new_current)
    
    def _calculate_drift_score(self, model_type: str) -> float:
        """Calculate drift score using Kolmogorov-Smirnov test."""
        if len(self.reference_data[model_type]) < self.reference_size // 2:
            return 0.0  # Not enough reference data
        
        try:
            from scipy import stats
            
            reference = list(self.reference_data[model_type])
            current = list(self.current_data[model_type])
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(reference, current)
            
            # Return KS statistic as drift score (higher = more drift)
            return ks_statistic
            
        except ImportError:
            logger.warning("scipy not available for drift detection")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0
    
    def get_drift_status(self) -> Dict[str, Any]:
        """Get current drift status for all models."""
        drift_status = {}
        
        for model_type in self.drift_scores:
            recent_scores = [
                score['score'] for score in self.drift_scores[model_type][-10:]  # Last 10 scores
            ]
            
            if recent_scores:
                avg_drift = np.mean(recent_scores)
                max_drift = max(recent_scores)
                
                # Determine drift level
                if max_drift > 0.3:
                    level = "high"
                elif max_drift > 0.1:
                    level = "medium"
                else:
                    level = "low"
                
                drift_status[model_type] = {
                    'average_drift_score': avg_drift,
                    'max_drift_score': max_drift,
                    'drift_level': level,
                    'measurements_count': len(recent_scores)
                }
            else:
                drift_status[model_type] = {
                    'drift_level': 'unknown',
                    'measurements_count': 0
                }
        
        return drift_status


class PredictionLogger:
    """Log predictions for audit and analysis."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize prediction logger.
        
        Args:
            log_file: File to log predictions to (optional)
        """
        self.log_file = log_file or "logs/predictions.jsonl"
        self.ensure_log_directory()
        
        logger.info(f"PredictionLogger initialized with file: {self.log_file}")
    
    def ensure_log_directory(self):
        """Ensure log directory exists."""
        import os
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def log(self, prediction_data: Dict[str, Any]):
        """Log a prediction entry."""
        try:
            with open(self.log_file, 'a') as f:
                json.dump(prediction_data, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def get_recent_logs(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent prediction logs."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_logs = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        if entry_time >= cutoff_time:
                            recent_logs.append(entry)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except FileNotFoundError:
            logger.warning(f"Log file {self.log_file} not found")
        
        return recent_logs


class PerformanceProfiler:
    """Profile model performance and resource usage."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
    def record_memory_usage(self, model_type: str):
        """Record current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.metrics[f"{model_type}_memory"].append({
                'timestamp': time.time(),
                'memory_mb': memory_mb
            })
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
    
    def record_cpu_usage(self, model_type: str):
        """Record current CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            self.metrics[f"{model_type}_cpu"].append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent
            })
        except ImportError:
            logger.warning("psutil not available for CPU monitoring")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for metric_type, measurements in self.metrics.items():
            if measurements:
                values = [m.get('memory_mb') or m.get('cpu_percent', 0) for m in measurements]
                summary[metric_type] = {
                    'average': np.mean(values),
                    'max': max(values),
                    'min': min(values),
                    'measurements_count': len(measurements)
                }
        
        return summary