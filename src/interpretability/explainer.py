"""
Model interpretability and explanation tools using LIME and SHAP.
Provides explanations for fake news detection predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import re
from abc import ABC, abstractmethod

# Try importing LIME and SHAP with fallbacks
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """Base class for model explainers."""
    
    def __init__(self, model: Any, model_type: str):
        """
        Initialize explainer.
        
        Args:
            model: The model to explain
            model_type: Type of model (bow, tfidf, deberta, ensemble)
        """
        self.model = model
        self.model_type = model_type
    
    @abstractmethod
    def explain_prediction(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            text: Input text to explain
            **kwargs: Additional parameters
            
        Returns:
            Explanation dictionary
        """
        pass
    
    def explain_batch(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions.
        
        Args:
            texts: List of texts to explain
            **kwargs: Additional parameters
            
        Returns:
            List of explanation dictionaries
        """
        return [self.explain_prediction(text, **kwargs) for text in texts]


class LimeExplainer(BaseExplainer):
    """LIME-based explainer for text classification."""
    
    def __init__(self, model: Any, model_type: str, class_names: List[str] = None):
        """
        Initialize LIME explainer.
        
        Args:
            model: The model to explain
            model_type: Type of model
            class_names: Names of the classes
        """
        super().__init__(model, model_type)
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LimeExplainer. Install with: pip install lime")
        
        self.class_names = class_names or ['Fake', 'Real']
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            mode='classification'
        )
        
        logger.info(f"LimeExplainer initialized for {model_type} model")
    
    def explain_prediction(self, text: str, num_features: int = 10, num_samples: int = 5000) -> Dict[str, Any]:
        """
        Explain prediction using LIME.
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation dictionary
        """
        try:
            # Create prediction function for LIME
            def predict_fn(texts):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(texts)
                else:
                    # Convert predictions to probabilities
                    preds = self.model.predict(texts)
                    proba = np.zeros((len(preds), 2))
                    proba[np.arange(len(preds)), preds] = 1.0
                    return proba
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract explanation data
            explanation_list = explanation.as_list()
            explanation_map = explanation.as_map()
            
            # Get prediction
            prediction_proba = predict_fn([text])[0]
            predicted_class = np.argmax(prediction_proba)
            confidence = prediction_proba[predicted_class]
            
            # Format explanation
            feature_importance = []
            for feature, weight in explanation_list:
                feature_importance.append({
                    'feature': feature,
                    'weight': weight,
                    'impact': 'increases' if weight > 0 else 'decreases',
                    'strength': abs(weight)
                })
            
            return {
                'method': 'LIME',
                'model_type': self.model_type,
                'prediction': self.class_names[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[i]: float(prob) for i, prob in enumerate(prediction_proba)
                },
                'feature_importance': feature_importance,
                'explanation_text': self._generate_explanation_text(feature_importance, predicted_class),
                'lime_score': explanation.score,
                'intercept': float(explanation.intercept[predicted_class]) if hasattr(explanation, 'intercept') else None
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {
                'method': 'LIME',
                'error': str(e),
                'fallback_explanation': self._get_fallback_explanation(text)
            }
    
    def _generate_explanation_text(self, feature_importance: List[Dict], predicted_class: int) -> str:
        """Generate human-readable explanation text."""
        prediction_name = self.class_names[predicted_class]
        
        # Get top positive and negative features
        positive_features = [f for f in feature_importance if f['weight'] > 0][:3]
        negative_features = [f for f in feature_importance if f['weight'] < 0][:3]
        
        explanation_parts = [f"This text is predicted to be {prediction_name}."]
        
        if positive_features:
            pos_words = [f['feature'] for f in positive_features]
            explanation_parts.append(f"Words supporting this prediction: {', '.join(pos_words)}")
        
        if negative_features:
            neg_words = [f['feature'] for f in negative_features]
            explanation_parts.append(f"Words opposing this prediction: {', '.join(neg_words)}")
        
        return " ".join(explanation_parts)
    
    def _get_fallback_explanation(self, text: str) -> Dict[str, Any]:
        """Provide fallback explanation when LIME fails."""
        # Simple heuristic-based explanation
        fake_indicators = ['shocking', 'amazing', 'incredible', 'secret', 'click', 'doctors hate']
        real_indicators = ['research', 'study', 'university', 'officials', 'announced']
        
        text_lower = text.lower()
        fake_score = sum(1 for word in fake_indicators if word in text_lower)
        real_score = sum(1 for word in real_indicators if word in text_lower)
        
        return {
            'method': 'heuristic',
            'fake_indicators_found': fake_score,
            'real_indicators_found': real_score,
            'explanation': f"Found {fake_score} fake indicators and {real_score} real indicators"
        }


class ShapExplainer(BaseExplainer):
    """SHAP-based explainer for model interpretability."""
    
    def __init__(self, model: Any, model_type: str, background_texts: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The model to explain
            model_type: Type of model
            background_texts: Background texts for SHAP (optional)
        """
        super().__init__(model, model_type)
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for ShapExplainer. Install with: pip install shap")
        
        self.background_texts = background_texts or []
        self.explainer = None
        self._initialize_explainer()
        
        logger.info(f"ShapExplainer initialized for {model_type} model")
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        try:
            if self.model_type in ['bow', 'tfidf']:
                # For traditional models, use Explainer with prediction function
                def predict_fn(texts):
                    if hasattr(self.model, 'predict_proba'):
                        return self.model.predict_proba(texts)
                    else:
                        preds = self.model.predict(texts)
                        proba = np.zeros((len(preds), 2))
                        proba[np.arange(len(preds)), preds] = 1.0
                        return proba
                
                # Use a simple explainer for traditional models
                self.explainer = shap.Explainer(predict_fn, self.background_texts[:100] if self.background_texts else [""])
                
            elif self.model_type == 'deberta':
                # For transformer models, use specialized explainer
                # This would require more setup with tokenizers
                logger.warning("SHAP for transformers requires additional setup. Using fallback.")
                self.explainer = None
                
            else:
                logger.warning(f"SHAP explainer not specifically configured for {self.model_type}")
                self.explainer = None
                
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, text: str, max_evals: int = 500) -> Dict[str, Any]:
        """
        Explain prediction using SHAP.
        
        Args:
            text: Input text to explain
            max_evals: Maximum evaluations for SHAP
            
        Returns:
            SHAP explanation dictionary
        """
        if self.explainer is None:
            return self._get_fallback_explanation(text)
        
        try:
            # Generate SHAP values
            shap_values = self.explainer([text], max_evals=max_evals)
            
            # Extract values for the positive class (Real)
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0][:, 1] if len(shap_values.values[0].shape) > 1 else shap_values.values[0]
                data = shap_values.data[0]
            else:
                values = shap_values[0]
                data = text.split()
            
            # Create feature importance list
            feature_importance = []
            for i, (token, value) in enumerate(zip(data, values)):
                feature_importance.append({
                    'feature': token,
                    'shap_value': float(value),
                    'impact': 'increases' if value > 0 else 'decreases',
                    'strength': abs(float(value))
                })
            
            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: x['strength'], reverse=True)
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([text])[0]
            else:
                pred = self.model.predict([text])[0]
                proba = [1-pred, pred] if pred in [0, 1] else [0.5, 0.5]
            
            predicted_class = np.argmax(proba)
            confidence = proba[predicted_class]
            
            return {
                'method': 'SHAP',
                'model_type': self.model_type,
                'prediction': 'Real' if predicted_class == 1 else 'Fake',
                'confidence': float(confidence),
                'probabilities': {
                    'Fake': float(proba[0]),
                    'Real': float(proba[1])
                },
                'feature_importance': feature_importance[:10],  # Top 10 features
                'explanation_text': self._generate_shap_explanation(feature_importance[:5], predicted_class),
                'base_value': float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._get_fallback_explanation(text)
    
    def _generate_shap_explanation(self, feature_importance: List[Dict], predicted_class: int) -> str:
        """Generate explanation text from SHAP values."""
        prediction_name = 'Real' if predicted_class == 1 else 'Fake'
        
        # Get features that support the prediction
        supporting_features = [f for f in feature_importance if 
                             (f['shap_value'] > 0 and predicted_class == 1) or 
                             (f['shap_value'] < 0 and predicted_class == 0)]
        
        if supporting_features:
            words = [f['feature'] for f in supporting_features[:3]]
            return f"Predicted as {prediction_name}. Key supporting words: {', '.join(words)}"
        else:
            return f"Predicted as {prediction_name} based on overall text analysis."
    
    def _get_fallback_explanation(self, text: str) -> Dict[str, Any]:
        """Provide fallback explanation when SHAP fails."""
        return {
            'method': 'SHAP_fallback',
            'model_type': self.model_type,
            'error': 'SHAP explanation not available',
            'fallback_explanation': 'Unable to generate SHAP explanation for this model type'
        }


class FeatureImportanceExplainer(BaseExplainer):
    """Explainer based on model feature importance (for traditional models)."""
    
    def __init__(self, model: Any, model_type: str):
        """
        Initialize feature importance explainer.
        
        Args:
            model: The model to explain
            model_type: Type of model
        """
        super().__init__(model, model_type)
        logger.info(f"FeatureImportanceExplainer initialized for {model_type} model")
    
    def explain_prediction(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Explain prediction using model feature importance.
        
        Args:
            text: Input text to explain
            top_k: Number of top features to show
            
        Returns:
            Feature importance explanation
        """
        try:
            # Get prediction
            prediction = self.model.predict([text])[0]
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([text])[0]
            else:
                probabilities = [1-prediction, prediction]
            
            # Get model feature importance if available
            if hasattr(self.model, 'get_feature_importance'):
                importance = self.model.get_feature_importance(top_k=top_k)
                
                # Transform text using model's vectorizer
                if hasattr(self.model, 'vectorizer'):
                    text_features = self.model.vectorizer.transform([text])
                    feature_names = self.model.vectorizer.get_feature_names_out()
                    
                    # Find which features are present in this text
                    present_features = []
                    if hasattr(text_features, 'toarray'):
                        feature_values = text_features.toarray()[0]
                        for i, value in enumerate(feature_values):
                            if value > 0:
                                present_features.append({
                                    'feature': feature_names[i],
                                    'value': float(value),
                                    'in_text': True
                                })
                    
                    return {
                        'method': 'feature_importance',
                        'model_type': self.model_type,
                        'prediction': 'Real' if prediction == 1 else 'Fake',
                        'confidence': float(max(probabilities)),
                        'probabilities': {
                            'Fake': float(probabilities[0]),
                            'Real': float(probabilities[1])
                        },
                        'global_importance': {
                            'fake_indicators': importance.get('fake', [])[:5],
                            'real_indicators': importance.get('real', [])[:5]
                        },
                        'text_features': present_features[:10],
                        'explanation_text': self._generate_feature_explanation(
                            present_features[:5], importance, prediction
                        )
                    }
            
            # Fallback explanation
            return self._get_simple_explanation(text, prediction, probabilities)
            
        except Exception as e:
            logger.error(f"Feature importance explanation failed: {e}")
            return self._get_simple_explanation(text, prediction if 'prediction' in locals() else 0, [0.5, 0.5])
    
    def _generate_feature_explanation(self, text_features: List[Dict], global_importance: Dict, prediction: int) -> str:
        """Generate explanation from feature analysis."""
        prediction_name = 'Real' if prediction == 1 else 'Fake'
        
        if text_features:
            present_words = [f['feature'] for f in text_features[:3]]
            return f"Predicted as {prediction_name}. Key words found: {', '.join(present_words)}"
        else:
            return f"Predicted as {prediction_name} based on overall text pattern."
    
    def _get_simple_explanation(self, text: str, prediction: int, probabilities: List[float]) -> Dict[str, Any]:
        """Simple explanation when detailed analysis fails."""
        return {
            'method': 'simple',
            'model_type': self.model_type,
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'Fake': float(probabilities[0]),
                'Real': float(probabilities[1])
            },
            'explanation_text': f"Classified as {'Real' if prediction == 1 else 'Fake'} based on learned patterns.",
            'text_length': len(text),
            'word_count': len(text.split())
        }


class ExplainerFactory:
    """Factory for creating appropriate explainers."""
    
    @staticmethod
    def create_explainer(
        explainer_type: str,
        model: Any,
        model_type: str,
        **kwargs
    ) -> BaseExplainer:
        """
        Create an explainer instance.
        
        Args:
            explainer_type: Type of explainer ('lime', 'shap', 'feature_importance')
            model: The model to explain
            model_type: Type of model
            **kwargs: Additional arguments for explainer
            
        Returns:
            Explainer instance
        """
        if explainer_type.lower() == 'lime':
            if not LIME_AVAILABLE:
                raise ImportError("LIME not available. Install with: pip install lime")
            return LimeExplainer(model, model_type, **kwargs)
        
        elif explainer_type.lower() == 'shap':
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP not available. Install with: pip install shap")
            return ShapExplainer(model, model_type, **kwargs)
        
        elif explainer_type.lower() == 'feature_importance':
            return FeatureImportanceExplainer(model, model_type, **kwargs)
        
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    @staticmethod
    def get_available_explainers() -> List[str]:
        """Get list of available explainer types."""
        available = ['feature_importance']
        
        if LIME_AVAILABLE:
            available.append('lime')
        
        if SHAP_AVAILABLE:
            available.append('shap')
        
        return available


class MultiExplainer:
    """Combine multiple explanation methods for comprehensive analysis."""
    
    def __init__(self, model: Any, model_type: str):
        """
        Initialize multi-explainer.
        
        Args:
            model: The model to explain
            model_type: Type of model
        """
        self.model = model
        self.model_type = model_type
        self.explainers = {}
        
        # Initialize available explainers
        available_types = ExplainerFactory.get_available_explainers()
        for explainer_type in available_types:
            try:
                self.explainers[explainer_type] = ExplainerFactory.create_explainer(
                    explainer_type, model, model_type
                )
            except Exception as e:
                logger.warning(f"Could not initialize {explainer_type} explainer: {e}")
        
        logger.info(f"MultiExplainer initialized with: {list(self.explainers.keys())}")
    
    def explain_prediction(self, text: str, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation using multiple methods.
        
        Args:
            text: Input text to explain
            methods: Specific methods to use (if None, use all available)
            
        Returns:
            Combined explanation
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        explanations = {}
        
        for method in methods:
            if method in self.explainers:
                try:
                    explanations[method] = self.explainers[method].explain_prediction(text)
                except Exception as e:
                    logger.error(f"Explanation failed for {method}: {e}")
                    explanations[method] = {'error': str(e)}
        
        # Combine explanations
        combined = self._combine_explanations(explanations, text)
        return combined
    
    def _combine_explanations(self, explanations: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Combine multiple explanations into a unified view."""
        # Get consensus prediction
        predictions = []
        confidences = []
        
        for method, exp in explanations.items():
            if 'error' not in exp:
                predictions.append(exp.get('prediction'))
                confidences.append(exp.get('confidence', 0))
        
        consensus_prediction = max(set(predictions), key=predictions.count) if predictions else 'Unknown'
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Aggregate feature importance
        all_features = {}
        for method, exp in explanations.items():
            if 'error' not in exp and 'feature_importance' in exp:
                for feature in exp['feature_importance']:
                    feature_name = feature.get('feature', '')
                    if feature_name:
                        if feature_name not in all_features:
                            all_features[feature_name] = []
                        all_features[feature_name].append({
                            'method': method,
                            'weight': feature.get('weight', feature.get('shap_value', 0))
                        })
        
        # Create consolidated feature importance
        consolidated_features = []
        for feature_name, method_weights in all_features.items():
            avg_weight = np.mean([w['weight'] for w in method_weights])
            consolidated_features.append({
                'feature': feature_name,
                'average_weight': avg_weight,
                'methods_agreement': len(method_weights),
                'impact': 'increases' if avg_weight > 0 else 'decreases'
            })
        
        consolidated_features.sort(key=lambda x: abs(x['average_weight']), reverse=True)
        
        return {
            'method': 'multi_explainer',
            'model_type': self.model_type,
            'consensus_prediction': consensus_prediction,
            'average_confidence': avg_confidence,
            'individual_explanations': explanations,
            'consolidated_features': consolidated_features[:10],
            'explanation_summary': self._generate_summary(
                consensus_prediction, consolidated_features[:3], explanations
            ),
            'agreement_score': self._calculate_agreement_score(predictions),
            'text_stats': {
                'length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
            }
        }
    
    def _generate_summary(self, prediction: str, top_features: List[Dict], explanations: Dict) -> str:
        """Generate a human-readable summary of all explanations."""
        summary_parts = [f"Consensus prediction: {prediction}"]
        
        if top_features:
            feature_names = [f['feature'] for f in top_features[:3]]
            summary_parts.append(f"Most important words: {', '.join(feature_names)}")
        
        method_count = len([exp for exp in explanations.values() if 'error' not in exp])
        summary_parts.append(f"Analysis based on {method_count} explanation methods")
        
        return ". ".join(summary_parts) + "."
    
    def _calculate_agreement_score(self, predictions: List[str]) -> float:
        """Calculate how much the different methods agree."""
        if not predictions:
            return 0.0
        
        most_common_count = predictions.count(max(set(predictions), key=predictions.count))
        return most_common_count / len(predictions)