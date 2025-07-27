"""
FastAPI deployment server for fake news detection.
Provides REST endpoints for model inference with production features.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
import logging
import time
import asyncio
import json
from datetime import datetime
import os
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipeline import FakeNewsDetector, PipelineConfig
from src.deployment.monitoring import ModelMonitor, PredictionLogger
from src.deployment.cache import PredictionCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="Production-ready API for detecting fake news using ensemble ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and monitoring
detector: Optional[FakeNewsDetector] = None
monitor: Optional[ModelMonitor] = None
cache: Optional[PredictionCache] = None
prediction_logger: Optional[PredictionLogger] = None

# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    model_type: str = Field(default="ensemble", description="Model type: bow, tfidf, deberta, ensemble")
    return_confidence: bool = Field(default=True, description="Return confidence scores")
    return_explanation: bool = Field(default=False, description="Return model explanation")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['bow', 'tfidf', 'deberta', 'ensemble']
        if v not in allowed_types:
            raise ValueError(f'model_type must be one of {allowed_types}')
        return v

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    model_type: str = Field(default="ensemble", description="Model type: bow, tfidf, deberta, ensemble")
    return_confidence: bool = Field(default=True, description="Return confidence scores")
    
    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if not text or len(text) > 10000:
                raise ValueError('Each text must be between 1 and 10000 characters')
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: str = Field(..., description="Prediction: Fake or Real")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    probability_fake: Optional[float] = Field(None, description="Probability of being fake")
    probability_real: Optional[float] = Field(None, description="Probability of being real")
    model_used: str = Field(..., description="Model type used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    explanation: Optional[Dict] = Field(None, description="Model explanation (if requested)")

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    batch_size: int = Field(..., description="Number of texts processed")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

class ModelStatsResponse(BaseModel):
    """Response model for model statistics."""
    total_predictions: int = Field(..., description="Total predictions made")
    predictions_by_model: Dict[str, int] = Field(..., description="Predictions by model type")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    uptime_seconds: float = Field(..., description="Service uptime")

# Global variables for tracking
start_time = time.time()

# Authentication dependency (optional)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication check (expand for production)."""
    if credentials is None:
        return None  # Allow unauthenticated access for demo
    
    # In production, verify JWT token here
    # For demo, accept any token
    return {"user_id": "demo_user"}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global detector, monitor, cache, prediction_logger
    
    logger.info("Starting Fake News Detection API...")
    
    try:
        # Initialize model detector
        config_path = os.getenv("CONFIG_PATH", "configs/default_config.yaml")
        if os.path.exists(config_path):
            detector = FakeNewsDetector.from_config_file(config_path)
        else:
            detector = FakeNewsDetector()
        
        # Try to load pre-trained models
        try:
            detector.load_models()
            logger.info("Pre-trained models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Models will be trained on-demand")
        
        # Initialize monitoring and caching
        monitor = ModelMonitor()
        cache = PredictionCache(max_size=1000, ttl_seconds=3600)
        prediction_logger = PredictionLogger()
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_status = {
        "bow": hasattr(detector, 'traditional_models') and 'bow' in detector.traditional_models,
        "tfidf": hasattr(detector, 'traditional_models') and 'tfidf' in detector.traditional_models,
        "deberta": hasattr(detector, 'transformer_model') and detector.transformer_model is not None,
        "ensemble": hasattr(detector, 'ensemble_model') and detector.ensemble_model is not None
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=models_status,
        uptime_seconds=time.time() - start_time
    )

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_text(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Predict if a single text is fake or real."""
    prediction_start_time = time.time()
    
    try:
        # Check cache first
        cache_key = cache.get_cache_key(request.text, request.model_type)
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached prediction")
            return cached_result
        
        # Ensure models are available
        await ensure_models_loaded(request.model_type)
        
        # Make prediction
        results = detector.predict([request.text], model_type=request.model_type)
        
        # Extract results
        prediction = results['labels'][0]
        probabilities = results['probabilities'][0]
        confidence = results['confidence'][0]
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            confidence=confidence if request.return_confidence else None,
            probability_fake=float(probabilities[0]) if request.return_confidence else None,
            probability_real=float(probabilities[1]) if request.return_confidence else None,
            model_used=request.model_type,
            processing_time_ms=(time.time() - prediction_start_time) * 1000
        )
        
        # Add explanation if requested
        if request.return_explanation:
            response.explanation = await get_prediction_explanation(
                request.text, request.model_type, prediction
            )
        
        # Cache the result
        cache.set(cache_key, response)
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction, request.text, prediction, confidence, request.model_type, user
        )
        
        # Update monitoring
        monitor.record_prediction(request.model_type, confidence, time.time() - prediction_start_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Predict if multiple texts are fake or real."""    
    batch_start_time = time.time()
    
    try:
        # Ensure models are available
        await ensure_models_loaded(request.model_type)
        
        # Make batch prediction
        results = detector.predict(request.texts, model_type=request.model_type)
        
        # Create individual responses
        predictions = []
        for i, text in enumerate(request.texts):
            prediction = results['labels'][i]
            probabilities = results['probabilities'][i]
            confidence = results['confidence'][i]
            
            pred_response = PredictionResponse(
                prediction=prediction,
                confidence=confidence if request.return_confidence else None,
                probability_fake=float(probabilities[0]) if request.return_confidence else None,
                probability_real=float(probabilities[1]) if request.return_confidence else None,
                model_used=request.model_type,
                processing_time_ms=0  # Set per-item time to 0 for batch
            )
            predictions.append(pred_response)
        
        total_time = (time.time() - batch_start_time) * 1000
        
        # Log batch prediction in background
        background_tasks.add_task(
            log_batch_prediction, request.texts, results['labels'], 
            results['confidence'], request.model_type, user
        )
        
        # Update monitoring
        for confidence in results['confidence']:
            monitor.record_prediction(request.model_type, confidence, total_time / len(request.texts))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time,
            batch_size=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Model statistics endpoint
@app.get("/stats", response_model=ModelStatsResponse)
async def get_model_stats(user=Depends(get_current_user)):
    """Get model usage statistics."""
    stats = monitor.get_stats()
    
    return ModelStatsResponse(
        total_predictions=stats['total_predictions'],
        predictions_by_model=stats['predictions_by_model'],
        average_processing_time_ms=stats['avg_processing_time'] * 1000,
        uptime_seconds=time.time() - start_time
    )

# Model management endpoints
@app.post("/models/{model_type}/train")
async def train_model(
    model_type: str,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Train a specific model type."""
    if model_type not in ['bow', 'tfidf', 'ensemble']:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    # Add training task to background
    background_tasks.add_task(train_model_background, model_type)
    
    return {"message": f"Training {model_type} model started in background", "status": "started"}

@app.get("/models")
async def list_models():
    """List available models and their status."""
    
    models_status = {
        "bow": {
            "available": hasattr(detector, 'traditional_models') and 'bow' in detector.traditional_models,
            "type": "traditional"
        },
        "tfidf": {
            "available": hasattr(detector, 'traditional_models') and 'tfidf' in detector.traditional_models,
            "type": "traditional"
        },
        "deberta": {
            "available": hasattr(detector, 'transformer_model') and detector.transformer_model is not None,
            "type": "transformer"
        },
        "ensemble": {
            "available": hasattr(detector, 'ensemble_model') and detector.ensemble_model is not None,
            "type": "ensemble"
        }
    }
    
    return {"models": models_status}

# Helper functions
async def ensure_models_loaded(model_type: str):
    """Ensure required models are loaded."""
    
    if model_type in ['bow', 'tfidf']:
        if not hasattr(detector, 'traditional_models') or model_type not in detector.traditional_models:
            # Try to load datasets and train
            datasets = detector.load_datasets()
            if datasets:
                detector.train_traditional_models()
            else:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Model {model_type} not available and cannot be trained without datasets"
                )
    
    elif model_type == 'ensemble':
        if not hasattr(detector, 'ensemble_model') or detector.ensemble_model is None:
            await ensure_models_loaded('bow')  # Ensure base models exist
            await ensure_models_loaded('tfidf')
            detector.create_ensemble()
    
    elif model_type == 'deberta':
        if not hasattr(detector, 'transformer_model') or detector.transformer_model is None:
            raise HTTPException(
                status_code=503,
                detail="DeBERTa model not available. Please train the model first."
            )

async def get_prediction_explanation(text: str, model_type: str, prediction: str) -> Dict:
    """Get explanation for prediction (placeholder for future LIME/SHAP integration)."""
    return {
        "method": "feature_importance",
        "explanation": f"Prediction '{prediction}' based on {model_type} model analysis",
        "top_features": ["feature1", "feature2", "feature3"],  # Placeholder
        "confidence_factors": {
            "text_length": len(text),
            "model_certainty": "high"
        }
    }

async def log_prediction(text: str, prediction: str, confidence: float, model_type: str, user: Optional[Dict]):
    """Log prediction for monitoring."""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text_hash": hash(text) % 1000000,  # Don't store actual text for privacy
        "prediction": prediction,
        "confidence": confidence,
        "model_type": model_type,
        "user_id": user.get("user_id") if user else None
    }
    
    prediction_logger.log(log_entry)

async def log_batch_prediction(texts: List[str], predictions: List[str], confidences: List[float], model_type: str, user: Optional[Dict]):
    """Log batch prediction for monitoring."""
    for text, prediction, confidence in zip(texts, predictions, confidences):
        await log_prediction(text, prediction, confidence, model_type, user)

async def train_model_background(model_type: str):
    """Train model in background."""
    
    try:
        logger.info(f"Starting background training for {model_type}")
        
        # Load datasets
        datasets = detector.load_datasets()
        if not datasets:
            logger.error("No datasets available for training")
            return
        
        if model_type in ['bow', 'tfidf']:
            detector.train_traditional_models()
        elif model_type == 'ensemble':
            detector.create_ensemble()
        
        logger.info(f"Background training completed for {model_type}")
        
    except Exception as e:
        logger.error(f"Background training failed for {model_type}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )