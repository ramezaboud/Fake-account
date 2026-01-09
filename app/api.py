"""
FastAPI application for Fake Account Detection.

This API provides endpoints for predicting whether social media accounts
are fake or genuine using a trained Random Forest model.

Run with: uvicorn app.api:app --reload --port 8000
"""

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schemas import (
    UserProfile,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fake Account Detection API",
    description="API for detecting fake/bot accounts on social media platforms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths: resolve relative to the repository root (parent of `app`)
BASE_DIR = Path(__file__).parent.parent.resolve()
MODEL_PATHS = [
    BASE_DIR / "models" / "randomforest_pipeline.joblib",
    BASE_DIR / "models" / "randomforest_best_model.joblib",
]

# Global model variable
model = None


def load_model():
    """Load the trained model."""
    global model
    for path in MODEL_PATHS:
        if path.exists():
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
    raise FileNotFoundError(f"No model found. Expected one of: {MODEL_PATHS}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except FileNotFoundError as e:
        logger.warning(f"Model not loaded: {e}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API info."""
    return {
        "name": "Fake Account Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version="1.0.0"
    )


def user_to_dataframe(user: UserProfile) -> pd.DataFrame:
    """Convert UserProfile to DataFrame for prediction."""
    return pd.DataFrame([{
        "name": user.name or "",
        "screen_name": user.screen_name,
        "statuses_count": user.statuses_count,
        "followers_count": user.followers_count,
        "friends_count": user.friends_count,
        "favourites_count": user.favourites_count,
        "listed_count": user.listed_count,
        "created_at": user.created_at,
        "description": user.description or "",
        "lang": user.lang or "en",
        "default_profile": int(user.default_profile),
        "verified": int(user.verified),
    }])


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, decision_threshold: float = Query(0.445, ge=0.0, le=1.0, description="Decision threshold for classifying as fake (probability >= threshold)")):
    """Predict if a single user account is fake or genuine."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        df = user_to_dataframe(request.user)
        probas = model.predict_proba(df)[0]

        # Determine prediction using the provided decision threshold on p(fake)
        p_fake = float(probas[1])
        prediction = 1 if p_fake >= decision_threshold else 0

        # Note: In the dataset, label 0 = genuine/real, label 1 = fake
        label = "fake" if prediction == 1 else "genuine"
        confidence = float(max(probas))

        return PredictionResponse(
            prediction=int(prediction),
            label=label,
            confidence=confidence,
            probabilities={
                "genuine": float(probas[0]),  # probability of class 0 (genuine)
                "fake": float(probas[1])      # probability of class 1 (fake)
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest, decision_threshold: float = Query(0.445, ge=0.0, le=1.0, description="Decision threshold for classifying as fake (probability >= threshold)")):
    """Predict if multiple user accounts are fake or genuine."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        predictions = []
        
        for user in request.users:
            df = user_to_dataframe(user)
            probas = model.predict_proba(df)[0]

            # use decision_threshold on p(fake)
            p_fake = float(probas[1])
            pred = 1 if p_fake >= decision_threshold else 0

            label = "fake" if pred == 1 else "genuine"
            confidence = float(max(probas))

            predictions.append(PredictionResponse(
                prediction=int(pred),
                label=label,
                confidence=confidence,
                probabilities={
                    "genuine": float(probas[0]),
                    "fake": float(probas[1])
                }
            ))
        
        fake_count = sum(1 for p in predictions if p.prediction == 1)
        genuine_count = len(predictions) - fake_count
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            fake_count=fake_count,
            genuine_count=genuine_count
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    info = {
        "type": type(model).__name__,
        "steps": [],
    }
    
    if hasattr(model, "steps"):
        info["steps"] = [
            {"name": name, "type": type(step).__name__}
            for name, step in model.steps
        ]
    
    if hasattr(model, "feature_names_in_"):
        info["features"] = list(model.feature_names_in_)
    
    return info
