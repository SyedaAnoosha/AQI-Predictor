import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from dotenv import load_dotenv
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

from backend.services import (
    load_model_artifacts, generate_forecast, check_alerts,
    get_feature_importance, get_historical_data, get_model_metrics,
    get_current_aqi, get_shap_values
)
from backend.hopsworks_client import (
    connect_hopsworks, get_all_model_metrics
)
from backend.schemas import (
    PredictionResponse, AlertResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

LATITUDE = 25.3792
LONGITUDE = 68.3683
TIMEZONE = 'Asia/Karachi'
HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
MODEL_CACHE: Dict[str, Any] = {}

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'cache'))
METRICS_CACHE_PATH = os.path.join(CACHE_DIR, 'all_model_metrics.json')
BEST_MODEL_META_PATH = os.path.join(CACHE_DIR, 'best_model_meta.json')

def _normalize_model_key(name: str) -> str:
    return (name or '').strip().lower().replace(' ', '_')

def _load_best_model_name() -> str:
    try:
        if os.path.exists(BEST_MODEL_META_PATH):
            with open(BEST_MODEL_META_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            best = data.get('best_model') if isinstance(data, dict) else None
            best_key = _normalize_model_key(best)
            if best_key:
                return best_key
    except Exception:
        pass
    return ""

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', '').strip().lower()
DEFAULT_MODEL = DEFAULT_MODEL if DEFAULT_MODEL else _load_best_model_name() or 'lightgbm'
AVAILABLE_MODELS: List[str] = [DEFAULT_MODEL]

def _load_and_cache_model(model_name: str) -> Any:
    normalized = _normalize_model_key(model_name)
    artifacts = load_model_artifacts(api_key=HOPSWORKS_API_KEY, model_name=normalized)
    if artifacts:
        MODEL_CACHE[normalized] = artifacts
    return artifacts

_load_and_cache_model(DEFAULT_MODEL)

model_artifacts = MODEL_CACHE.get(DEFAULT_MODEL)

def _get_model(model_name: str) -> Any:
    requested = _normalize_model_key(model_name) or DEFAULT_MODEL
    artifacts = MODEL_CACHE.get(requested)
    if artifacts is None:
        artifacts = _load_and_cache_model(requested)
    if artifacts is None and requested != DEFAULT_MODEL:
        artifacts = MODEL_CACHE.get(DEFAULT_MODEL) or _load_and_cache_model(DEFAULT_MODEL)
    if artifacts is None:
        raise HTTPException(status_code=503, detail=f"Model '{requested}' not available")
    return artifacts

def _load_cached_metrics() -> Dict[str, Any]:
    try:
        if os.path.exists(METRICS_CACHE_PATH):
            with open(METRICS_CACHE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                normalized = {}
                for name, metrics in data.items():
                    key = _normalize_model_key(name)
                    if key:
                        normalized[key] = metrics
                return normalized or data
    except Exception:
        pass
    return {}

@router.get(
    "/predictions/next-3-days",
    response_model=PredictionResponse,
    summary="Get 3-day AQI forecast",
    tags=["Predictions"]
)
async def get_3day_forecast(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)
    
    response = generate_forecast(
        artifacts,
        hours=72,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        timezone=TIMEZONE
    )
    return response

@router.get(
    "/predictions/next-24-hours",
    response_model=PredictionResponse,
    summary="Get 24-hour AQI forecast",
    tags=["Predictions"]
)
async def get_24hour_forecast(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)
    response = generate_forecast(
        artifacts,
        hours=24,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        timezone=TIMEZONE
    )
    return response

@router.get(
    "/alerts",
    response_model=AlertResponse,
    summary="Get AQI alerts",
    tags=["Alerts"]
)
async def get_alerts(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)
    
    try:
        current_data = get_current_aqi(
            artifacts,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            timezone=TIMEZONE
        )
        
        forecast = generate_forecast(
            artifacts,
            hours=72,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            timezone=TIMEZONE
        )
        
        alert = check_alerts(forecast.predictions, current_aqi=current_data['aqi'])
        return alert
        
    except Exception as e:
        logger.error(f"Error generating alert: {str(e)}")
        from backend.schemas import AlertResponse
        return AlertResponse(
            alert_level="Moderate",
            current_aqi=100.0,
            peak_aqi_predicted=120.0,
            peak_time=datetime.now(),
            recommendation="Avoid prolonged outdoor activities",
            affected_groups=["children", "elderly", "asthma patients"]
        )

@router.get(
    "/feature-importance",
    summary="Get feature importance",
    tags=["Explainability"]
)
async def get_feature_importance_endpoint(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)
    
    try:
        importance_dict = get_feature_importance(artifacts)
        top_features = list(importance_dict.items())[:20]
        
        response = {
            "model_name": model,
            "model_version": 1,
            "generated_at": datetime.now(),
            "top_features": [
                {
                    "feature_name": name,
                    "importance_score": float(score)
                }
                for name, score in top_features
            ]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feature importance: {str(e)}")

@router.get(
    "/shap-values",
    summary="Get SHAP values for model explainability",
    tags=["Explainability"]
)
async def get_shap_values_endpoint(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)
    
    try:
        shap_data = get_shap_values(artifacts)
        
        response = {
            "model_name": model,
            "model_version": 1,
            "generated_at": datetime.now(),
            "method": shap_data.get("method", "SHAP"),
            "base_value": shap_data.get("base_value", 0),
            "top_features": shap_data.get("top_features", [])
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving SHAP values: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve SHAP values: {str(e)}")


@router.get(
    "/historical",
    summary="Get historical AQI data",
    tags=["Historical Data"]
)
async def get_historical_data_endpoint(days: int = 7):
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
    
    try:
        historical_df = get_historical_data(
            days=days,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            timezone=TIMEZONE
        )

        if 'time' in historical_df.columns:
            daily_df = historical_df.copy()
            daily_df['date'] = pd.to_datetime(daily_df['time']).dt.date
            daily_avg = (
                daily_df.groupby('date')['aqi']
                .mean()
                .reset_index()
                .rename(columns={'aqi': 'avg_aqi'})
            )
            daily_avg_records = daily_avg.to_dict('records')
        else:
            daily_avg_records = []

        response = {
            "days": days,
            "data_points": daily_avg_records,
            "average_aqi": float(historical_df['aqi'].mean()) if 'aqi' in historical_df.columns else 0,
            "max_aqi": float(historical_df['aqi'].max()) if 'aqi' in historical_df.columns else 0,
            "min_aqi": float(historical_df['aqi'].min()) if 'aqi' in historical_df.columns else 0,
            "total_records": len(historical_df)
        }

        return response

    except Exception as e:
        logger.error(f"Error retrieving historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical data: {str(e)}")


@router.get(
    "/health-guide",
    summary="Get health recommendations",
    tags=["Health Guidance"]
)
async def get_health_guide(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)
    
    try:
        current_data = get_current_aqi(
            artifacts,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            timezone=TIMEZONE
        )
        
        current_aqi = current_data['aqi']
        
        if current_aqi > 300:
            outdoor = "Avoid all outdoor activities"
            indoor = "Stay indoors with air purifiers running"
        elif current_aqi > 200:
            outdoor = "Avoid outdoor activities"
            indoor = "Keep indoors with good ventilation"
        elif current_aqi > 150:
            outdoor = "Limit outdoor activities"
            indoor = "Indoor activities are safe"
        elif current_aqi > 100:
            outdoor = "Sensitive groups should limit outdoor activities"
            indoor = "All indoor activities are safe"
        else:
            outdoor = "Outdoor activities are safe for all"
            indoor = "Indoor activities are safe"
        
        response = {
            "current_aqi": current_aqi,
            "category": current_data['category'],
            "recommendations": current_data['health_recommendation'],
            "outdoor_activity": outdoor,
            "indoor_activity": indoor,
            "timestamp": current_data['timestamp']
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving health guide: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve health guide: {str(e)}")


@router.get(
    "/model-metrics",
    summary="Get model performance metrics",
    tags=["Model Info"]
)
async def get_model_metrics_endpoint(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)

    try:
        cached = _load_cached_metrics()
        model_key = _normalize_model_key(model)
        metrics = cached.get(model_key) or get_model_metrics(artifacts)

        response = {
            "model_name": model_key or model,
            "model_version": 1,
            "evaluation_date": datetime.now(),
            "metrics": metrics
        }

        return response

    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model metrics: {str(e)}")


@router.get(
    "/model-metrics/all",
    summary="Get metrics for all loaded models",
    tags=["Model Info"]
)
async def get_all_model_metrics_endpoint():
    try:
        if not HOPSWORKS_API_KEY:
            logger.warning("Hopsworks API key not set, falling back to cache")
            cached = _load_cached_metrics()
            metrics_payload = cached if cached else {}
        else:
            try:
                project, fs = connect_hopsworks(HOPSWORKS_API_KEY)
                mr = project.get_model_registry()
                metrics_payload = get_all_model_metrics(mr)
                
                if not metrics_payload:
                    metrics_payload = _load_cached_metrics()
            except Exception as e:
                logger.warning(f"Could not fetch from Hopsworks: {e}, falling back to cache")
                metrics_payload = _load_cached_metrics()
        
        available = list(metrics_payload.keys()) if metrics_payload else AVAILABLE_MODELS

        return {
            "default_model": DEFAULT_MODEL,
            "available_models": available,
            "metrics": metrics_payload
        }
    except Exception as e:
        logger.error(f"Error fetching all model metrics: {str(e)}")
        cached = _load_cached_metrics()
        return {
            "default_model": DEFAULT_MODEL,
            "available_models": list(cached.keys()) if cached else AVAILABLE_MODELS,
            "metrics": cached
        }

@router.get(
    "/models",
    summary="List available models",
    tags=["Model Info"]
)
async def list_models():
    cached = _load_cached_metrics()
    available = list(cached.keys()) if cached else (list(MODEL_CACHE.keys()) if MODEL_CACHE else [DEFAULT_MODEL])
    return {
        "default_model": DEFAULT_MODEL,
        "available_models": available,
        "loaded_models": list(MODEL_CACHE.keys())
    }

@router.get(
    "/current-aqi",
    summary="Get current AQI",
    tags=["Current Data"]
)
async def get_current_aqi_endpoint(model: str = Query(DEFAULT_MODEL, description="Model name to use")):
    artifacts = _get_model(model)
    
    try:
        current_data = get_current_aqi(
            artifacts,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            timezone=TIMEZONE
        )
        
        response = {
            "location": "Hyderabad, Sindh",
            "timestamp": current_data['timestamp'],
            "aqi": current_data['aqi'],
            "category": current_data['category'],
            "health_recommendation": current_data['health_recommendation'],
            # Add pollutant data
            "pm2_5": current_data.get('pm2_5', 0),
            "pm10": current_data.get('pm10', 0),
            "nitrogen_dioxide": current_data.get('nitrogen_dioxide', 0),
            "sulphur_dioxide": current_data.get('sulphur_dioxide', 0),
            "carbon_monoxide": current_data.get('carbon_monoxide', 0),
            # Add weather data
            "temperature": current_data.get('temperature', 0),
            "humidity": current_data.get('humidity', 0),
            "wind_speed": current_data.get('wind_speed', 0),
            "wind_direction": current_data.get('wind_direction', 0),
            "pressure": current_data.get('pressure', 0),
            "precipitation": current_data.get('precipitation', 0)
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving current AQI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve current AQI: {str(e)}")