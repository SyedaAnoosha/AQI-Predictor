from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict


class WeatherDataInput(BaseModel):
    time: datetime
    temperature_2m: float = Field(..., description="Temperature at 2m height (°C)")
    relative_humidity_2m: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    pressure_msl: float = Field(..., description="Mean sea level pressure (hPa)")
    wind_speed_10m: float = Field(..., ge=0, description="Wind speed at 10m height (km/h)")
    wind_direction_10m: float = Field(..., ge=0, le=360, description="Wind direction (degrees)")
    precipitation: float = Field(..., ge=0, description="Precipitation (mm)")


class AirQualityDataInput(BaseModel):
    time: datetime
    pm10: float = Field(..., ge=0, description="PM10 concentration (μg/m³)")
    pm2_5: float = Field(..., ge=0, description="PM2.5 concentration (μg/m³)")
    nitrogen_dioxide: Optional[float] = Field(None, ge=0, description="NO2 concentration")
    sulphur_dioxide: Optional[float] = Field(None, ge=0, description="SO2 concentration")
    ozone: float = Field(..., ge=0, description="O3 concentration")
    carbon_monoxide: float = Field(..., ge=0, description="CO concentration")
    aqi: Optional[float] = Field(None, ge=0, le=500, description="US AQI (may not be present in forecasts)")


class ModelFeatureVector(BaseModel):
    temperature_2m: float
    relative_humidity_2m: float
    pressure_msl: float
    wind_speed_10m: float
    wind_direction_10m: float
    precipitation: float
    ozone: float
    
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    day_of_month: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    quarter: int = Field(..., ge=1, le=4)
    week_of_year: int = Field(..., ge=1, le=53)
    is_weekend: int = Field(..., ge=0, le=1)
    is_daytime: int = Field(..., ge=0, le=1)
    hour_sin: float
    hour_cos: float
    day_of_week_sin: float
    day_of_week_cos: float
    month_sin: float
    month_cos: float
    
    pm2_5_lag_1h: float
    pm2_5_lag_3h: float
    pm2_5_lag_6h: float
    pm2_5_lag_12h: float
    pm2_5_lag_24h: float
    carbon_monoxide_lag_1h: float
    carbon_monoxide_lag_3h: float
    carbon_monoxide_lag_6h: float
    carbon_monoxide_lag_12h: float
    carbon_monoxide_lag_24h: float
    
    aqi_change_1h: float
    aqi_change_3h: float
    aqi_change_6h: float
    aqi_change_24h: float
    aqi_rate_1h: float
    aqi_rate_3h: float
    aqi_rate_24h: float
    
    temp_humidity_interaction: float
    wind_pressure_interaction: float
    
    season_0: int = Field(..., ge=0, le=1, description="Winter")
    season_1: int = Field(..., ge=0, le=1, description="Spring")
    season_2: int = Field(..., ge=0, le=1, description="Summer")
    season_3: int = Field(..., ge=0, le=1, description="Fall")


class ModelMetadata(BaseModel):
    model_name: str
    model_version: int
    framework: str = Field(..., description="e.g., LightGBM, XGBoost, TensorFlow")
    feature_count: int
    feature_names: List[str]
    trained_date: datetime
    metrics: Dict[str, float] = Field(..., description="RMSE, MAE, R² scores")

class ModelArtifacts(BaseModel):
    metadata: ModelMetadata
    model_loaded: bool
    scaler_loaded: bool
    features_validated: bool

class PredictionItem(BaseModel):
    timestamp: datetime = Field(..., description="Prediction timestamp")
    predicted_aqi: float = Field(..., ge=0, le=500, description="Predicted US AQI value (0-500)")
    aqi_category: str = Field(..., description="AQI category (Good, Moderate, Unhealthy, etc.)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    
    weather_context: Optional[Dict[str, float]] = Field(
        None,
        description="Weather conditions at prediction time",
        example={"temperature": 25.3, "humidity": 65.0, "wind_speed": 12.5}
    )
    trend: Optional[str] = Field(
        None,
        description="AQI trend: improving, worsening, stable"
    )
    
class PredictionResponse(BaseModel):
    location: str = Field("Hyderabad, Sindh", description="Prediction location")
    latitude: float = Field(25.3548, description="Location latitude")
    longitude: float = Field(68.3711, description="Location longitude")
    generated_at: datetime = Field(..., description="Timestamp when prediction was generated")
    predictions: List[PredictionItem] = Field(..., description="List of hourly predictions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": "Hyderabad, Sindh",
                "latitude": 25.3548,
                "longitude": 68.3711,
                "generated_at": "2026-01-23T14:30:00Z",
                "predictions": [
                    {
                        "timestamp": "2026-01-23T15:00:00Z",
                        "predicted_aqi": 75,
                        "aqi_category": "Moderate",
                        "confidence": 0.92
                    }
                ]
            }
        }


class AlertResponse(BaseModel):
    alert_level: str = Field(..., description="Alert level: normal, caution, warning, hazard")
    current_aqi: float = Field(..., description="Current measured US AQI")
    peak_aqi_predicted: float = Field(..., description="Highest predicted AQI in next 72 hours")
    peak_time: datetime = Field(..., description="Time of peak AQI")
    recommendation: str = Field(..., description="Health recommendation")
    affected_groups: List[str] = Field(..., description="Sensitive groups affected")


class FeatureImportanceItem(BaseModel):
    feature_name: str = Field(..., description="Name of the feature")
    importance_score: float = Field(..., ge=0, le=1, description="Importance score (0-1)")
    contribution: str = Field(..., description="How feature affects prediction")

class FeatureImportanceResponse(BaseModel):
    model_name: str = Field(..., description="Name of the trained model")
    model_version: int = Field(..., description="Model version number")
    generated_at: datetime = Field(..., description="When analysis was generated")
    top_features: List[FeatureImportanceItem] = Field(..., description="Top 10 important features")
    explanation: str = Field(..., description="General explanation of feature importance")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "xgboost",
                "model_version": 1,
                "generated_at": "2026-01-23T14:30:00Z",
                "top_features": [
                    {
                        "feature_name": "pm2_5",
                        "importance_score": 0.35,
                        "contribution": "PM2.5 is the strongest predictor of AQI"
                    }
                ],
                "explanation": "These features have the strongest influence on AQI predictions"
            }
        }


class HistoricalDataItem(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    measured_aqi: float = Field(..., description="Measured US AQI value")
    temperature: float = Field(..., description="Temperature (°C)")
    humidity: float = Field(..., description="Relative humidity (%)")
    pressure: Optional[float] = Field(None, description="Pressure (hPa)")
    wind_speed: float = Field(..., description="Wind speed (km/h)")
    wind_direction: Optional[float] = Field(None, description="Wind direction (degrees)")
    precipitation: Optional[float] = Field(None, description="Precipitation (mm)")
    
    pm2_5: Optional[float] = Field(None, description="PM2.5 concentration (μg/m³)")
    pm10: Optional[float] = Field(None, description="PM10 concentration (μg/m³)")
    ozone: Optional[float] = Field(None, description="Ozone concentration")
    carbon_monoxide: Optional[float] = Field(None, description="CO concentration")
    
    aqi_category: Optional[str] = Field(None, description="AQI category")
    is_weekend: Optional[bool] = Field(None, description="Is weekend day")
    hour_of_day: Optional[int] = Field(None, ge=0, le=23, description="Hour of day")

class HistoricalDataResponse(BaseModel):
    location: str = Field("Hyderabad, Sindh", description="Location")
    period: str = Field(..., description="Time period (e.g., 'last_30_days')")
    data_points: List[HistoricalDataItem] = Field(..., description="Historical records")
    average_aqi: float = Field(..., description="Average AQI for period")
    max_aqi: float = Field(..., description="Maximum AQI in period")
    min_aqi: float = Field(..., description="Minimum AQI in period")

class HealthRecommendation(BaseModel):
    aqi_range: str = Field(..., description="AQI range (e.g., '0-50')")
    category: str = Field(..., description="AQI category")
    description: str = Field(..., description="Category description")
    general_public: str = Field(..., description="Recommendation for general public")
    sensitive_groups: str = Field(..., description="Recommendation for sensitive groups")
    children_elderly: str = Field(..., description="Specific advice for children and elderly")

class HealthGuideResponse(BaseModel):
    current_aqi: float = Field(..., description="Current AQI value")
    recommendations: HealthRecommendation = Field(..., description="Health recommendation")
    outdoor_activity: str = Field(..., description="Outdoor activity advisory")
    indoor_activity: str = Field(..., description="Indoor activity recommendation")

class ModelMetricsResponse(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    model_version: int = Field(..., description="Model version")
    evaluation_date: datetime = Field(..., description="When model was evaluated")
    rmse: float = Field(..., ge=0, description="Root Mean Squared Error")
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    r2_score: float = Field(..., ge=-1, le=1, description="R² Score (-1 to 1)")
    test_samples: int = Field(..., ge=0, description="Number of test samples")

class HealthStatusResponse(BaseModel):
    status: str = Field("healthy", description="API status")
    timestamp: datetime = Field(..., description="Current server time")
    model_loaded: bool = Field(..., description="Whether prediction model is loaded")
    feature_store_available: bool = Field(..., description="Whether Feature Store is accessible")
    version: str = Field("1.0.0", description="API version")

AQI_CATEGORIES = {
    "good": {"range": (0, 50), "color": "green"},
    "moderate": {"range": (51, 100), "color": "yellow"},
    "unhealthy_for_sensitive": {"range": (101, 150), "color": "orange"},
    "unhealthy": {"range": (151, 200), "color": "red"},
    "very_unhealthy": {"range": (201, 300), "color": "purple"},
    "hazardous": {"range": (301, 500), "color": "maroon"},
}

HEALTH_RECOMMENDATIONS = {
    "good": {
        "category": "Good",
        "description": "Air quality is satisfactory and air pollution poses little or no risk.",
        "general_public": "No activity restrictions.",
        "sensitive_groups": "Enjoy outdoor activities normally.",
        "children_elderly": "No precautions needed."
    },
    "moderate": {
        "category": "Moderate",
        "description": "Air quality is acceptable, but there may be a risk for some people.",
        "general_public": "No activity restrictions for most people.",
        "sensitive_groups": "Consider reducing prolonged outdoor exertion.",
        "children_elderly": "Limit prolonged outdoor activities."
    },
    "unhealthy_for_sensitive": {
        "category": "Unhealthy for Sensitive Groups",
        "description": "Members of sensitive groups (children, elderly, and people with respiratory/heart diseases) may experience health effects.",
        "general_public": "General public is not affected.",
        "sensitive_groups": "Reduce outdoor activities. If you must go out, wear N95/P100 masks.",
        "children_elderly": "Keep indoors and avoid strenuous activities."
    },
    "unhealthy": {
        "category": "Unhealthy",
        "description": "Some members of general public may begin to experience health effects.",
        "general_public": "Reduce outdoor activities and keep indoors.",
        "sensitive_groups": "Avoid all outdoor activities.",
        "children_elderly": "Stay indoors, keep windows closed, use air purifiers."
    },
    "very_unhealthy": {
        "category": "Very Unhealthy",
        "description": "Health warnings of emergency conditions exist.",
        "general_public": "Avoid outdoor activities. Minimize outdoor movement.",
        "sensitive_groups": "Remain indoors and keep activity levels low.",
        "children_elderly": "Stay indoors with air filtration, use N95 masks if outdoors."
    },
    "hazardous": {
        "category": "Hazardous",
        "description": "The entire population is more likely to be affected.",
        "general_public": "Stay indoors and keep activities at minimum levels.",
        "sensitive_groups": "Take all precautions to remain indoors.",
        "children_elderly": "Avoid all outdoor activities. Seek air-filtered environments."
    },
}


def get_aqi_category(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "good"
    elif aqi_value <= 100:
        return "moderate"
    elif aqi_value <= 150:
        return "unhealthy_for_sensitive"
    elif aqi_value <= 200:
        return "unhealthy"
    elif aqi_value <= 300:
        return "very_unhealthy"
    else:
        return "hazardous"


def get_health_recommendation(aqi_value: float) -> HealthRecommendation:
    category_key = get_aqi_category(aqi_value)
    rec_data = HEALTH_RECOMMENDATIONS[category_key]
    
    for cat, info in AQI_CATEGORIES.items():
        if info["range"][0] <= aqi_value <= info["range"][1]:
            aqi_range = f"{info['range'][0]}-{info['range'][1]}"
            break
    
    return HealthRecommendation(
        aqi_range=aqi_range,
        category=rec_data["category"],
        description=rec_data["description"],
        general_public=rec_data["general_public"],
        sensitive_groups=rec_data["sensitive_groups"],
        children_elderly=rec_data["children_elderly"]
    )