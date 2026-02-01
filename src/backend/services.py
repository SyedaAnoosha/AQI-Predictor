import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import shap

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.api_client import (
    fetch_weather_forecast, fetch_aqi_forecast,
    fetch_historical_weather, fetch_historical_aqi
)
from backend.hopsworks_client import (
    connect_hopsworks, load_model_from_registry
)
from features.feature_engineering import process_features, prepare_for_prediction
from backend.schemas import (
    PredictionResponse, PredictionItem, AlertResponse, 
    get_aqi_category, get_health_recommendation
)

def _resolve_feature_names(model: Any, feature_names_from_file: List[str]) -> List[str]:
    model_feature_names: List[str] = []
    if hasattr(model, "feature_names_"):
        model_feature_names = list(model.feature_names_)
    elif hasattr(model, "feature_name_"):
        model_feature_names = list(model.feature_name_)

    if model_feature_names and len(model_feature_names) != len(feature_names_from_file):
        return model_feature_names

    return feature_names_from_file or model_feature_names

def load_model_artifacts(api_key: str = None, model_name: str = "lightgbm") -> Optional[Dict[str, Any]]:
    try:
        if api_key is None:
            api_key = os.getenv('HOPSWORKS_API_KEY')
            if not api_key:
                return load_model_from_disk(model_name=model_name)
        
        try:
            project, fs = connect_hopsworks(api_key)
            mr = project.get_model_registry()

            metric = os.getenv("MODEL_SELECTION_METRIC", "val_rmse")
            sort_by = os.getenv("MODEL_SELECTION_SORT", "min")
            
            model_info = load_model_from_registry(mr, model_name, metric=metric, sort_by=sort_by)

            if isinstance(model_info, dict):
                model_dir = model_info.get("path")
                registry_metrics = model_info.get("registry_metrics") or {}
            else:
                model_dir = model_info
                registry_metrics = {}
            
            if model_dir is None:
                return load_model_from_disk(model_name=model_name)
            
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            features_path = os.path.join(model_dir, "feature_names.json")
            metrics_path = os.path.join(model_dir, "metrics.json")
            
            if not os.path.exists(model_path):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f.lower()]
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                else:
                    raise FileNotFoundError(f"No model pickle file found in {model_dir}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(features_path, 'r') as f:
                feature_names = json.load(f)

            feature_names = _resolve_feature_names(model, feature_names)
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            if registry_metrics:
                merged_metrics = {}
                merged_metrics.update(metrics if isinstance(metrics, dict) else {})
                merged_metrics.update(registry_metrics if isinstance(registry_metrics, dict) else {})
                metrics = merged_metrics
            
            return {
                'model': model,
                'feature_names': feature_names,
                'metrics': metrics
            }
        except Exception as hops_error:
            return load_model_from_disk(model_name=model_name)
            
    except Exception as e:
        return None

def load_model_from_disk(model_dir: str = None, model_name: str = "lightgbm") -> Optional[Dict[str, Any]]:
    try:
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        
        candidate_model = f"{model_name}_model.pkl"
        model_path = os.path.join(model_dir, candidate_model)
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "lightgbm_model.pkl")

        candidate_features = f"feature_names_{model_name}.json"
        features_path = os.path.join(model_dir, candidate_features if os.path.exists(os.path.join(model_dir, candidate_features)) else "feature_names.json")

        candidate_metrics = f"metrics_{model_name}.json"
        metrics_path = os.path.join(model_dir, candidate_metrics if os.path.exists(os.path.join(model_dir, candidate_metrics)) else "metrics.json")
                
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
        else:
            if hasattr(model, 'feature_names_'):
                feature_names = list(model.feature_names_)
            elif hasattr(model, 'feature_name_'):
                feature_names = list(model.feature_name_)
            else:
                return None

        feature_names = _resolve_feature_names(model, feature_names)
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        return {
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics
        }
    except Exception as e:
        return None

def make_predictions(model_artifacts: Dict[str, Any], features_df: pd.DataFrame) -> np.ndarray:
    if model_artifacts is None:
        raise ValueError("Model artifacts not loaded")
    
    model = model_artifacts['model']
    feature_names = model_artifacts['feature_names']
    
    X = prepare_for_prediction(features_df)
    
    missing_features = set(feature_names) - set(X.columns)
    if missing_features:
        for feat in missing_features:
            X[feat] = 0
    
    X = X[feature_names].copy()
    
    X = X.ffill().bfill().fillna(0)
    
    predictions = model.predict(X)
    
    predictions = np.clip(predictions, 0, 500)
    
    return predictions

def generate_forecast(model_artifacts: Dict[str, Any], 
                     hours: int = 72, 
                     latitude: float = 25.3792,
                     longitude: float = 68.3683,
                     timezone: str = "Asia/Karachi") -> PredictionResponse:
    try:
        weather_forecast_df = fetch_weather_forecast(
            days=int(hours / 24) + 1,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        aqi_forecast_df = fetch_aqi_forecast(
            days=int(hours / 24) + 1,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        forecast_df = pd.merge(weather_forecast_df, aqi_forecast_df, on='time', how='inner')
        
        forecast_df = forecast_df.head(hours + 24)
        
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d')
        
        hist_weather = fetch_historical_weather(
            start_date=start_date,
            end_date=end_date,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        hist_aqi = fetch_historical_aqi(
            start_date=start_date,
            end_date=end_date,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        hist_combined = pd.merge(hist_weather, hist_aqi, on='time', how='inner')
        
        combined_df = pd.concat([hist_combined, forecast_df], ignore_index=True)
        combined_df = combined_df.sort_values('time').reset_index(drop=True)

        combined_df['time'] = pd.to_datetime(combined_df['time'], utc=True).dt.tz_convert(timezone)

        next_local_hour = pd.Timestamp.now(tz=timezone).ceil('H')
        combined_df['is_forecast'] = combined_df['time'] >= next_local_hour
        
        features_df = process_features(
            combined_df,
            include_lags=True,
            include_aqi_change_rate=True,
            include_aqi_rate=False
        )

        forecast_mask = combined_df.loc[features_df.index, 'is_forecast']
        forecast_features = features_df[forecast_mask.to_numpy()].copy()
        forecast_features = forecast_features[forecast_features['time'] >= next_local_hour]
        forecast_features = forecast_features.head(hours)

        if forecast_features.empty:
            raise ValueError("No forecast rows available after aligning to next local hour")
        
        predictions = make_predictions(model_artifacts, forecast_features)
        
        prediction_items = []
        for idx, (_, row) in enumerate(forecast_features.iterrows()):
            aqi_value = float(predictions[idx])
            
            prediction_items.append(PredictionItem(
                timestamp=row['time'],
                predicted_aqi=aqi_value,
                aqi_category=get_aqi_category(aqi_value),
                confidence=0.85
            ))
        
        peak_aqi = max(predictions)
        peak_time = forecast_features.iloc[np.argmax(predictions)]['time']
        
        response = PredictionResponse(
            predictions=prediction_items,
            peak_aqi=float(peak_aqi),
            peak_time=peak_time,
            forecast_hours=hours,
            generated_at=datetime.now()
        )
        
        return response
        
    except Exception as e:
        raise

def get_current_aqi(model_artifacts: Dict[str, Any],
                   latitude: float = 25.3792,
                   longitude: float = 68.3683,
                   timezone: str = "Asia/Karachi") -> Dict[str, Any]:
    try:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        weather_hist = fetch_historical_weather(start_date, end_date, latitude, longitude, timezone)
        aqi_hist = fetch_historical_aqi(start_date, end_date, latitude, longitude, timezone)

        weather_fc = fetch_weather_forecast(days=1, latitude=latitude, longitude=longitude, timezone=timezone)
        aqi_fc = fetch_aqi_forecast(days=1, latitude=latitude, longitude=longitude, timezone=timezone)

        hist_combined = pd.merge(weather_hist, aqi_hist, on='time', how='inner')
        fc_combined = pd.merge(weather_fc, aqi_fc, on='time', how='inner')
        combined_df = pd.concat([hist_combined, fc_combined], ignore_index=True).sort_values('time')
        combined_df['time'] = pd.to_datetime(combined_df['time'], utc=True).dt.tz_convert(timezone)

        features_df = process_features(
            combined_df,
            include_lags=True,
            include_aqi_change_rate=True,
            include_aqi_rate=False
        )

        current_time = pd.Timestamp.now(tz=features_df['time'].dt.tz if hasattr(features_df['time'], 'dt') else None)
        future_mask = features_df['time'] >= current_time
        if future_mask.any():
            current_features = features_df[future_mask].head(1)
        else:
            current_features = features_df.tail(1)

        current_index = current_features.index[0]
        current_row = combined_df.loc[current_index]

        current_aqi = make_predictions(model_artifacts, current_features)[0]
        
        return {
            "aqi": float(current_aqi),
            "category": get_aqi_category(current_aqi),
            "timestamp": current_features.iloc[0]['time'],
            "health_recommendation": get_health_recommendation(current_aqi),
            "pm2_5": float(current_row.get('pm2_5', 0)),
            "pm10": float(current_row.get('pm10', 0)),
            "nitrogen_dioxide": float(current_row.get('nitrogen_dioxide', 0)),
            "sulphur_dioxide": float(current_row.get('sulphur_dioxide', 0)),
            "ozone": float(current_row.get('ozone', 0)),
            "carbon_monoxide": float(current_row.get('carbon_monoxide', 0)),
            "temperature": float(current_row.get('temperature_2m', 0)),
            "humidity": float(current_row.get('relative_humidity_2m', 0))
        }
        
    except Exception as e:
        raise

def check_alerts(predictions: List[PredictionItem], current_aqi: float = None) -> AlertResponse:
    max_aqi = max([p.predicted_aqi for p in predictions])
    max_prediction = [p for p in predictions if p.predicted_aqi == max_aqi][0]
    
    if current_aqi is None:
        current_aqi = predictions[0].predicted_aqi if predictions else max_aqi
    
    if max_aqi >= 301:
        alert_level = "Hazardous"
    elif max_aqi >= 201:
        alert_level = "Very Unhealthy"
    elif max_aqi >= 151:
        alert_level = "Unhealthy"
    elif max_aqi >= 101:
        alert_level = "Unhealthy for Sensitive Groups"
    else:
        alert_level = "Good/Moderate"
    
    recommendation = get_health_recommendation(max_aqi).description
    
    affected_groups = []
    if max_aqi >= 101:
        affected_groups.append("children")
    if max_aqi >= 101:
        affected_groups.append("elderly")
    if max_aqi >= 151:
        affected_groups.append("asthma patients")
    if max_aqi >= 201:
        affected_groups.append("general public")
    
    return AlertResponse(
        alert_level=alert_level,
        current_aqi=float(current_aqi),
        peak_aqi_predicted=float(max_aqi),
        peak_time=max_prediction.timestamp,
        recommendation=recommendation,
        affected_groups=affected_groups
    )

def get_feature_importance(model_artifacts: Dict[str, Any]) -> Dict[str, float]:
    if model_artifacts is None:
        raise ValueError("Model artifacts not loaded")
    
    model = model_artifacts['model']
    feature_names = model_artifacts['feature_names']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return {}
    
    importance_dict = dict(zip(feature_names, importances))
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict

def get_shap_values(model_artifacts: Dict[str, Any], 
                    sample_data: pd.DataFrame = None,
                    num_samples: int = 100) -> Dict[str, Any]:
    if model_artifacts is None:
        raise ValueError("Model artifacts not loaded")
    
    model = model_artifacts['model']
    feature_names = model_artifacts['feature_names']
    
    try:
        if hasattr(model, 'predict') and hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            
            if sample_data is None:
                background_data = shap.sample(
                    pd.DataFrame(np.zeros((num_samples, len(feature_names))), columns=feature_names),
                    min(10, num_samples)
                )
                shap_values = explainer.shap_values(background_data)
            else:
                X_sample = prepare_for_prediction(sample_data)
                X_sample = X_sample[feature_names]
                
                shap_values = explainer.shap_values(X_sample)
            
            base_value = explainer.expected_value
            
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            shap_importance = dict(zip(feature_names, mean_abs_shap))
            shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "shap_importance": shap_importance,
                "base_value": float(base_value) if not isinstance(base_value, (list, np.ndarray)) else float(base_value[0]),
                "method": "SHAP TreeExplainer",
                "top_features": [
                    {"feature_name": name, "importance_score": float(score)}
                    for name, score in list(shap_importance.items())[:20]
                ]
            }
            
        else:
            return {
                "shap_importance": get_feature_importance(model_artifacts),
                "method": "Feature Importance (Fallback)",
                "top_features": [
                    {"feature_name": name, "importance_score": float(score)}
                    for name, score in list(get_feature_importance(model_artifacts).items())[:20]
                ]
            }
    
    except Exception as e:
        return {
            "shap_importance": get_feature_importance(model_artifacts),
            "method": "Feature Importance (Error Fallback)",
            "error": str(e),
            "top_features": [
                {"feature_name": name, "importance_score": float(score)}
                for name, score in list(get_feature_importance(model_artifacts).items())[:20]
            ]
        }

def get_historical_data(days: int = 7,
                       latitude: float = 25.3792,
                       longitude: float = 68.3683,
                       timezone: str = "Asia/Karachi") -> pd.DataFrame:
    try:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days+1)).strftime('%Y-%m-%d')
        
        weather_df = fetch_historical_weather(start_date, end_date, latitude, longitude, timezone)
        aqi_df = fetch_historical_aqi(start_date, end_date, latitude, longitude, timezone)
        
        historical_df = pd.merge(weather_df, aqi_df, on='time', how='inner')
        
        return historical_df
        
    except Exception as e:
        raise

def get_model_metrics(model_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    if model_artifacts is None or 'metrics' not in model_artifacts:
        return {}
    
    return model_artifacts['metrics']