import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import shap

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.api_client import (
    fetch_weather_forecast,
    fetch_historical_weather, fetch_historical_aqi
)
from backend.hopsworks_client import (
    connect_hopsworks, load_model_from_registry, get_forecast_features
)
from features.feature_engineering import process_features, process_forecast_features, prepare_for_prediction
from backend.schemas import (
    PredictionResponse, PredictionItem, AlertResponse, 
    get_aqi_category, get_health_recommendation
)

def _now_in_timezone(timezone: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(timezone))
    except Exception:
        return datetime.now()

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
        model_name = (model_name or "").strip().lower().replace(" ", "_")
        if api_key is None:
            api_key = os.getenv('HOPSWORKS_API_KEY')
            if not api_key:
                return load_model_from_disk(model_name=model_name)
        allow_disk_fallback = os.getenv("ALLOW_DISK_FALLBACK", "false").strip().lower() == "true"
        
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
                return load_model_from_disk(model_name=model_name) if allow_disk_fallback else None
            
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
            
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    feature_names = json.load(f)
            else:
                feature_names = []

            feature_names = _resolve_feature_names(model, feature_names)
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}

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
            return load_model_from_disk(model_name=model_name) if allow_disk_fallback else None
            
    except Exception as e:
        return None

def load_model_from_disk(model_dir: str = None, model_name: str = "lightgbm") -> Optional[Dict[str, Any]]:
    try:
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        
        candidate_model = f"{model_name}.pkl"
        model_path = os.path.join(model_dir, candidate_model)
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "lightgbm.pkl")

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

def _aqi_to_pm25(aqi: float) -> float:
    breakpoints = [
        (0,   50,   0.0,  12.0),
        (51,  100,  12.1, 35.4),
        (101, 150,  35.5, 55.4),
        (151, 200,  55.5, 150.4),
        (201, 300,  150.5, 250.4),
        (301, 500,  250.5, 500.4),
    ]
    aqi = max(0.0, min(aqi, 500.0))
    for aqi_lo, aqi_hi, pm_lo, pm_hi in breakpoints:
        if aqi <= aqi_hi:
            return pm_lo + (aqi - aqi_lo) * (pm_hi - pm_lo) / (aqi_hi - aqi_lo)
    return 500.0


def _aqi_to_co(aqi: float, last_co: float) -> float:
    return max(0.0, last_co)


def generate_forecast(model_artifacts: Dict[str, Any], 
                     hours: int = 72, 
                     latitude: float = 25.3792,
                     longitude: float = 68.3683,
                     timezone: str = "Asia/Karachi") -> PredictionResponse:
 
    try:
        next_local_hour = pd.Timestamp.now(tz=timezone).ceil('H')
        next_local_hour_utc = next_local_hour.tz_convert('UTC')
        end_time_utc = next_local_hour_utc + pd.Timedelta(hours=hours)

        def _to_utc(series: pd.Series, tz_name: str) -> pd.Series:
            ts = pd.to_datetime(series)
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(tz_name)
            return ts.dt.tz_convert('UTC')

        # --- 1. Weather forecast -----------------------------------------------
        forecast_df = pd.DataFrame()
        hopsworks_api_key = os.getenv('HOPSWORKS_API_KEY')
        if hopsworks_api_key:
            try:
                project, fs = connect_hopsworks(hopsworks_api_key)
                forecast_df = get_forecast_features(
                    fs,
                    start_time=next_local_hour_utc.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time=end_time_utc.strftime('%Y-%m-%d %H:%M:%S'),
                )
            except Exception:
                forecast_df = pd.DataFrame()

        if forecast_df.empty:
            forecast_days = int(np.ceil(hours / 24)) + 1
            weather_forecast_df = fetch_weather_forecast(
                days=forecast_days,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone
            )
            forecast_df = weather_forecast_df.copy()

        # --- 2. Freshest historical data (archive API lags by ~1 day) -------
        now_local = _now_in_timezone(timezone)
        end_date = (now_local - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (now_local - timedelta(days=4)).strftime('%Y-%m-%d')

        hist_weather = fetch_historical_weather(
            start_date=start_date, end_date=end_date,
            latitude=latitude, longitude=longitude, timezone=timezone,
        )
        hist_aqi = fetch_historical_aqi(
            start_date=start_date, end_date=end_date,
            latitude=latitude, longitude=longitude, timezone=timezone,
        )

        hist_combined = pd.merge(hist_weather, hist_aqi, on='time', how='inner')
        hist_combined = hist_combined.sort_values('time').reset_index(drop=True)
        hist_combined['time'] = pd.to_datetime(
            hist_combined['time'], utc=True
        ).dt.tz_convert(timezone)

        # --- 3. Align forecast times -------------------------------------------
        forecast_df = forecast_df.copy()
        forecast_df['time'] = _to_utc(forecast_df['time'], timezone)
        forecast_df = forecast_df.sort_values('time').reset_index(drop=True)
        forecast_df = forecast_df[forecast_df['time'] >= next_local_hour_utc].head(hours)

        if forecast_df.empty:
            forecast_days = int(np.ceil(hours / 24)) + 1
            weather_forecast_df = fetch_weather_forecast(
                days=forecast_days, latitude=latitude,
                longitude=longitude, timezone=timezone,
            )
            forecast_df = weather_forecast_df.copy()
            forecast_df['time'] = _to_utc(forecast_df['time'], timezone)
            forecast_df = forecast_df.sort_values('time').reset_index(drop=True)
            forecast_df = forecast_df[forecast_df['time'] >= next_local_hour_utc].head(hours)

        if forecast_df.empty:
            raise ValueError("No forecast rows available after aligning to next hour")

        forecast_local = forecast_df.copy()
        forecast_local['time'] = forecast_local['time'].dt.tz_convert(timezone)

        # --- 4. Build the rolling buffer for autoregressive prediction ----------
        # Keep the last 26 rows of raw history (enough for 24h lags + 1h margin)
        BUFFER_LEN = 26
        hist_sorted = hist_combined.sort_values('time').reset_index(drop=True)
        buffer_df = hist_sorted.tail(BUFFER_LEN).copy()
        # Store AQI, PM2.5, CO, temperature, pressure as rolling arrays
        aqi_buffer = list(buffer_df['aqi'].values)
        pm25_buffer = list(buffer_df['pm2_5'].values)
        co_buffer = list(buffer_df['carbon_monoxide'].values)
        temp_buffer = list(buffer_df['temperature_2m'].values)
        pres_buffer = list(buffer_df['pressure_msl'].values)
        last_co = co_buffer[-1] if co_buffer else 300.0

        # --- 5. Process per-row weather + time features -------------------------
        forecast_features = process_forecast_features(forecast_local)

        model = model_artifacts['model']
        feature_names = model_artifacts['feature_names']

        prediction_items = []
        predictions_list = []

        for idx in range(len(forecast_features)):
            row = forecast_features.iloc[[idx]].copy()

            # ---- Compute lag features from buffer ----
            def _lag(buf, n):
                return buf[-n] if len(buf) >= n else buf[0]

            for lag in [1, 3, 6, 12, 24]:
                row[f'pm2_5_lag_{lag}h'] = _lag(pm25_buffer, lag)
                row[f'carbon_monoxide_lag_{lag}h'] = _lag(co_buffer, lag)

            row['temperature_2m_lag_12h'] = _lag(temp_buffer, 12)
            row['pressure_msl_lag_12h'] = _lag(pres_buffer, 12)

            # ---- Compute AQI change features (using only past values) ----
            aqi_prev = _lag(aqi_buffer, 1)
            row['aqi_change_1h']  = aqi_prev - _lag(aqi_buffer, 2)
            row['aqi_change_3h']  = aqi_prev - _lag(aqi_buffer, 4)
            row['aqi_change_6h']  = aqi_prev - _lag(aqi_buffer, 7)
            row['aqi_change_24h'] = aqi_prev - _lag(aqi_buffer, 25)
            row['aqi_rate_1h']  = np.clip(row['aqi_change_1h'].values[0]  / 1.0, -10, 10)
            row['aqi_rate_3h']  = np.clip(row['aqi_change_3h'].values[0]  / 3.0, -10, 10)
            row['aqi_rate_24h'] = np.clip(row['aqi_change_24h'].values[0] / 24.0, -10, 10)

            # ---- Align to model features & predict ----
            X = prepare_for_prediction(row)
            for feat in feature_names:
                if feat not in X.columns:
                    X[feat] = 0
            X = X[feature_names].ffill().bfill().fillna(0)

            pred_aqi = float(np.clip(model.predict(X)[0], 0, 500))
            predictions_list.append(pred_aqi)

            prediction_items.append(PredictionItem(
                timestamp=pd.Timestamp(row['time'].iloc[0]).to_pydatetime(),
                predicted_aqi=pred_aqi,
                aqi_category=get_aqi_category(pred_aqi),
            ))

            # ---- Update buffers for next step ----
            aqi_buffer.append(pred_aqi)
            pm25_buffer.append(_aqi_to_pm25(pred_aqi))
            co_buffer.append(_aqi_to_co(pred_aqi, last_co))
            # Weather values come from the forecast row
            temp_buffer.append(
                float(row['temperature_2m'].values[0])
                if 'temperature_2m' in row.columns else temp_buffer[-1]
            )
            pres_buffer.append(
                float(row['pressure_msl'].values[0])
                if 'pressure_msl' in row.columns else pres_buffer[-1]
            )

        predictions = np.array(predictions_list)
        peak_idx = int(np.argmax(predictions))
        peak_aqi = float(predictions[peak_idx])
        peak_time = pd.Timestamp(forecast_features.iloc[peak_idx]['time']).to_pydatetime()

        response = PredictionResponse(
            predictions=prediction_items,
            peak_aqi=peak_aqi,
            peak_time=peak_time,
            forecast_hours=hours,
            generated_at=_now_in_timezone(timezone),
        )

        return response

    except Exception as e:
        raise

def get_current_aqi(model_artifacts: Dict[str, Any],
                   latitude: float = 25.3792,
                   longitude: float = 68.3683,
                   timezone: str = "Asia/Karachi") -> Dict[str, Any]:
    try:
        now_local = _now_in_timezone(timezone)
        end_date = (now_local - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (now_local - timedelta(days=2)).strftime('%Y-%m-%d')
        
        weather_hist = fetch_historical_weather(start_date, end_date, latitude, longitude, timezone)
        aqi_hist = fetch_historical_aqi(start_date, end_date, latitude, longitude, timezone)

        hist_combined = pd.merge(weather_hist, aqi_hist, on='time', how='inner')
        hist_combined = hist_combined.sort_values('time').reset_index(drop=True)
        hist_combined['time'] = pd.to_datetime(hist_combined['time'], utc=True).dt.tz_convert(timezone)

        features_df = process_features(
            hist_combined,
            include_lags=True,
            include_aqi_change_rate=True,
            include_aqi_rate=False
        )

        current_features = features_df.tail(1)
        current_row = hist_combined.loc[current_features.index[0]]

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
    elif max_aqi >= 51:
        alert_level = "Moderate"
    else:
        alert_level = "Good"
    
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

def get_historical_data(days: int = 5,
                       latitude: float = 25.3792,
                       longitude: float = 68.3683,
                       timezone: str = "Asia/Karachi") -> pd.DataFrame:
    try:
        now_local = _now_in_timezone(timezone)
        end_date = now_local.strftime('%Y-%m-%d')
        start_date = (now_local - timedelta(days=days)).strftime('%Y-%m-%d')
        
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

def get_actual_vs_predicted(model_artifacts: Dict[str, Any],
                            days: int = 5,
                            latitude: float = 25.3792,
                            longitude: float = 68.3683,
                            timezone: str = "Asia/Karachi") -> pd.DataFrame:
    try:
        historical_df = get_historical_data(days, latitude, longitude, timezone)
        
        if historical_df.empty:
            return pd.DataFrame()
        
        model = model_artifacts.get('model')
        feature_names = model_artifacts.get('feature_names', [])
        
        if model is None:
            return pd.DataFrame()
        
        feature_names = _resolve_feature_names(model, feature_names)
        
        historical_df = historical_df.sort_values('time').reset_index(drop=True)
        
        comparison_data = []
        
        for idx in range(len(historical_df)):
            try:
                current_row = historical_df.iloc[idx]
                
                actual_aqi = current_row.get('us_aqi') or current_row.get('aqi')
                if pd.isna(actual_aqi):
                    continue
                
                timestamp = current_row.get('time')
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                
                feature_vector = {}
                
                for feat in feature_names:
                    if feat in current_row.index:
                        val = current_row[feat]
                        feature_vector[feat] = float(val) if not pd.isna(val) else 0.0
                
                for lag in [1, 3, 6, 12, 24]:
                    lag_idx = idx - lag
                    if lag_idx >= 0:
                        lag_row = historical_df.iloc[lag_idx]
                        
                        if f'pm2_5_lag_{lag}h' in feature_names:
                            feature_vector[f'pm2_5_lag_{lag}h'] = float(
                                lag_row.get('pm2_5', 0.0) if not pd.isna(lag_row.get('pm2_5')) else 0.0
                            )
                        
                        if f'carbon_monoxide_lag_{lag}h' in feature_names:
                            feature_vector[f'carbon_monoxide_lag_{lag}h'] = float(
                                lag_row.get('carbon_monoxide', 0.0) if not pd.isna(lag_row.get('carbon_monoxide')) else 0.0
                            )
                
                if idx >= 12:
                    lag12_row = historical_df.iloc[idx - 12]
                    if 'temperature_2m_lag_12h' in feature_names:
                        feature_vector['temperature_2m_lag_12h'] = float(
                            lag12_row.get('temperature_2m', 0.0) if not pd.isna(lag12_row.get('temperature_2m')) else 0.0
                        )
                    if 'pressure_msl_lag_12h' in feature_names:
                        feature_vector['pressure_msl_lag_12h'] = float(
                            lag12_row.get('pressure_msl', 0.0) if not pd.isna(lag12_row.get('pressure_msl')) else 0.0
                        )
                
                if idx >= 1:
                    aqi_prev = historical_df.iloc[idx - 1].get('aqi') or historical_df.iloc[idx - 1].get('us_aqi')
                    aqi_curr = actual_aqi
                    
                    if 'aqi_change_1h' in feature_names:
                        change_1h = aqi_curr - aqi_prev if not pd.isna(aqi_prev) else 0.0
                        feature_vector['aqi_change_1h'] = float(change_1h)
                    
                    if 'aqi_rate_1h' in feature_names:
                        change_1h = aqi_curr - aqi_prev if not pd.isna(aqi_prev) else 0.0
                        feature_vector['aqi_rate_1h'] = float(np.clip(change_1h / 1.0, -10, 10))
                
                for feat in feature_names:
                    if feat not in feature_vector:
                        feature_vector[feat] = 0.0
                
                X_pred = np.array([feature_vector.get(feat, 0.0) for feat in feature_names]).reshape(1, -1)
                
                predicted_aqi = float(model.predict(X_pred)[0])
                
                comparison_data.append({
                    'timestamp': timestamp,
                    'actual_aqi': float(actual_aqi),
                    'predicted_aqi': predicted_aqi,
                    'error': float(predicted_aqi - actual_aqi),
                    'abs_error': abs(float(predicted_aqi - actual_aqi))
                })
                
            except Exception as e:
                continue
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('timestamp').reset_index(drop=True)
            return comparison_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        raise