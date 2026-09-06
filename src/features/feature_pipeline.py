# Feature pipeline for AQI Predictor project
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import time
from config import load_env

load_env()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.api_client import (
    fetch_historical_weather, 
    fetch_historical_aqi,
    fetch_weather_forecast
)
from features.feature_engineering import process_features, process_forecast_features
from backend.storage import get_feature_store, HISTORICAL_COLLECTION, FORECAST_COLLECTION

def get_yesterday_date() -> str:
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")

def get_today_date() -> str:
    today = datetime.now()
    return today.strftime("%Y-%m-%d")

def run_feature_pipeline():
    try:
        latitude = 25.3792
        longitude = 68.3683
        timezone = "Asia/Karachi"
        hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
        hopsworks_project = os.getenv("HOPSWORKS_PROJECT")
        
        from backend.storage import mongo_uri as _mongo_uri
        mongo_uri = _mongo_uri()
        if not mongo_uri and not hopsworks_api_key:
            raise ValueError("MONGODB_URI (preferred) or HOPSWORKS_API_KEY must be set in environment")
        
        now = datetime.now()
        lookback_hours = 26
        lookback_start = now - timedelta(hours=lookback_hours)
        
        start_str = lookback_start.strftime("%Y-%m-%d")
        end_str = now.strftime("%Y-%m-%d")
        
        print(f"Incremental update: Fetching observed data from last {lookback_hours} hours")
        print(f"Date range: {start_str} to {end_str}")
        
        def _fetch_with_retry(fetch_fn, max_retries: int = 3, base_wait: int = 3, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return fetch_fn(**kwargs)
                except Exception:
                    if attempt >= max_retries:
                        raise
                    time.sleep(base_wait * attempt)
            raise RuntimeError(f"{getattr(fetch_fn, '__name__', 'fetch')} exhausted retries")

        weather_df = _fetch_with_retry(
            fetch_historical_weather,
            start_date=start_str,
            end_date=end_str,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )

        aqi_df = _fetch_with_retry(
            fetch_historical_aqi,
            start_date=start_str,
            end_date=end_str,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        weather_df = weather_df.drop_duplicates(subset=['time'], keep='last')
        aqi_df = aqi_df.drop_duplicates(subset=['time'], keep='last')
        
        combined_df = pd.merge(weather_df, aqi_df, on='time', how='inner')
        
        if 'us_aqi' in combined_df.columns:
            combined_df = combined_df.rename(columns={'us_aqi': 'aqi'})

        
        print(f"Fetched {len(combined_df)} rows of observed data")
        
        features_df = process_features(
            combined_df,
            include_lags=True,
            include_aqi_rate=False,
            include_aqi_change_rate=True,
            use_causal_imputation=True  # Backward-only rolling median (safe for incremental updates)
        )
        
        if features_df.isnull().sum().sum() > 0:
            print(f"Dropping {features_df.isnull().sum().sum()} rows with NaN values")
            features_df = features_df.dropna()
        
        if len(features_df) == 0:
            print("No new data to insert after feature engineering")
            return True
        
        features_df['time'] = pd.to_datetime(features_df['time'], utc=True)
        
        # cutoff_time = pd.Timestamp(now - timedelta(hours=2)).tz_localize('UTC')
        # new_data = features_df[features_df['time'] >= cutoff_time].copy()
        
        # if len(new_data) == 0:
        #     print("No new timestamps to insert (all data already in feature store)")
        #     return True
        
        new_data = features_df.copy()

        if len(new_data) == 0:
            print("No data to insert")
            return True
      
        float32_cols = [
            'pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'aqi',
            'pm2_5_lag_1h', 'pm2_5_lag_3h', 'pm2_5_lag_6h', 'pm2_5_lag_12h', 'pm2_5_lag_24h',
            'aqi_change_1h', 'aqi_change_3h', 'aqi_change_6h', 'aqi_change_24h',
            'aqi_rate_1h', 'aqi_rate_3h', 'aqi_rate_24h'
        ]
        for col in float32_cols:
            if col in new_data.columns:
                new_data[col] = new_data[col].astype('float32')
        
        # Write through the storage layer: MongoDB first, then Hopsworks.
        store = get_feature_store()
        results = store.write_features(HISTORICAL_COLLECTION, new_data)
        print(f"Observed features written: {results}")

        if not any(isinstance(v, int) and v >= 0 for v in results.values()):
            raise RuntimeError(f"No storage backend accepted observed features: {results}")

        try:
            # Fetch 4 days to account for timezone offset (Asia/Karachi is UTC+5)
            # This ensures we get full 72-hour forecast from current time
            weather_forecast_df = _fetch_with_retry(
                fetch_weather_forecast,
                days=4,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone
            )

            weather_forecast_df = weather_forecast_df.drop_duplicates(subset=['time'], keep='last')
            
            # Filter to only future timestamps (remove any past data accidentally included)
            now_utc = pd.Timestamp.now(tz='UTC')
            weather_forecast_df['time'] = pd.to_datetime(weather_forecast_df['time'], utc=True)
            weather_forecast_df = weather_forecast_df[weather_forecast_df['time'] > now_utc].copy()
            
            if len(weather_forecast_df) == 0:
                print("⚠️  Warning: No future forecast data available after filtering")
            else:
                print(f"Forecast time range: {weather_forecast_df['time'].min()} → {weather_forecast_df['time'].max()}")
            
            forecast_features = process_forecast_features(weather_forecast_df)
            forecast_features['time'] = pd.to_datetime(forecast_features['time'], utc=True)

            forecast_results = store.write_features(FORECAST_COLLECTION, forecast_features)
            print(f"Forecast features written: {forecast_results}")
        except Exception as forecast_error:
            print(f"Forecast feature update skipped: {forecast_error}")
        
        return True
        
    except Exception as e:
        print(f"PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_feature_pipeline()
    sys.exit(0 if success else 1)
