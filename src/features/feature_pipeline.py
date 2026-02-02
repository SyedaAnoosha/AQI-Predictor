import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.api_client import (
    fetch_historical_weather, 
    fetch_historical_aqi,
    fetch_weather_forecast,
    fetch_aqi_forecast
)
from features.feature_engineering import process_features
from backend.hopsworks_client import connect_hopsworks, create_feature_group, insert_features

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
        
        if not hopsworks_api_key:
            raise ValueError("HOPSWORKS_API_KEY not set in environment")
        
        now = datetime.now()
        lookback_hours = 26
        lookback_start = now - timedelta(hours=lookback_hours)
        
        start_str = lookback_start.strftime("%Y-%m-%d")
        end_str = now.strftime("%Y-%m-%d")
        
        print(f"Incremental update: Fetching observed data from last {lookback_hours} hours")
        print(f"Date range: {start_str} to {end_str}")
        
        weather_df = fetch_historical_weather(
            start_date=start_str,
            end_date=end_str,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        aqi_df = fetch_historical_aqi(
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
            include_aqi_change_rate=True
        )
        
        if features_df.isnull().sum().sum() > 0:
            print(f"Dropping {features_df.isnull().sum().sum()} rows with NaN values")
            features_df = features_df.dropna()
        
        if len(features_df) == 0:
            print("No new data to insert after feature engineering")
            return True
        
        features_df['time'] = pd.to_datetime(features_df['time'], utc=True)
        
        cutoff_time = pd.Timestamp(now - timedelta(hours=2)).tz_localize('UTC')
        new_data = features_df[features_df['time'] >= cutoff_time].copy()
        
        if len(new_data) == 0:
            print("No new timestamps to insert (all data already in feature store)")
            return True
        
        print(f"Inserting {len(new_data)} new rows (last 2 hours only)")
        
        float32_cols = [
            'pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'aqi',
            'pm2_5_lag_1h', 'pm2_5_lag_3h', 'pm2_5_lag_6h', 'pm2_5_lag_12h', 'pm2_5_lag_24h',
            'aqi_change_1h', 'aqi_change_3h', 'aqi_change_6h', 'aqi_change_24h',
            'aqi_rate_1h', 'aqi_rate_3h', 'aqi_rate_24h'
        ]
        for col in float32_cols:
            if col in new_data.columns:
                new_data[col] = new_data[col].astype('float32')
        
        project, fs = connect_hopsworks(hopsworks_api_key, hopsworks_project)
        
        fg = create_feature_group(
            fs,
            name="aqi_features",
            version=1,
        )
        
        insert_features(fg, new_data)
        print(f"Successfully inserted {len(new_data)} new rows")
        
        return True
        
    except Exception as e:
        print(f"PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_feature_pipeline()
    sys.exit(0 if success else 1)
