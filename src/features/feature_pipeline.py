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
        
        yesterday = datetime.now() - timedelta(days=1)
        two_years_ago = yesterday - timedelta(days=730)
        
        historical_start = two_years_ago.strftime("%Y-%m-%d")
        historical_end = yesterday.strftime("%Y-%m-%d")
        
        historical_weather = fetch_historical_weather(
            start_date=historical_start,
            end_date=historical_end,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        historical_aqi = fetch_historical_aqi(
            start_date=historical_start,
            end_date=historical_end,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        forecast_weather = fetch_weather_forecast(
            days=5,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        forecast_aqi = fetch_aqi_forecast(
            days=5,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )
        
        weather_df = pd.concat([historical_weather, forecast_weather], ignore_index=True)
        aqi_df = pd.concat([historical_aqi, forecast_aqi], ignore_index=True)
        
        weather_df = weather_df.drop_duplicates(subset=['time'], keep='first')
        aqi_df = aqi_df.drop_duplicates(subset=['time'], keep='first')
        
        combined_df = pd.merge(weather_df, aqi_df, on='time', how='inner')
        
        if 'us_aqi' in combined_df.columns:
            combined_df = combined_df.rename(columns={'us_aqi': 'aqi'})
        
        features_df = process_features(
            combined_df,
            include_lags=True,
            include_aqi_rate=False,
            include_aqi_change_rate=True
        )
        
        if features_df.isnull().sum().sum() > 0:
            features_df = features_df.dropna()
        
        if len(features_df) == 0:
            return True
        
        float32_cols = [
            'pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'aqi',
            'pm2_5_lag_1h', 'pm2_5_lag_3h', 'pm2_5_lag_6h', 'pm2_5_lag_12h', 'pm2_5_lag_24h',
            'aqi_change_1h', 'aqi_change_3h', 'aqi_change_6h', 'aqi_change_24h',
            'aqi_rate_1h', 'aqi_rate_3h', 'aqi_rate_24h'
        ]
        for col in float32_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype('float32')
        
        project, fs = connect_hopsworks(hopsworks_api_key, hopsworks_project)
        
        fg = create_feature_group(
            fs,
            name="aqi_features",
            version=1,
        )
        
        batch_size = 1000
        total_rows = len(features_df)
        print(f"Uploading {total_rows} rows in batches of {batch_size}")
        
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = features_df.iloc[i:batch_end]
            print(f"Uploading batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size} ({len(batch_df)} rows)")
            insert_features(fg, batch_df)
        
        print(f"Successfully uploaded all {total_rows} rows")
        return True
        
    except Exception as e:
        print(f"PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_feature_pipeline()
    sys.exit(0 if success else 1)
