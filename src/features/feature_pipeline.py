import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.api_client import fetch_historical_weather, fetch_historical_aqi
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
        latitude = float(os.getenv("LATITUDE", 25.3792))
        longitude = float(os.getenv("LONGITUDE", 68.3683))
        timezone = os.getenv("TIMEZONE", "Asia/Karachi")
        hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
        hopsworks_project = os.getenv("HOPSWORKS_PROJECT")
        
        if not hopsworks_api_key:
            raise ValueError("HOPSWORKS_API_KEY not set in environment")
        
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(hours=72)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
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
        
        cutoff_time = pd.Timestamp(end_date - timedelta(hours=24)).tz_localize('UTC')
        recent_df = features_df[features_df['time'] >= cutoff_time].copy()
        
        if len(recent_df) == 0:
            return True
        
        project, fs = connect_hopsworks(hopsworks_api_key, hopsworks_project)
        
        fg = create_feature_group(
            fs,
            name="aqi_features",
            version=1,
        )
        
        insert_features(fg, recent_df)
        
        return True
        
    except Exception as e:
        print(f"PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_feature_pipeline()
    sys.exit(0 if success else 1)
