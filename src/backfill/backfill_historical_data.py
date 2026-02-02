import sys, os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.api_client import fetch_historical_weather, fetch_historical_aqi
from features.feature_engineering import process_features
from backend.hopsworks_client import connect_hopsworks, create_feature_group, insert_features

load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'), help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--batch-days', type=int, default=90, help='Days per batch (default: 90)')
    return parser.parse_args()

def fetch_data_batch(start_date: str, end_date: str, latitude: float, longitude: float, timezone: str) -> pd.DataFrame:
    weather_df = fetch_historical_weather(start_date=start_date, end_date=end_date, latitude=latitude, longitude=longitude, timezone=timezone)
    aqi_df = fetch_historical_aqi(start_date=start_date, end_date=end_date, latitude=latitude, longitude=longitude, timezone=timezone)
    merged_df = pd.merge(weather_df, aqi_df, on='time', how='inner')
    return merged_df

def main():
    args = parse_arguments()
      
    LATITUDE = 25.3792
    LONGITUDE = 68.3683
    TIMEZONE = 'Asia/Karachi'
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    HOPSWORKS_PROJECT = os.getenv('HOPSWORKS_PROJECT')
    
    upload_to_hopsworks = bool(HOPSWORKS_API_KEY)    
    
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    total_days = (end - start).days
    print(f"Backfilling data from {args.start_date} to {args.end_date}")
    print(f"Estimated: {total_days} days (~{total_days * 24} hourly records)\n")
    
    project, fs, fg = None, None, None
    if upload_to_hopsworks:
        project, fs = connect_hopsworks(HOPSWORKS_API_KEY, HOPSWORKS_PROJECT)
        fg = create_feature_group(fs, "aqi_features", version=1)
    
    all_data = []
    current_date = start
    batch_num = 1
    
    while current_date < end:
        batch_end = min(current_date + timedelta(days=args.batch_days), end)
        batch_start_str = current_date.strftime('%Y-%m-%d')
        batch_end_str = batch_end.strftime('%Y-%m-%d')
        
        try:
            batch_df = fetch_data_batch(batch_start_str, batch_end_str, LATITUDE, LONGITUDE, TIMEZONE)
            all_data.append(batch_df)
            batch_num += 1
        except Exception:
            pass
        
        current_date = batch_end
    
    if not all_data:
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Fetched {len(combined_df)} raw data rows")
    
    features_df = process_features(combined_df, include_lags=True, include_aqi_rate=False)
    print(f"Generated {len(features_df)} feature rows after processing (after lag computation and NaN removal)")
    
    if upload_to_hopsworks and fs and fg:
        try:
            insert_features(fg, features_df)
        except Exception:
            pass

if __name__ == "__main__":
    main()