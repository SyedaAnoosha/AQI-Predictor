#Historical data backfill script for AQI Predictor project
import sys, os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.api_client import fetch_historical_weather, fetch_historical_aqi
from features.feature_engineering import process_features
from backend.storage import get_feature_store, HISTORICAL_COLLECTION
from config import load_env

load_env()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'), help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--batch-days', type=int, default=90, help='Days per batch (default: 90)')
    return parser.parse_args()

def fetch_data_batch(start_date: str, end_date: str, latitude: float, longitude: float, timezone: str) -> pd.DataFrame:
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
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone
    )
    aqi_df = _fetch_with_retry(
        fetch_historical_aqi,
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone
    )
    merged_df = pd.merge(weather_df, aqi_df, on='time', how='inner')
    return merged_df

def main():
    args = parse_arguments()
      
    LATITUDE = 25.3792
    LONGITUDE = 68.3683
    TIMEZONE = 'Asia/Karachi'
    from backend.storage import mongo_uri
    if not mongo_uri() and not os.getenv('HOPSWORKS_API_KEY'):
        print("ERROR: Set MONGODB_URI (preferred) or HOPSWORKS_API_KEY")
        return

    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    total_days = (end - start).days
    print(f"Backfilling data from {args.start_date} to {args.end_date}")
    print(f"Estimated: {total_days} days (~{total_days * 24} hourly records)")

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
        except Exception as e:
            print(f"Batch {batch_num} failed: {e}")
        
        current_date = batch_end
    
    if not all_data:
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['time'], keep='last')
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    print(f"Fetched {len(combined_df)} raw data rows")
    
    features_df = process_features(combined_df, include_lags=True, include_aqi_rate=False)
    print(f"Generated {len(features_df)} feature rows after processing (after lag computation and NaN removal)")
    
    # Upserts are keyed on `time`, so re-running a backfill is safe.
    results = get_feature_store().write_features(HISTORICAL_COLLECTION, features_df)
    print(f"Backfill written: {results}")


if __name__ == "__main__":
    main()