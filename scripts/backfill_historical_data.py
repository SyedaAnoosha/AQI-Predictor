"""Historical Data Backfill Script - Fetches data and inserts into Hopsworks Feature Store"""

import sys, os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.backend.api_client import fetch_historical_weather, fetch_historical_aqi
from src.features.feature_engineering import process_features
from src.backend.hopsworks_client import connect_hopsworks, create_feature_group, insert_features

load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Backfill historical AQI data to Hopsworks')
    parser.add_argument('--start-date', type=str, default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'), help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--batch-days', type=int, default=90, help='Days per batch (default: 90)')
    return parser.parse_args()


def fetch_data_batch(start_date: str, end_date: str, latitude: float, longitude: float, timezone: str) -> pd.DataFrame:
    print(f"Fetching data: {start_date} to {end_date}")
    weather_df = fetch_historical_weather(start_date=start_date, end_date=end_date, latitude=latitude, longitude=longitude, timezone=timezone)
    aqi_df = fetch_historical_aqi(start_date=start_date, end_date=end_date, latitude=latitude, longitude=longitude, timezone=timezone)
    merged_df = pd.merge(weather_df, aqi_df, on='time', how='inner')
    print(f"  ✓ {len(merged_df)} records")
    return merged_df


def main():
    args = parse_arguments()
      
    LATITUDE = float(os.getenv('LATITUDE', 25.3792))
    LONGITUDE = float(os.getenv('LONGITUDE', 68.3683))
    TIMEZONE = os.getenv('TIMEZONE', 'Asia/Karachi')
    LOCATION_NAME = os.getenv('LOCATION_NAME', 'Hyderabad_Sindh')
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    HOPSWORKS_PROJECT = os.getenv('HOPSWORKS_PROJECT')
    
    print(f"Location: {LOCATION_NAME} ({LATITUDE}°N, {LONGITUDE}°E, {TIMEZONE})")
    print(f"Date Range: {args.start_date} to {args.end_date} (batch size: {args.batch_days} days)")
    
    upload_to_hopsworks = bool(HOPSWORKS_API_KEY)
    if not upload_to_hopsworks:
        print("⚠️  No HOPSWORKS_API_KEY - data will be processed but not uploaded")
    else:
        print(f"Hopsworks Project: {HOPSWORKS_PROJECT}")    
    
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    total_days = (end - start).days
    print(f"Total: {total_days} days (~{total_days * 24} hourly records)\n")
    
    project, fs, fg = None, None, None
    if upload_to_hopsworks:
        print("Connecting to Hopsworks...")
        project, fs = connect_hopsworks(HOPSWORKS_API_KEY, HOPSWORKS_PROJECT)
        fg = create_feature_group(fs, "aqi_features", version=1)
        print("  ✓ Connected and feature group ready")
    
    all_data = []
    current_date = start
    batch_num = 1
    
    while current_date < end:
        batch_end = min(current_date + timedelta(days=args.batch_days), end)
        batch_start_str = current_date.strftime('%Y-%m-%d')
        batch_end_str = batch_end.strftime('%Y-%m-%d')
        
        print(f"\nBatch {batch_num}: {batch_start_str} to {batch_end_str}")
        
        try:
            batch_df = fetch_data_batch(batch_start_str, batch_end_str, LATITUDE, LONGITUDE, TIMEZONE)
            all_data.append(batch_df)
            batch_num += 1
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        current_date = batch_end
    
    if not all_data:
        print("\n✗ No data fetched. Exiting.")
        return
    
    print(f"\nCombining {len(all_data)} batches...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total: {len(combined_df)} records from {combined_df['time'].min()} to {combined_df['time'].max()}")
    
    print("\nProcessing features (cleaning, validation, lags)...")
    features_df = process_features(combined_df, include_lags=True, include_aqi_rate=False)
    print(f"  ✓ {len(features_df)} records, {len(features_df.columns)} features, {features_df.isnull().sum().sum()} missing values")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'backfill_{args.start_date}_{args.end_date}.csv')
    
    features_df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file} ({os.path.getsize(output_file) / (1024*1024):.2f} MB)")
    
    if upload_to_hopsworks and fs and fg:
        print("Uploading to Hopsworks...")
        try:
            insert_features(fg, features_df)
            print(f"  ✓ Uploaded {len(features_df)} records")
        except Exception as e:
            print(f"  ✗ Upload error: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"✓ Complete: {len(features_df)} records, {len(features_df.columns)} features")
    print(f"  Date: {features_df['time'].min()} to {features_df['time'].max()}")
    print(f"  AQI: {features_df['aqi'].min():.1f} - {features_df['aqi'].max():.1f}")
    print(f"  File: {output_file}")
    print(f"  Hopsworks: {'✓ Uploaded' if upload_to_hopsworks else '⚠️  Not uploaded'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
