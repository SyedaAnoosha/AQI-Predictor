import hopsworks
import pandas as pd
from datetime import datetime

# Connect to Hopsworks
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Get the forecast features group
print("📖 Fetching 'weather_forecast_features'...")
fg = fs.get_feature_group("weather_forecast_features", version=1)

# Read all data from feature group
df = fg.read()

print(f"\n✅ Successfully read data from 'weather_forecast_features'")
print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Analyze the time range
df['time'] = pd.to_datetime(df['time'], utc=True)
min_time = df['time'].min()
max_time = df['time'].max()
now = pd.Timestamp.now(tz='UTC')

print(f"\n⏰ FORECAST TIME RANGE:")
print(f"   Now (UTC):        {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Forecast Start:   {min_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Forecast End:     {max_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Duration:         {(max_time - min_time).total_seconds() / 3600:.1f} hours")
print(f"   Expected:         ~72 hours (3 days)")

# Check if any past data exists
past_data = df[df['time'] <= now]
if len(past_data) > 0:
    print(f"\n⚠️  WARNING: Found {len(past_data)} rows with PAST timestamps!")
    print(f"   These should have been filtered out")
else:
    print(f"\n✅ No past data found (good!)")

print("\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n🔍 First 5 rows:")
print(df.head())

print("\n📈 Data Types:")
print(df.dtypes)

print("\n💾 Missing Values:")
print(df.isnull().sum())

# Option: Save to CSV
save_choice = input("\n💾 Save to CSV? (y/n): ").strip().lower()
if save_choice == 'y':
    csv_path = "forecast_features_export.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved to {csv_path}")