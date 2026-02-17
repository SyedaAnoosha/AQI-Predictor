# Compare actual vs predicted US AQI for the current hour.

import os
import sys
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.backend.api_client import get_api_client, _weather_api_with_cache_retry

LATITUDE = 25.3792
LONGITUDE = 68.3683
TIMEZONE = "Asia/Karachi"
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'cache'))
PREDICTIONS_PATH = os.path.join(CACHE_DIR, 'predictions_72h.json')


def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def fetch_current_aqi() -> dict:
    """Fetch current-hour actual AQI + pollutants from Open-Meteo."""
    client = get_api_client()
    today = datetime.now().strftime("%Y-%m-%d")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10", "pm2_5", "nitrogen_dioxide",
            "sulphur_dioxide", "carbon_monoxide", "us_aqi",
        ],
        "start_date": today,
        "end_date": today,
        "domains": "auto",
        "timezone": TIMEZONE,
    }

    response = _weather_api_with_cache_retry(
        client,
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params,
    )

    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        periods=len(hourly.Variables(0).ValuesAsNumpy()),
        freq=pd.DateOffset(hours=1),
    )

    df = pd.DataFrame({
        "time": times,
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "nitrogen_dioxide": hourly.Variables(2).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(4).ValuesAsNumpy(),
        "us_aqi": hourly.Variables(5).ValuesAsNumpy(),
    })

    # Find the row closest to the current hour
    now_utc = pd.Timestamp.now(tz="UTC")
    current_hour = now_utc.floor("h")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    idx = (df["time"] - current_hour).abs().idxmin()
    row = df.loc[idx]

    return {
        "time": row["time"],
        "us_aqi": float(row["us_aqi"]) if not math.isnan(row["us_aqi"]) else None,
        "pm2_5": float(row["pm2_5"]) if not math.isnan(row["pm2_5"]) else None,
        "pm10": float(row["pm10"]) if not math.isnan(row["pm10"]) else None,
        "carbon_monoxide": float(row["carbon_monoxide"]) if not math.isnan(row["carbon_monoxide"]) else None,
        "nitrogen_dioxide": float(row["nitrogen_dioxide"]) if not math.isnan(row["nitrogen_dioxide"]) else None,
        "sulphur_dioxide": float(row["sulphur_dioxide"]) if not math.isnan(row["sulphur_dioxide"]) else None,
    }

def load_predicted_aqi(target_utc: pd.Timestamp) -> dict | None:
    """Return the cached prediction closest to *target_utc*, or None."""
    if not os.path.exists(PREDICTIONS_PATH):
        return None

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        return None

    generated_at = data.get("generated_at", "unknown")

    best, best_diff = None, timedelta(hours=999)
    for p in predictions:
        ts = pd.Timestamp(p["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Karachi")
        ts_utc = ts.tz_convert("UTC")
        diff = abs(ts_utc - target_utc)
        if diff < best_diff:
            best_diff = diff
            best = p

    if best is None or best_diff > timedelta(hours=1):
        return None

    return {
        "timestamp": best["timestamp"],
        "predicted_aqi": best["predicted_aqi"],
        "category": best.get("aqi_category", ""),
        "generated_at": generated_at,
        "match_offset": str(best_diff),
    }

def compare():
    print("=" * 60)
    print("  AQI COMPARISON — Actual vs Predicted (Current Hour)")
    print("=" * 60)

    # 1. Actual
    print("\nFetching actual AQI from Open-Meteo …")
    try:
        actual = fetch_current_aqi()
    except Exception as e:
        print(f"  ERROR fetching actual AQI: {e}")
        return

    local_time = actual["time"].tz_convert(TIMEZONE)
    actual_aqi = actual["us_aqi"]

    print(f"\n  Location      : Hyderabad, Sindh ({LATITUDE}, {LONGITUDE})")
    print(f"  Time (local)  : {local_time.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  Time (UTC)    : {actual['time'].strftime('%Y-%m-%d %H:%M UTC')}")

    if actual_aqi is not None:
        print(f"\n  Actual US AQI : {actual_aqi:.1f}  [{get_aqi_category(actual_aqi)}]")
    else:
        print("\n  Actual US AQI : unavailable (NaN from API)")

    print(f"  PM2.5         : {actual['pm2_5']:.1f} µg/m³" if actual["pm2_5"] else "  PM2.5         : N/A")
    print(f"  PM10          : {actual['pm10']:.1f} µg/m³" if actual["pm10"] else "  PM10          : N/A")
    print(f"  CO            : {actual['carbon_monoxide']:.1f} µg/m³" if actual["carbon_monoxide"] else "  CO            : N/A")
    print(f"  NO₂           : {actual['nitrogen_dioxide']:.1f} µg/m³" if actual["nitrogen_dioxide"] else "  NO₂           : N/A")
    print(f"  SO₂           : {actual['sulphur_dioxide']:.1f} µg/m³" if actual["sulphur_dioxide"] else "  SO₂           : N/A")

    # 2. Predicted
    print("\n" + "-" * 60)
    pred = load_predicted_aqi(actual["time"])

    if pred is None:
        print("  No cached prediction found for this hour.")
        print(f"  (Checked: {PREDICTIONS_PATH})")
        print("  Run the training pipeline to generate fresh predictions.")
        return

    predicted_aqi = pred["predicted_aqi"]
    print(f"  Predicted AQI : {predicted_aqi:.1f}  [{get_aqi_category(predicted_aqi)}]")
    print(f"  Forecast from : {pred['generated_at']}")
    print(f"  Match offset  : {pred['match_offset']}")

    # 3. Error metrics
    if actual_aqi is not None:
        error = predicted_aqi - actual_aqi
        abs_error = abs(error)
        pct_error = (abs_error / actual_aqi * 100) if actual_aqi != 0 else float("inf")
        direction = "over" if error > 0 else "under"

        print("\n" + "-" * 60)
        print("  COMPARISON")
        print(f"  Error         : {error:+.1f} ({direction}-predicted)")
        print(f"  Abs Error     : {abs_error:.1f}")
        print(f"  % Error       : {pct_error:.1f}%")

        cat_actual = get_aqi_category(actual_aqi)
        cat_pred = get_aqi_category(predicted_aqi)
        cat_match = "YES ✓" if cat_actual == cat_pred else f"NO ✗  (actual={cat_actual}, pred={cat_pred})"
        print(f"  Category match: {cat_match}")

        # Quick verdict
        if pct_error < 10:
            verdict = "EXCELLENT — within 10%"
        elif pct_error < 20:
            verdict = "GOOD — within 20%"
        elif pct_error < 30:
            verdict = "FAIR — within 30%"
        else:
            verdict = "POOR — error exceeds 30%"
        print(f"\n  Verdict       : {verdict}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    compare()
