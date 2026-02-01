import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd

_api_client = None

def get_api_client():
    global _api_client
    if _api_client is None:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
        _api_client = openmeteo_requests.Client(session=retry_session)
    return _api_client


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Karachi"
) -> pd.DataFrame:
    try:
        client = get_api_client()
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
            ],
            "timezone": timezone,
        }
        
        response = client.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                periods=len(hourly.Variables(0).ValuesAsNumpy()),
                freq=pd.DateOffset(hours=1),
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "pressure_msl": hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(3).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(4).ValuesAsNumpy(),
            "precipitation": hourly.Variables(5).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(hourly_data)
        return df
        
    except Exception as e:
        raise


def fetch_historical_aqi(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Karachi"
) -> pd.DataFrame:
    try:
        client = get_api_client()
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "pm10",
                "pm2_5",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "carbon_monoxide",
                "us_aqi",
            ],
            "domains": "auto",
            "timezone": timezone,
        }
        
        response = client.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                periods=len(hourly.Variables(0).ValuesAsNumpy()),
                freq=pd.DateOffset(hours=1),
            ),
            "pm10": hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(2).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(3).ValuesAsNumpy(),
            "ozone": hourly.Variables(4).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(5).ValuesAsNumpy(),
            "aqi": hourly.Variables(6).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(hourly_data)
        return df
        
    except Exception as e:
        raise


def fetch_weather_forecast(
    days: int,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Karachi"
) -> pd.DataFrame:
    try:
        client = get_api_client()
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
            ],
            "forecast_days": days,
            "timezone": timezone,
        }
        
        response = client.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                periods=len(hourly.Variables(0).ValuesAsNumpy()),
                freq=pd.DateOffset(hours=1),
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "pressure_msl": hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(3).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(4).ValuesAsNumpy(),
            "precipitation": hourly.Variables(5).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(hourly_data)
        return df
        
    except Exception as e:
        raise


def fetch_aqi_forecast(
    days: int,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Karachi"
) -> pd.DataFrame:
    try:
        client = get_api_client()
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "pm10",
                "pm2_5",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "carbon_monoxide",
                "us_aqi",
            ],
            "forecast_days": days,
            "domains": "auto",
            "timezone": timezone,
        }
        
        response = client.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                periods=len(hourly.Variables(0).ValuesAsNumpy()),
                freq=pd.DateOffset(hours=1),
            ),
            "pm10": hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(2).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(3).ValuesAsNumpy(),
            "ozone": hourly.Variables(4).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(5).ValuesAsNumpy(),
            "aqi": hourly.Variables(6).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(hourly_data)
        return df
        
    except Exception as e:
        raise
