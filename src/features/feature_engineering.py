# Feature engineering module for AQI Predictor project
import pandas as pd
import numpy as np
from typing import Tuple

def _causal_rolling_median(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Compute rolling median using only past values (backward-looking).
    Prevents look-ahead bias by excluding current and future values.
    """
    return series.rolling(window=window, min_periods=1).median()

def handle_missing_values(df: pd.DataFrame, use_causal: bool = True) -> pd.DataFrame:
    """
    Handle missing values with optional causal (backward-only) imputation.
    
    WARNING: If use_causal=False, this applies non-causal median which can cause
    look-ahead bias. Only use use_causal=False on data that has already been split!
    """
    df_clean = df.copy()

    if 'time' in df_clean.columns:
        df_clean = df_clean.sort_values('time').reset_index(drop=True)
    
    weather_vars = ['temperature_2m', 'pressure_msl', 'wind_speed_10m']
    for var in weather_vars:
        if var in df_clean.columns:
            df_clean[var] = df_clean[var].ffill(limit=3)
            df_clean[var] = df_clean[var].interpolate(method='linear')
    
    pollutant_vars = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
    for var in pollutant_vars:
        if var in df_clean.columns:
            df_clean[var] = df_clean[var].ffill(limit=2)
            # Use backward-only rolling median to prevent look-ahead bias
            if use_causal:
                causal_median = _causal_rolling_median(df_clean[var], window=12)
            else:
                causal_median = df_clean[var].rolling(window=12, min_periods=1).median()
            df_clean[var] = df_clean[var].fillna(causal_median)

    if 'aqi' in df_clean.columns:
        df_clean['aqi'] = df_clean['aqi'].ffill(limit=2)
        if use_causal:
            aqi_causal_median = _causal_rolling_median(df_clean['aqi'], window=12)
        else:
            aqi_causal_median = df_clean['aqi'].rolling(window=12, min_periods=1).median()
        df_clean['aqi'] = df_clean['aqi'].fillna(aqi_causal_median)
    
    return df_clean

def detect_duplicate_weather(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    df_dup = df.copy()
    weather_features = ['temperature_2m', 'pressure_msl', 'wind_speed_10m']
    is_duplicate = pd.Series(False, index=df_dup.index)
    
    for i in range(1, len(df_dup)):
        diffs = []
        for feature in weather_features:
            if feature in df_dup.columns:
                diff = abs(df_dup[feature].iloc[i] - df_dup[feature].iloc[i-1])
                feature_range = df_dup[feature].max() - df_dup[feature].min()
                if feature_range > 0:
                    diffs.append(diff / feature_range)
        
        if len(diffs) > 0 and all(d < threshold for d in diffs):
            is_duplicate.iloc[i] = True
    
    df_dup['is_duplicate'] = is_duplicate
    return df_dup

def validate_data_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df.copy()
    
    if 'aqi' in df_valid.columns:
        df_valid = df_valid[(df_valid['aqi'] >= 0) & (df_valid['aqi'] <= 600)]
    
    if 'temperature_2m' in df_valid.columns:
        df_valid = df_valid[(df_valid['temperature_2m'] >= -40) & (df_valid['temperature_2m'] <= 60)]
    
    if 'pm2_5' in df_valid.columns and 'pm10' in df_valid.columns:
        df_valid = df_valid[df_valid['pm2_5'] <= df_valid['pm10']]
    
    return df_valid

def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df_time = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_time['time']):
        df_time['time'] = pd.to_datetime(df_time['time'])
    
    df_time['hour'] = df_time['time'].dt.hour
    df_time['day_of_week'] = df_time['time'].dt.dayofweek
    df_time['day_of_month'] = df_time['time'].dt.day
    df_time['month'] = df_time['time'].dt.month
    df_time['quarter'] = df_time['time'].dt.quarter
    df_time['week_of_year'] = df_time['time'].dt.isocalendar().week
    df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype('int32')
    df_time['is_daytime'] = ((df_time['hour'] >= 6) & (df_time['hour'] < 18)).astype('int32')
    df_time['season'] = df_time['month'].apply(get_season)
    
    return df_time

def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df_cyc = df.copy()
    
    if 'hour' in df_cyc.columns:
        df_cyc['hour_sin'] = np.sin(2 * np.pi * df_cyc['hour'] / 24)
        df_cyc['hour_cos'] = np.cos(2 * np.pi * df_cyc['hour'] / 24)
    
    if 'day_of_week' in df_cyc.columns:
        df_cyc['day_of_week_sin'] = np.sin(2 * np.pi * df_cyc['day_of_week'] / 7)
        df_cyc['day_of_week_cos'] = np.cos(2 * np.pi * df_cyc['day_of_week'] / 7)
    
    if 'month' in df_cyc.columns:
        df_cyc['month_sin'] = np.sin(2 * np.pi * df_cyc['month'] / 12)
        df_cyc['month_cos'] = np.cos(2 * np.pi * df_cyc['month'] / 12)
    
    return df_cyc

def create_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """Create lag features for time series prediction.
    
    Based on EDA lag analysis (ACF/PACF, cross-correlation, Mutual Information):
    - pm2_5: lags [1,3,6,12,24] — strong predictor peaking at 12h (r=0.836, MI=0.68)
    - carbon_monoxide: lags [1,3,6,12,24] — moderate predictor stable across lags
    - temperature_2m: lag [12h] only — diurnal cycle effect (r=-0.483, MI=0.197)
    - pressure_msl: lag [12h] only — weak linear but strong non-linear (MI=0.321)
    """
    df_lagged = df.copy()
    
    # Strong pollutant predictors: use full lag set [1,3,6,12,24]
    if 'pm2_5' in df_lagged.columns:
        for lag in [1, 3, 6, 12, 24]:
            df_lagged[f'pm2_5_lag_{lag}h'] = df_lagged['pm2_5'].shift(lag)
    
    if 'carbon_monoxide' in df_lagged.columns:
        for lag in [1, 3, 6, 12, 24]:
            df_lagged[f'carbon_monoxide_lag_{lag}h'] = df_lagged['carbon_monoxide'].shift(lag)
    
    # Weather predictors: use only 12h lag (peak predictive power)
    if 'temperature_2m' in df_lagged.columns:
        df_lagged['temperature_2m_lag_12h'] = df_lagged['temperature_2m'].shift(12)
    
    if 'pressure_msl' in df_lagged.columns:
        df_lagged['pressure_msl_lag_12h'] = df_lagged['pressure_msl'].shift(12)
    
    return df_lagged

def create_aqi_change_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create AQI momentum/trend features using ONLY past values.
    
    CRITICAL: All changes use shift(1) as the baseline (most recent known AQI)
    to prevent target leakage. aqi_change_Nh = aqi[t-1] - aqi[t-1-N],
    i.e. how much AQI changed in the N hours ending at the last observation.
    
    Previous (LEAKY) version used: aqi[t] - aqi[t-N], which encoded the target.
    """
    df_aqi_change = df.copy()
    
    if 'aqi' not in df_aqi_change.columns:
        return df_aqi_change
    
    # Use shift(1) as baseline to prevent target leakage
    # aqi_change_1h = aqi[t-1] - aqi[t-2]  (1h change ending at last observation)
    # aqi_change_3h = aqi[t-1] - aqi[t-4]  (3h change ending at last observation)
    aqi_prev = df_aqi_change['aqi'].shift(1)
    df_aqi_change['aqi_change_1h'] = aqi_prev - df_aqi_change['aqi'].shift(2)
    df_aqi_change['aqi_change_3h'] = aqi_prev - df_aqi_change['aqi'].shift(4)
    df_aqi_change['aqi_change_6h'] = aqi_prev - df_aqi_change['aqi'].shift(7)
    df_aqi_change['aqi_change_24h'] = aqi_prev - df_aqi_change['aqi'].shift(25)
    
    df_aqi_change['aqi_rate_1h'] = df_aqi_change['aqi_change_1h'] / 1.0
    df_aqi_change['aqi_rate_3h'] = df_aqi_change['aqi_change_3h'] / 3.0
    df_aqi_change['aqi_rate_24h'] = df_aqi_change['aqi_change_24h'] / 24.0
    
    for col in ['aqi_rate_1h', 'aqi_rate_3h', 'aqi_rate_24h']:
        df_aqi_change[col] = df_aqi_change[col].fillna(0).clip(-10, 10)
    
    return df_aqi_change

def create_rate_of_change_features(df: pd.DataFrame, include_aqi_rate: bool = False) -> pd.DataFrame:
    """Create rate-of-change features using ONLY past values (no target leakage)."""
    df_rate = df.copy()
    
    if include_aqi_rate and 'aqi' in df_rate.columns:
        # Use shift(1) as baseline to prevent target leakage
        aqi_prev = df_rate['aqi'].shift(1)
        df_rate['aqi_change_1h'] = aqi_prev - df_rate['aqi'].shift(2)
        df_rate['aqi_change_3h'] = aqi_prev - df_rate['aqi'].shift(4)
        df_rate['aqi_change_24h'] = aqi_prev - df_rate['aqi'].shift(25)
        
        # pct_change also shifted to use only past values
        df_rate['aqi_pct_change_1h'] = aqi_prev.pct_change(periods=1).fillna(0).clip(-1, 1)
        df_rate['aqi_pct_change_24h'] = aqi_prev.pct_change(periods=24).fillna(0).clip(-1, 1)
    
    return df_rate

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df_inter = df.copy()
    
    if 'temperature_2m' in df_inter.columns and 'relative_humidity_2m' in df_inter.columns:
        df_inter['temp_humidity_interaction'] = (
            df_inter['temperature_2m'] * df_inter['relative_humidity_2m']
        )
    
    if 'wind_speed_10m' in df_inter.columns and 'pressure_msl' in df_inter.columns:
        df_inter['wind_pressure_interaction'] = (
            df_inter['wind_speed_10m'] * df_inter['pressure_msl']
        )
    
    return df_inter

def process_features(
    df: pd.DataFrame,
    include_lags: bool = True,
    include_aqi_rate: bool = False,
    include_aqi_change_rate: bool = True,
    use_causal_imputation: bool = True
) -> pd.DataFrame:
    df = handle_missing_values(df, use_causal=use_causal_imputation)
    df = detect_duplicate_weather(df)
    df = validate_data_ranges(df)
    
    df = extract_time_features(df)
    df = create_cyclical_features(df)
    
    if include_lags:
        df = create_lag_features(df)
    
    if include_aqi_change_rate:
        df = create_aqi_change_rate_features(df)
    
    if include_aqi_rate:
        df = create_rate_of_change_features(df, include_aqi_rate=True)
    
    df = create_interaction_features(df)
    
    if include_lags or include_aqi_change_rate:
        df = df.dropna()
    
    return df

def process_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare forecast-only features from weather data and time fields."""
    df = handle_missing_values(df)
    df = extract_time_features(df)
    df = create_cyclical_features(df)
    df = create_interaction_features(df)
    return df

def prepare_for_training(df: pd.DataFrame, target_col: str = 'aqi') -> Tuple[pd.DataFrame, pd.Series]:
    df_model = df.copy()
    
    if 'time' in df_model.columns:
        df_model = df_model.drop('time', axis=1)
    
    # Remove ALL same-hour pollutants (data leakage prevention)
    same_hour_pollutants = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide']
    cols_to_remove = [col for col in same_hour_pollutants if col in df_model.columns]
    
    if 'pm2_5_pm10_ratio' in df_model.columns:
        cols_to_remove.append('pm2_5_pm10_ratio')
    
    if 'is_duplicate' in df_model.columns:
        cols_to_remove.append('is_duplicate')
    
    if cols_to_remove:
        df_model = df_model.drop(columns=cols_to_remove)
    
    if 'season' in df_model.columns:
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        df_model['season'] = df_model['season'].map(season_map)
        df_model = pd.get_dummies(df_model, columns=['season'], prefix='season', drop_first=False)
        
        for i in range(4):
            col_name = f'season_{i}'
            if col_name not in df_model.columns:
                df_model[col_name] = 0
    
    y = df_model[target_col]
    X = df_model.drop(target_col, axis=1)
    
    return X, y

def prepare_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df_pred = df.copy()
    
    cols_to_drop = [col for col in ['time', 'aqi'] if col in df_pred.columns]
    if cols_to_drop:
        df_pred = df_pred.drop(columns=cols_to_drop)
    
    # Remove ALL same-hour pollutants (data leakage prevention)
    same_hour_pollutants = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide']
    cols_to_remove = [col for col in same_hour_pollutants if col in df_pred.columns]
    
    if 'pm2_5_pm10_ratio' in df_pred.columns:
        cols_to_remove.append('pm2_5_pm10_ratio')
    
    if 'is_duplicate' in df_pred.columns:
        cols_to_remove.append('is_duplicate')
    
    if cols_to_remove:
        df_pred = df_pred.drop(columns=cols_to_remove)
    
    if 'season' in df_pred.columns:
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        df_pred['season'] = df_pred['season'].map(season_map)
        df_pred = pd.get_dummies(df_pred, columns=['season'], prefix='season', drop_first=False)
        
        for i in range(4):
            col_name = f'season_{i}'
            if col_name not in df_pred.columns:
                df_pred[col_name] = 0
    
    return df_pred
