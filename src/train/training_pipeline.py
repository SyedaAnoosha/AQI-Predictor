import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import time
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.backend.hopsworks_client import connect_hopsworks, get_feature_view
from src.backend.services import generate_forecast
from src.features.feature_engineering import prepare_for_training


CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'cache'))
PREDICTION_CACHE_PATH = os.path.join(CACHE_DIR, 'predictions_72h.json')
METRICS_CACHE_PATH = os.path.join(CACHE_DIR, 'all_model_metrics.json')
BEST_MODEL_META_PATH = os.path.join(CACHE_DIR, 'best_model_meta.json')
LATITUDE =25.3792
LONGITUDE = 68.3683
TIMEZONE = os.getenv('TIMEZONE', 'Asia/Karachi')


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _save_json(payload: dict, path: str) -> None:
    """Persist JSON payload with safe defaults for datetimes."""
    _ensure_cache_dir()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)


def load_data_from_hopsworks(api_key, project_name):
    project, fs = connect_hopsworks(api_key, project_name)
    fg = fs.get_feature_group(name="aqi_features", version=1)
    fv = get_feature_view(fs, name="aqi_feature_view", version=1)
    df = fg.read()
    
    X, y = prepare_for_training(df, target_col='aqi')
    
    if 'season' in X.columns:
        X = pd.get_dummies(X, columns=['season'], prefix='season', drop_first=False)
    
    return X, y, df, project


def create_time_series_splits(X, y, n_splits=5, test_size=0.10):
    n_total = len(X)
    n_test = int(n_total * test_size)
    n_train_val = n_total - n_test
    
    X_train_val = X.iloc[:n_train_val]
    y_train_val = y.iloc[:n_train_val]
    X_test = X.iloc[n_train_val:]
    y_test = y.iloc[n_train_val:]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_train_val))
    train_idx, val_idx = splits[-1]
    
    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, tscv


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_random_forest(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=800, max_depth=20, min_samples_split=5,
        min_samples_leaf=10, random_state=42, n_jobs=-1, verbose=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.02, max_depth=3, min_child_weight=10,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=3.0,
        gamma=0.25, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=10, num_leaves=32,
        min_child_samples=20, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, n_jobs=-1,
        early_stopping_round=100, verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_elasticnet(X_train_scaled, y_train, X_val_scaled, y_val):
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_tensorflow(X_train_scaled, y_train, X_val_scaled, y_val):
    input_dim = X_train_scaled.shape[1]
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    
    model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
              epochs=100, batch_size=32, verbose=0,
              callbacks=[
                  keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                               restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                   patience=5, min_lr=1e-6)
              ])
    
    y_pred = model.predict(X_val_scaled, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}


def select_best_model(models_dict, metrics_dict):
    comparison_data = []
    for name, metrics in metrics_dict.items():
        comparison_data.append({
            'Model': name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2']
        })

    comparison_df = pd.DataFrame(comparison_data)

    filtered = [row for row in comparison_data if not np.isclose(row['R²'], 1.0)]
    if not filtered:
        filtered = comparison_data

    ranked = sorted(
        filtered,
        key=lambda r: (r['RMSE'], r['MAE'], -r['R²'], r['Model'])
    )

    best_model_name = ranked[0]['Model']
    best_model = models_dict[best_model_name]
    best_metrics = metrics_dict[best_model_name]

    return best_model_name, best_model, best_metrics


def evaluate_on_test(model, X_test, y_test, model_name, use_scaled=False, X_test_scaled=None):
    X_input = X_test_scaled if use_scaled else X_test
    
    if model_name == 'TensorFlow NN':
        y_pred = model.predict(X_input, verbose=0).flatten()
    else:
        y_pred = model.predict(X_input)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def analyze_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names, 'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance_df


def save_model_artifacts(model, scaler, feature_names, metrics, model_name, models_dir='../../models'):
    models_path = os.path.join(os.path.dirname(__file__), models_dir)
    os.makedirs(models_path, exist_ok=True)
    
    model_filename = f'{model_name.lower().replace(" ", "_")}_model.pkl'
    with open(os.path.join(models_path, model_filename), 'wb') as f:
        pickle.dump(model, f)
    
    if scaler is not None:
        with open(os.path.join(models_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
    with open(os.path.join(models_path, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    metrics_payload = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()}
    metrics_payload.update({
        'model_name': model_name,
        'saved_at': datetime.now().isoformat()
    })
    with open(os.path.join(models_path, 'metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=2)


def cache_daily_outputs(best_model_name: str, all_models_metrics: dict, prediction_response) -> None:
    selection_meta = {
        'best_model': best_model_name,
        'selected_at': datetime.now().isoformat(),
        'selection_rule': 'lowest val_rmse, then lowest val_mae, then highest val_r2 (excluding val_r2 == 1.0)',
        'lat': LATITUDE,
        'lon': LONGITUDE,
        'timezone': TIMEZONE
    }
    _save_json(selection_meta, BEST_MODEL_META_PATH)
    _save_json(all_models_metrics, METRICS_CACHE_PATH)

    if prediction_response is not None:
        payload = prediction_response.model_dump() if hasattr(prediction_response, 'model_dump') else prediction_response.dict()
        _save_json(payload, PREDICTION_CACHE_PATH)


def register_models_to_hopsworks(project, best_model_name, all_models_metrics, feature_names):
    try:
        mr = project.get_model_registry()
        for model_name, metrics in all_models_metrics.items():
            try:
                if model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
                    model_module = mr.sklearn
                elif model_name == 'TensorFlow NN':
                    model_module = mr.tensorflow
                else:
                    model_module = mr.sklearn
                
                model_meta = model_module.create_model(
                    name=model_name.lower().replace(' ', '_'),
                    metrics={k: float(v) for k, v in metrics.items()},
                )
                model_dir = os.path.join(os.path.dirname(__file__), '../../models')
                model_meta.save(model_dir)
            except Exception:
                pass
        return mr
    except Exception:
        return None


def run_training_and_inference():
    from dotenv import load_dotenv
    load_dotenv()

    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
    HOPSWORKS_PROJECT = os.getenv('HOPSWORKS_PROJECT')

    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        return

    X, y, df, project = load_data_from_hopsworks(HOPSWORKS_API_KEY, HOPSWORKS_PROJECT)
    X_train, X_val, X_test, y_train, y_val, y_test, tscv = create_time_series_splits(X, y, n_splits=5)

    pollutant_patterns = ['pm2_5', 'pm10', 'carbon_monoxide', 'sulphur_dioxide', 'nitrogen_dioxide']
    cols_to_drop = []

    for col in X_train.columns:
        if col in pollutant_patterns:
            cols_to_drop.append(col)
        for pattern in pollutant_patterns:
            if pattern in col and 'lag' not in col and col not in cols_to_drop:
                cols_to_drop.append(col)
                break

    if 'is_duplicate' in X_train.columns:
        cols_to_drop.append('is_duplicate')

    if cols_to_drop:
        X_train = X_train.drop(columns=cols_to_drop)
        X_val = X_val.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    models = {}
    metrics = {}

    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = rf_model
    metrics['Random Forest'] = rf_metrics

    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    models['XGBoost'] = xgb_model
    metrics['XGBoost'] = xgb_metrics

    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
    models['LightGBM'] = lgb_model
    metrics['LightGBM'] = lgb_metrics

    en_model, en_metrics = train_elasticnet(X_train_scaled, y_train, X_val_scaled, y_val)
    models['ElasticNet'] = en_model
    metrics['ElasticNet'] = en_metrics

    tf_model, tf_metrics = train_tensorflow(X_train_scaled, y_train, X_val_scaled, y_val)
    models['TensorFlow NN'] = tf_model
    metrics['TensorFlow NN'] = tf_metrics

    best_model_name, best_model, best_val_metrics = select_best_model(models, metrics)

    train_metrics_all = {}
    for model_name, model in models.items():
        use_scaled = (model_name in ['ElasticNet', 'TensorFlow NN'])
        X_train_input = X_train_scaled if use_scaled else X_train

        if model_name == 'TensorFlow NN':
            y_train_pred = model.predict(X_train_input, verbose=0).flatten()
        else:
            y_train_pred = model.predict(X_train_input)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        train_metrics_all[model_name] = {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2}

    use_scaled = (best_model_name in ['ElasticNet', 'TensorFlow NN'])
    test_metrics = evaluate_on_test(best_model, X_test, y_test, best_model_name,
                                     use_scaled=use_scaled, X_test_scaled=X_test_scaled)

    test_metrics_all = {}
    for model_name, model in models.items():
        use_scaled = (model_name in ['ElasticNet', 'TensorFlow NN'])
        X_test_input = X_test_scaled if use_scaled else X_test

        if model_name == 'TensorFlow NN':
            y_test_pred = model.predict(X_test_input, verbose=0).flatten()
        else:
            y_test_pred = model.predict(X_test_input)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        test_metrics_all[model_name] = {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}

    if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        analyze_feature_importance(best_model, list(X_train.columns))

    save_scaler = scaler if best_model_name in ['ElasticNet', 'TensorFlow NN'] else None

    all_models_metrics = {}
    for model_name in models.keys():
        val_metrics = metrics[model_name]
        all_models_metrics[model_name] = {
            'train_rmse': float(train_metrics_all[model_name]['rmse']),
            'train_mae': float(train_metrics_all[model_name]['mae']),
            'train_r2': float(train_metrics_all[model_name]['r2']),
            'val_rmse': float(val_metrics['rmse']),
            'val_mae': float(val_metrics['mae']),
            'val_r2': float(val_metrics['r2']),
            'test_rmse': float(test_metrics_all[model_name]['rmse']),
            'test_mae': float(test_metrics_all[model_name]['mae']),
            'test_r2': float(test_metrics_all[model_name]['r2']),
        }

    save_model_artifacts(best_model, save_scaler, list(X_train.columns),
                        all_models_metrics[best_model_name], best_model_name)

    register_models_to_hopsworks(project, best_model_name, all_models_metrics, list(X_train.columns))

    best_artifacts = {
        'model': best_model,
        'feature_names': list(X_train.columns),
        'metrics': all_models_metrics[best_model_name]
    }

    prediction_response = None
    try:
        prediction_response = generate_forecast(
            best_artifacts,
            hours=72,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            timezone=TIMEZONE
        )
    except Exception:
        pass

    cache_daily_outputs(best_model_name, all_models_metrics, prediction_response)

if __name__ == "__main__":
    run_training_and_inference()
