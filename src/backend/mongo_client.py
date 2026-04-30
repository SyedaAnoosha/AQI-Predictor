import os
import tempfile
import json
import base64
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
from datetime import datetime

from pymongo import MongoClient
import gridfs


def connect_mongo(uri: str, db_name: str = "aqi_predictor") -> Tuple[MongoClient, object]:
    client = MongoClient(uri)
    db = client[db_name]
    return client, db


def _ensure_time_dt(record: dict, time_field: str = "time"):
    if time_field in record and not isinstance(record[time_field], datetime):
        try:
            record[time_field] = pd.to_datetime(record[time_field])
        except Exception:
            pass
    return record


def insert_features(db, collection_name: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    col = db[collection_name]
    records = df.copy()
    # Convert numpy types to native
    records['time'] = pd.to_datetime(records['time']).dt.tz_convert('UTC').dt.tz_localize(None)
    recs = records.to_dict(orient='records')
    for r in recs:
        _ensure_time_dt(r, 'time')
    col.insert_many(recs)


def get_forecast_features(db, start_time: str, end_time: str, collection_name: str = 'weather_forecast_features') -> pd.DataFrame:
    col = db[collection_name]
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    cursor = col.find({'time': {'$gte': start, '$lte': end}}).sort('time', 1)
    rows = list(cursor)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
    return df


def get_batch_data(db, start_time: str, end_time: str, collection_name: str = 'aqi_historical_features') -> pd.DataFrame:
    col = db[collection_name]
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    cursor = col.find({'time': {'$gte': start, '$lte': end}}).sort('time', 1)
    rows = list(cursor)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
    return df


def get_latest_features(db, n_hours: int = 72, collection_name: str = 'aqi_historical_features') -> pd.DataFrame:
    col = db[collection_name]
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=n_hours)
    cursor = col.find({'time': {'$gte': cutoff}}).sort('time', 1)
    rows = list(cursor)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
    return df


def load_model_from_registry_mongo(db, model_name: str = 'lightgbm') -> Optional[dict]:
    """
    Look for a model document in collection `model_registry` with fields:
      - name
      - version (int)
      - filesystem_path (optional) -> path on disk
      - model_blob (optional) -> base64 encoded zip/tar of model dir
      - metrics (optional)

    Returns similar dict to Hopsworks loader: { 'path': <dir>, 'registry_metrics': {...} }
    """
    col = db['model_registry']
    try:
        doc = col.find_one({'name': model_name}, sort=[('version', -1)])
        if not doc:
            return None
        metrics = doc.get('metrics', {})
        if 'filesystem_path' in doc and os.path.exists(doc['filesystem_path']):
            return {'path': doc['filesystem_path'], 'registry_metrics': metrics}

        if 'model_blob' in doc:
            # decode blob and extract into temp dir
            b64 = doc['model_blob']
            data = base64.b64decode(b64)
            # assume blob is a pickle file or zip; try to write single file
            tmpdir = tempfile.mkdtemp(prefix='model_')
            model_file = os.path.join(tmpdir, f"{model_name}.pkl")
            with open(model_file, 'wb') as f:
                f.write(data)
            return {'path': tmpdir, 'registry_metrics': metrics}

        # GridFS support: check for gridfs_id
        if 'gridfs_id' in doc:
            fs = gridfs.GridFS(db)
            grid_id = doc['gridfs_id']
            try:
                grid_out = fs.get(grid_id)
                data = grid_out.read()
                tmpdir = tempfile.mkdtemp(prefix='model_')
                model_file = os.path.join(tmpdir, f"{model_name}.pkl")
                with open(model_file, 'wb') as f:
                    f.write(data)
                return {'path': tmpdir, 'registry_metrics': metrics}
            except Exception:
                return None

        return None
    except Exception:
        return None


def register_models_to_mongo(
    db,
    best_model_name: str,
    all_models_metrics: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    models_dir: Optional[str] = None,
):
    """Persist model registry metadata to MongoDB Atlas.

    Stores one document per model in `model_registry` so the app can later
    resolve model artifacts from the local `models/` directory.
    """
    if models_dir is None:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

    col = db['model_registry']
    saved_at = datetime.utcnow()

    for model_name, metrics in all_models_metrics.items():
        normalized_name = model_name.strip().lower().replace(' ', '_')
        doc = {
            'name': normalized_name,
            'display_name': model_name,
            'version': 1,
            'best_model': model_name == best_model_name,
            'filesystem_path': models_dir,
            'feature_names': feature_names,
            'metrics': metrics,
            'registry_source': 'training_pipeline',
            'saved_at': saved_at,
        }
        col.update_one(
            {'name': normalized_name, 'version': 1},
            {'$set': doc},
            upsert=True,
        )

    return True
