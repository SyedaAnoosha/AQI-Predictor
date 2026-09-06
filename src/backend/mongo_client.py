"""MongoDB Atlas client - the project's primary feature store and model registry.

Responsibilities:
  * Feature storage (`aqi_historical_features`, `weather_forecast_features`)
    with idempotent, time-keyed upserts so re-running the hourly pipeline
    never duplicates rows.
  * Model registry (`model_registry`) with artifacts stored in GridFS, so a
    model trained on a GitHub Actions runner can be loaded by the API on
    Render without any shared filesystem.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional, Tuple

import gridfs
import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = os.getenv("MONGO_DB", "aqi_predictor")
HISTORICAL_COLLECTION = "aqi_historical_features"
FORECAST_COLLECTION = "weather_forecast_features"
REGISTRY_COLLECTION = "model_registry"

# Sidecar files bundled alongside a model pickle in the registry.
_ARTIFACT_FILES = ("feature_names.json", "metrics.json")

_INDEXES_ENSURED: set = set()


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

def connect_mongo(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    *,
    ensure_indexes: bool = True,
) -> Tuple[MongoClient, Any]:
    """Open a MongoDB connection and verify it is actually reachable.

    Unlike a bare ``MongoClient(uri)`` (which connects lazily and so appears to
    succeed even with a bad URI), this pings the server so callers fail fast
    and can fall back deliberately.
    """
    uri = uri or os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGODB_URI (or MONGO_URI) is not set")

    db_name = db_name or os.getenv("MONGO_DB") or DEFAULT_DB_NAME

    options: Dict[str, Any] = {
        "serverSelectionTimeoutMS": int(os.getenv("MONGO_TIMEOUT_MS", "20000")),
        "tz_aware": False,
        "retryWrites": True,
    }

    # Atlas requires TLS. Some hosts (Render's image among them) ship a CA
    # store the driver cannot use, which surfaces as
    # "TLSV1_ALERT_INTERNAL_ERROR" during the handshake. Pointing pymongo at
    # certifi's bundle gives it a CA set that is always present and current.
    if uri.startswith("mongodb+srv://") or "tls=true" in uri or "ssl=true" in uri:
        try:
            import certifi

            options["tlsCAFile"] = certifi.where()
        except ImportError:
            logger.warning("certifi is not installed; using the system CA store")

    client = MongoClient(uri, **options)
    client.admin.command("ping")
    db = client[db_name]

    if ensure_indexes:
        _ensure_indexes(db)

    logger.info("Connected to MongoDB database '%s'", db_name)
    return client, db


def _ensure_indexes(db) -> None:
    """Create the indexes the query patterns rely on. Idempotent and cached."""
    key = (id(db.client), db.name)
    if key in _INDEXES_ENSURED:
        return
    try:
        for name in (HISTORICAL_COLLECTION, FORECAST_COLLECTION):
            # Unique on time: makes upserts idempotent and range scans fast.
            db[name].create_index([("time", ASCENDING)], unique=True, name="time_unique")
        db[REGISTRY_COLLECTION].create_index(
            [("name", ASCENDING), ("version", DESCENDING)], name="name_version"
        )
        _INDEXES_ENSURED.add(key)
    except PyMongoError as exc:
        # Index creation can fail for a read-only user; queries still work.
        logger.warning("Could not ensure MongoDB indexes: %s", exc)


# --------------------------------------------------------------------------
# Time handling
# --------------------------------------------------------------------------

def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Normalise a time column to tz-naive UTC, the canonical storage form.

    Handles tz-naive input (assumed UTC), tz-aware input in any zone, and
    object columns holding a mix of both. The previous implementation called
    ``.dt.tz_convert`` directly and raised on tz-naive input.
    """
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_localize(None)


def _localize_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Re-attach UTC to the tz-naive times coming back out of MongoDB."""
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize("UTC")
    return df


def _clean_docs(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    df = _localize_utc(df)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Feature storage
# --------------------------------------------------------------------------

def insert_features(db, collection_name: str, df: pd.DataFrame) -> int:
    """Upsert feature rows keyed on ``time``.

    Returns the number of rows written. Using upserts rather than
    ``insert_many`` means the hourly pipeline can overlap its lookback window
    (it fetches 26h each run) without ever duplicating a timestamp.
    """
    if df is None or df.empty:
        logger.info("insert_features: nothing to write to '%s'", collection_name)
        return 0

    records = df.copy()
    if "time" not in records.columns:
        raise ValueError(f"DataFrame for '{collection_name}' has no 'time' column")

    records["time"] = _to_naive_utc(records["time"])
    records = records.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
    if records.empty:
        return 0

    # numpy scalars are not BSON-encodable; a JSON round-trip yields native
    # Python types for every column in one pass.
    docs = json.loads(records.to_json(orient="records", date_unit="s"))
    times = list(records["time"])

    now = datetime.now(dt_timezone.utc).replace(tzinfo=None)
    operations = []
    for doc, ts in zip(docs, times):
        doc["time"] = pd.Timestamp(ts).to_pydatetime()
        doc["updated_at"] = now
        operations.append(UpdateOne({"time": doc["time"]}, {"$set": doc}, upsert=True))

    _ensure_indexes(db)
    col = db[collection_name]
    try:
        result = col.bulk_write(operations, ordered=False)
        upserted, modified = result.upserted_count, result.modified_count
    except BulkWriteError as exc:
        logger.error("Bulk write to '%s' partially failed: %s", collection_name, exc.details)
        raise
    except TypeError:
        # Some driver/mock combinations reject the bulk UpdateOne signature;
        # fall back to individual upserts, which are equivalent but slower.
        logger.debug("bulk_write unsupported here; using per-document upserts")
        upserted = modified = 0
        for doc in docs:
            outcome = col.update_one({"time": doc["time"]}, {"$set": doc}, upsert=True)
            if outcome.upserted_id is not None:
                upserted += 1
            else:
                modified += outcome.modified_count

    written = upserted + modified
    logger.info(
        "Wrote %s rows to '%s' (%s new, %s updated)",
        written, collection_name, upserted, modified,
    )
    return written


def get_batch_data(
    db,
    start_time: str,
    end_time: str,
    collection_name: str = HISTORICAL_COLLECTION,
) -> pd.DataFrame:
    """Read observed feature rows in a closed time range."""
    start = _to_naive_utc(pd.Series([start_time])).iloc[0]
    end = _to_naive_utc(pd.Series([end_time])).iloc[0]
    rows = list(
        db[collection_name]
        .find({"time": {"$gte": start, "$lte": end}}, {"updated_at": 0})
        .sort("time", ASCENDING)
    )
    return _clean_docs(rows)


def get_forecast_features(
    db,
    start_time: str,
    end_time: str,
    collection_name: str = FORECAST_COLLECTION,
) -> pd.DataFrame:
    """Read forecast feature rows in a closed time range."""
    return get_batch_data(db, start_time, end_time, collection_name=collection_name)


def get_latest_features(
    db,
    n_hours: int = 72,
    collection_name: str = HISTORICAL_COLLECTION,
) -> pd.DataFrame:
    """Read the most recent ``n_hours`` of observed features."""
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=n_hours)
    rows = list(
        db[collection_name]
        .find({"time": {"$gte": cutoff.to_pydatetime()}}, {"updated_at": 0})
        .sort("time", ASCENDING)
    )
    return _clean_docs(rows)


def get_all_features(db, collection_name: str = HISTORICAL_COLLECTION) -> pd.DataFrame:
    """Read the full history - used by the training pipeline."""
    rows = list(db[collection_name].find({}, {"updated_at": 0}).sort("time", ASCENDING))
    return _clean_docs(rows)


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

def _bundle_model_files(models_dir: str, model_name: str) -> Optional[bytes]:
    """Zip a model's pickle plus its feature-name/metric sidecars."""
    # sklearn-style models ship a pickle; Keras models a .keras directory file.
    artifact = None
    for ext in (".pkl", ".keras"):
        candidate = os.path.join(models_dir, f"{model_name}{ext}")
        if os.path.exists(candidate):
            artifact = candidate
            break

    if artifact is None:
        logger.warning("No artifact for '%s' in %s; registering metadata only",
                       model_name, models_dir)
        return None

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact, arcname=os.path.basename(artifact))
        for sidecar in _ARTIFACT_FILES:
            # Prefer a model-specific sidecar, else the shared one.
            stem = sidecar.replace(".json", "")
            specific = os.path.join(models_dir, f"{stem}_{model_name}.json")
            shared = os.path.join(models_dir, sidecar)
            source = specific if os.path.exists(specific) else shared
            if os.path.exists(source):
                zf.write(source, arcname=sidecar)
    return buffer.getvalue()


def register_models_to_mongo(
    db,
    best_model_name: str,
    all_models_metrics: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    models_dir: Optional[str] = None,
) -> bool:
    """Publish trained models to the MongoDB registry.

    Artifacts go into GridFS so any process holding the connection string can
    load them. The previous ``filesystem_path`` approach only worked when
    training and serving shared a filesystem, which they never do here
    (training runs on GitHub Actions, serving on Render).
    """
    if models_dir is None:
        models_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models")
        )

    col = db[REGISTRY_COLLECTION]
    fs = gridfs.GridFS(db)
    saved_at = datetime.now(dt_timezone.utc).replace(tzinfo=None)
    best_key = (best_model_name or "").strip().lower().replace(" ", "_")

    for display_name, metrics in all_models_metrics.items():
        name = display_name.strip().lower().replace(" ", "_")

        float_metrics: Dict[str, Any] = {}
        for key, value in (metrics or {}).items():
            try:
                float_metrics[key] = float(value)
            except (TypeError, ValueError):
                float_metrics[key] = value

        # Version monotonically per model name.
        latest = col.find_one({"name": name}, sort=[("version", DESCENDING)])
        version = int(latest.get("version", 0)) + 1 if latest else 1

        doc: Dict[str, Any] = {
            "name": name,
            "display_name": display_name,
            "version": version,
            "best_model": name == best_key,
            "feature_names": feature_names,
            "metrics": float_metrics,
            "registry_source": "training_pipeline",
            "saved_at": saved_at,
        }

        bundle = _bundle_model_files(models_dir, name)
        if bundle is not None:
            doc["gridfs_id"] = fs.put(
                bundle,
                filename=f"{name}-v{version}.zip",
                model_name=name,
                version=version,
            )
            doc["artifact_sha256"] = hashlib.sha256(bundle).hexdigest()
            doc["artifact_bytes"] = len(bundle)

        col.update_one({"name": name, "version": version}, {"$set": doc}, upsert=True)
        logger.info(
            "Registered %s v%s to MongoDB (%s bytes)",
            name, version, doc.get("artifact_bytes", 0),
        )

    # Exactly one model carries the best flag.
    col.update_many({"name": {"$ne": best_key}}, {"$set": {"best_model": False}})
    return True


def load_model_from_registry_mongo(
    db, model_name: str = "lightgbm", version: Optional[int] = None
) -> Optional[dict]:
    """Fetch a model bundle from the registry and unpack it to a temp dir.

    Returns ``{'path': <dir>, 'registry_metrics': {...}, 'version': int}`` -
    the same shape the Hopsworks loader returns, so callers stay uniform.
    """
    col = db[REGISTRY_COLLECTION]
    query: Dict[str, Any] = {"name": model_name}
    if version is not None:
        query["version"] = version

    doc = col.find_one(query, sort=[("version", DESCENDING)])
    if not doc:
        logger.info("No registry entry for model '%s'", model_name)
        return None

    result = {
        "registry_metrics": doc.get("metrics", {}) or {},
        "version": doc.get("version"),
    }

    if doc.get("gridfs_id") is not None:
        fs = gridfs.GridFS(db)
        try:
            data = fs.get(doc["gridfs_id"]).read()
        except gridfs.NoFile:
            logger.error("GridFS object %s missing for %s", doc["gridfs_id"], model_name)
            return None

        tmpdir = tempfile.mkdtemp(prefix=f"model_{model_name}_")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(tmpdir)
        except zipfile.BadZipFile:
            # Older registry entries stored a bare pickle rather than a zip.
            with open(os.path.join(tmpdir, f"{model_name}.pkl"), "wb") as fh:
                fh.write(data)
        result["path"] = tmpdir
        return result

    # Legacy entries: a path only meaningful on the training machine.
    path = doc.get("filesystem_path")
    if path and os.path.exists(os.path.join(path, f"{model_name}.pkl")):
        result["path"] = path
        return result

    logger.warning("Registry entry for '%s' has no loadable artifact", model_name)
    return None


def get_all_model_metrics_mongo(db) -> Dict[str, Dict[str, Any]]:
    """Latest metrics for every registered model."""
    metrics: Dict[str, Dict[str, Any]] = {}
    for doc in db[REGISTRY_COLLECTION].find().sort("version", ASCENDING):
        name = doc.get("name") or doc.get("model_name")
        if name:
            # Later (higher) versions overwrite earlier ones.
            metrics[name] = doc.get("metrics", {}) or {}
    return metrics


def get_best_model_name(db) -> Optional[str]:
    """The model the most recent training run flagged as best."""
    doc = db[REGISTRY_COLLECTION].find_one(
        {"best_model": True}, sort=[("saved_at", DESCENDING)]
    )
    return doc.get("name") if doc else None
