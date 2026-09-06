"""Unified storage layer for features and models.

The project supports two backends. **MongoDB Atlas is the primary store**;
Hopsworks is the secondary, tried only when Mongo is unconfigured or fails.
A local `models/` directory is the last-resort read-only fallback.

Everything above this module (API, pipelines, training) talks to the
``FeatureStore`` / ``ModelRegistry`` facades rather than to a specific vendor
client, so the priority order lives in exactly one place instead of being
re-implemented as nested try/except blocks in every caller.

Priority is configurable with ``STORAGE_BACKEND``:
    auto      - Mongo, then Hopsworks, then disk (default)
    mongo     - Mongo only
    hopsworks - Hopsworks only
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

HISTORICAL_COLLECTION = "aqi_historical_features"
FORECAST_COLLECTION = "weather_forecast_features"


def _backend_preference() -> List[str]:
    """Ordered list of backends to attempt, honouring STORAGE_BACKEND."""
    mode = os.getenv("STORAGE_BACKEND", "auto").strip().lower()
    if mode == "mongo":
        return ["mongo"]
    if mode == "hopsworks":
        return ["hopsworks"]
    # Default: MongoDB first, Hopsworks second.
    return ["mongo", "hopsworks"]


def mongo_uri() -> Optional[str]:
    """The Mongo connection string.

    Accepts either MONGODB_URI (the spelling the Atlas console suggests) or
    MONGO_URI, so an existing .env works without renaming anything.
    """
    return os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")


def _mongo_configured() -> bool:
    return bool(mongo_uri())


def _hopsworks_configured() -> bool:
    return bool(os.getenv("HOPSWORKS_API_KEY"))


# ==========================================================================
# Feature store
# ==========================================================================

class FeatureStore:
    """Reads and writes feature rows, MongoDB first.

    Connections are established lazily and cached for the life of the
    instance, so a long-lived API process does not reconnect per request.
    """

    def __init__(self) -> None:
        self._mongo_db = None
        self._mongo_failed = False
        self._hopsworks_fs = None
        self._hopsworks_project = None
        self._hopsworks_failed = False

    # -- connections -------------------------------------------------------

    def mongo(self):
        """The Mongo database handle, or None if unavailable."""
        if self._mongo_db is not None or self._mongo_failed:
            return self._mongo_db
        if not _mongo_configured():
            self._mongo_failed = True
            return None
        try:
            from backend.mongo_client import connect_mongo

            _, self._mongo_db = connect_mongo(mongo_uri())
        except Exception as exc:
            logger.warning("MongoDB unavailable: %s", exc)
            self._mongo_failed = True
        return self._mongo_db

    def hopsworks(self):
        """The Hopsworks feature store handle, or None if unavailable."""
        if self._hopsworks_fs is not None or self._hopsworks_failed:
            return self._hopsworks_fs
        if not _hopsworks_configured():
            self._hopsworks_failed = True
            return None
        try:
            # Imported lazily: the hopsworks package is optional, and a
            # MongoDB-only deployment need not install it.
            from backend.hopsworks_client import connect_hopsworks

            # _hopsworks_configured() has already established the key is set.
            api_key = os.getenv("HOPSWORKS_API_KEY") or ""
            self._hopsworks_project, self._hopsworks_fs = connect_hopsworks(
                api_key, os.getenv("HOPSWORKS_PROJECT") or "aqi_predictor"
            )
        except ImportError:
            logger.info("hopsworks package not installed; skipping that backend")
            self._hopsworks_failed = True
            return None
        except Exception as exc:
            logger.warning("Hopsworks unavailable: %s", exc)
            self._hopsworks_failed = True
        return self._hopsworks_fs

    @property
    def hopsworks_project(self):
        self.hopsworks()
        return self._hopsworks_project

    def active_backend(self) -> str:
        """Which backend reads will actually use right now."""
        for backend in _backend_preference():
            if backend == "mongo" and self.mongo() is not None:
                return "mongo"
            if backend == "hopsworks" and self.hopsworks() is not None:
                return "hopsworks"
        return "none"

    # -- writes ------------------------------------------------------------

    def write_features(self, collection: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Write features to every configured backend.

        Writes fan out rather than falling back: keeping both stores in sync
        is the point when both are configured. The result reports each
        backend's outcome so the pipeline can log and exit meaningfully.
        """
        results: Dict[str, Any] = {}
        if df is None or df.empty:
            return {"skipped": "empty dataframe"}

        for backend in _backend_preference():
            if backend == "mongo":
                db = self.mongo()
                if db is None:
                    continue
                try:
                    from backend.mongo_client import insert_features

                    results["mongo"] = insert_features(db, collection, df)
                except Exception as exc:
                    logger.error("Mongo write to '%s' failed: %s", collection, exc)
                    results["mongo"] = f"error: {exc}"

            elif backend == "hopsworks":
                fs = self.hopsworks()
                if fs is None:
                    continue
                try:
                    from backend.hopsworks_client import (
                        create_feature_group,
                        insert_features as hopsworks_insert,
                    )

                    fg = create_feature_group(
                        fs, name=collection, version=1,
                        primary_key=["time"], event_time="time",
                    )
                    hopsworks_insert(fg, df)
                    results["hopsworks"] = len(df)
                except Exception as exc:
                    logger.error("Hopsworks write to '%s' failed: %s", collection, exc)
                    results["hopsworks"] = f"error: {exc}"

        return results

    # -- reads -------------------------------------------------------------

    def read_range(
        self, collection: str, start_time: str, end_time: str
    ) -> pd.DataFrame:
        """Read rows in a time range from the first backend that returns data."""
        for backend in _backend_preference():
            try:
                if backend == "mongo":
                    db = self.mongo()
                    if db is None:
                        continue
                    from backend.mongo_client import get_batch_data

                    df = get_batch_data(db, start_time, end_time, collection_name=collection)
                elif backend == "hopsworks":
                    fs = self.hopsworks()
                    if fs is None:
                        continue
                    from backend.hopsworks_client import get_batch_data as hops_batch
                    from backend.hopsworks_client import get_forecast_features as hops_forecast

                    if collection == FORECAST_COLLECTION:
                        df = hops_forecast(fs, start_time, end_time)
                    else:
                        df = hops_batch(fs, start_time, end_time)
                else:
                    continue

                if df is not None and not df.empty:
                    logger.info("Read %s rows from %s via %s", len(df), collection, backend)
                    return df
            except ImportError as exc:
                logger.debug("Backend %s not installed: %s", backend, exc)
            except Exception as exc:
                logger.warning("Read from %s failed for '%s': %s", backend, collection, exc)

        logger.warning("No data for '%s' in %s..%s from any backend",
                       collection, start_time, end_time)
        return pd.DataFrame()

    def read_forecast(self, start_time: str, end_time: str) -> pd.DataFrame:
        return self.read_range(FORECAST_COLLECTION, start_time, end_time)

    def read_all_historical(self) -> pd.DataFrame:
        """Full observed history - used for training."""
        for backend in _backend_preference():
            try:
                if backend == "mongo":
                    db = self.mongo()
                    if db is None:
                        continue
                    from backend.mongo_client import get_all_features

                    df = get_all_features(db, HISTORICAL_COLLECTION)
                elif backend == "hopsworks":
                    fs = self.hopsworks()
                    if fs is None:
                        continue
                    fg = fs.get_feature_group(HISTORICAL_COLLECTION, version=1)
                    df = fg.read()
                    if df is not None and "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"], utc=True)
                        df = df.sort_values("time").reset_index(drop=True)
                else:
                    continue

                if df is not None and not df.empty:
                    logger.info("Loaded %s historical rows via %s", len(df), backend)
                    return df
            except Exception as exc:
                logger.warning("Historical read from %s failed: %s", backend, exc)

        return pd.DataFrame()


# ==========================================================================
# Model registry
# ==========================================================================

class ModelRegistry:
    """Loads and publishes models, MongoDB first."""

    def __init__(self, feature_store: Optional[FeatureStore] = None) -> None:
        self.store = feature_store or FeatureStore()

    def _models_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models")
        )

    def load(self, model_name: str = "lightgbm") -> Optional[Dict[str, Any]]:
        """Locate a model artifact directory.

        Returns ``{'path': dir, 'registry_metrics': {...}, 'source': str}``,
        or None when no backend can supply the model.
        """
        model_name = (model_name or "").strip().lower().replace(" ", "_")

        for backend in _backend_preference():
            try:
                if backend == "mongo":
                    db = self.store.mongo()
                    if db is None:
                        continue
                    from backend.mongo_client import load_model_from_registry_mongo

                    info = load_model_from_registry_mongo(db, model_name)
                elif backend == "hopsworks":
                    project = self.store.hopsworks_project
                    if project is None:
                        continue
                    from backend.hopsworks_client import load_model_from_registry

                    mr = project.get_model_registry()
                    info = load_model_from_registry(
                        mr,
                        model_name,
                        metric=os.getenv("MODEL_SELECTION_METRIC", "val_rmse"),
                        sort_by=os.getenv("MODEL_SELECTION_SORT", "min"),
                    )
                else:
                    continue

                if info and info.get("path"):
                    info["source"] = backend
                    logger.info("Loaded model '%s' from %s", model_name, backend)
                    return info
            except Exception as exc:
                logger.warning("Model load from %s failed for '%s': %s",
                               backend, model_name, exc)

        # Last resort: a model committed to the repo / baked into the image.
        if os.getenv("ALLOW_DISK_FALLBACK", "true").strip().lower() == "true":
            models_dir = self._models_dir()
            if os.path.exists(os.path.join(models_dir, f"{model_name}.pkl")):
                logger.info("Loaded model '%s' from local disk", model_name)
                return {"path": models_dir, "registry_metrics": {}, "source": "disk"}

        logger.error("Model '%s' not available from any backend", model_name)
        return None

    def publish(
        self,
        best_model_name: str,
        all_models_metrics: Dict[str, Dict[str, Any]],
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Publish trained models to every configured backend."""
        results: Dict[str, Any] = {}

        for backend in _backend_preference():
            if backend == "mongo":
                db = self.store.mongo()
                if db is None:
                    continue
                try:
                    from backend.mongo_client import register_models_to_mongo

                    register_models_to_mongo(
                        db, best_model_name, all_models_metrics,
                        feature_names, models_dir=self._models_dir(),
                    )
                    results["mongo"] = "ok"
                except Exception as exc:
                    logger.error("Mongo model publish failed: %s", exc)
                    results["mongo"] = f"error: {exc}"

            elif backend == "hopsworks":
                project = self.store.hopsworks_project
                if project is None:
                    continue
                try:
                    results["hopsworks"] = self._publish_hopsworks(
                        project, all_models_metrics
                    )
                except Exception as exc:
                    logger.error("Hopsworks model publish failed: %s", exc)
                    results["hopsworks"] = f"error: {exc}"

        return results

    def _publish_hopsworks(self, project, all_models_metrics) -> str:
        mr = project.get_model_registry()
        models_dir = self._models_dir()
        for display_name, metrics in all_models_metrics.items():
            try:
                module = mr.tensorflow if display_name == "TensorFlow NN" else mr.sklearn
                float_metrics = {}
                for key, value in (metrics or {}).items():
                    try:
                        float_metrics[key] = float(value)
                    except (TypeError, ValueError):
                        continue
                meta = module.create_model(
                    name=display_name.lower().replace(" ", "_"), metrics=float_metrics
                )
                meta.save(models_dir)
            except Exception as exc:
                logger.error("Hopsworks registration of %s failed: %s", display_name, exc)
        return "ok"

    def all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Metrics for every registered model, from the first backend with data."""
        for backend in _backend_preference():
            try:
                if backend == "mongo":
                    db = self.store.mongo()
                    if db is None:
                        continue
                    from backend.mongo_client import get_all_model_metrics_mongo

                    metrics = get_all_model_metrics_mongo(db)
                elif backend == "hopsworks":
                    project = self.store.hopsworks_project
                    if project is None:
                        continue
                    from backend.hopsworks_client import get_all_model_metrics

                    metrics = get_all_model_metrics(project.get_model_registry())
                else:
                    continue

                if metrics:
                    return metrics
            except Exception as exc:
                logger.warning("Metric fetch from %s failed: %s", backend, exc)
        return {}

    def best_model_name(self) -> Optional[str]:
        """The model flagged best by the latest training run (Mongo only)."""
        db = self.store.mongo()
        if db is None:
            return None
        try:
            from backend.mongo_client import get_best_model_name

            return get_best_model_name(db)
        except Exception as exc:
            logger.warning("Could not read best model from Mongo: %s", exc)
            return None


# Shared singletons for the API process.
_feature_store: Optional[FeatureStore] = None
_model_registry: Optional[ModelRegistry] = None


def get_feature_store() -> FeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry(get_feature_store())
    return _model_registry


def storage_status() -> Dict[str, Any]:
    """Backend availability, for the /health endpoint."""
    store = get_feature_store()
    return {
        "preference": _backend_preference(),
        "mongo_configured": _mongo_configured(),
        "hopsworks_configured": _hopsworks_configured(),
        "mongo_connected": store.mongo() is not None,
        "hopsworks_connected": store.hopsworks() is not None,
        "active_backend": store.active_backend(),
    }
