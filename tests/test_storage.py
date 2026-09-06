"""Tests for the storage layer and the MongoDB client.

These use mongomock / fakes, so they run without a live Atlas cluster.
"""

import io
import json
import os
import pickle
import sys
import zipfile
from datetime import datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from backend import mongo_client
from backend import storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mongo_db():
    """An in-memory Mongo database (no GridFS support)."""
    mongomock = pytest.importorskip("mongomock")
    client = mongomock.MongoClient()
    return client["aqi_test"]


@pytest.fixture
def real_mongo_db():
    """A real MongoDB, required for GridFS-backed registry tests.

    Set TEST_MONGODB_URI to run these; they are skipped otherwise since
    mongomock does not implement GridFS.
    """
    uri = os.getenv("TEST_MONGODB_URI")
    if not uri:
        pytest.skip("TEST_MONGODB_URI not set; skipping GridFS registry tests")
    from pymongo import MongoClient

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client["aqi_registry_test"]
    for name in db.list_collection_names():
        db[name].drop()
    yield db
    client.drop_database("aqi_registry_test")


@pytest.fixture
def features_df():
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "aqi": [100.0, 110.0, 120.0, 130.0, 140.0],
            "pm2_5": [35.0, 40.0, 45.0, 50.0, 55.0],
            "temperature_2m": [20.0, 21.0, 22.0, 23.0, 24.0],
        }
    )


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("MONGODB_URI", "HOPSWORKS_API_KEY", "STORAGE_BACKEND",
                "MONGO_DB", "ALLOW_DISK_FALLBACK"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Time normalisation
# ---------------------------------------------------------------------------

def test_to_naive_utc_handles_tz_naive_input():
    """The old client called .dt.tz_convert directly and raised here."""
    naive = pd.Series(pd.date_range("2026-01-01", periods=3, freq="h"))
    result = mongo_client._to_naive_utc(naive)
    assert result.dt.tz is None
    assert result.iloc[0] == pd.Timestamp("2026-01-01 00:00:00")


def test_to_naive_utc_converts_non_utc_zone():
    karachi = pd.Series(
        pd.date_range("2026-01-01 05:00", periods=2, freq="h", tz="Asia/Karachi")
    )
    result = mongo_client._to_naive_utc(karachi)
    # Asia/Karachi is UTC+5, so 05:00 local is 00:00 UTC.
    assert result.iloc[0] == pd.Timestamp("2026-01-01 00:00:00")


# ---------------------------------------------------------------------------
# Feature upserts
# ---------------------------------------------------------------------------

def test_insert_features_is_idempotent(mongo_db, features_df):
    """Re-running the pipeline over the same window must not duplicate rows."""
    mongo_client.insert_features(mongo_db, "aqi_historical_features", features_df)
    mongo_client.insert_features(mongo_db, "aqi_historical_features", features_df)

    assert mongo_db["aqi_historical_features"].count_documents({}) == 5


def test_insert_features_updates_existing_row(mongo_db, features_df):
    mongo_client.insert_features(mongo_db, "aqi_historical_features", features_df)

    revised = features_df.copy()
    revised.loc[0, "aqi"] = 999.0
    mongo_client.insert_features(mongo_db, "aqi_historical_features", revised)

    doc = mongo_db["aqi_historical_features"].find_one(
        {"time": datetime(2026, 1, 1, 0, 0)}
    )
    assert doc["aqi"] == 999.0
    assert mongo_db["aqi_historical_features"].count_documents({}) == 5


def test_insert_features_deduplicates_within_batch(mongo_db, features_df):
    doubled = pd.concat([features_df, features_df], ignore_index=True)
    mongo_client.insert_features(mongo_db, "aqi_historical_features", doubled)
    assert mongo_db["aqi_historical_features"].count_documents({}) == 5


def test_insert_features_empty_is_noop(mongo_db):
    assert mongo_client.insert_features(mongo_db, "x", pd.DataFrame()) == 0
    assert mongo_client.insert_features(mongo_db, "x", None) == 0


def test_insert_features_requires_time_column(mongo_db):
    with pytest.raises(ValueError, match="no 'time' column"):
        mongo_client.insert_features(mongo_db, "x", pd.DataFrame({"aqi": [1.0]}))


def test_read_roundtrip_preserves_utc(mongo_db, features_df):
    mongo_client.insert_features(mongo_db, "aqi_historical_features", features_df)
    out = mongo_client.get_batch_data(
        mongo_db, "2026-01-01 00:00:00", "2026-01-01 04:00:00"
    )
    assert len(out) == 5
    assert str(out["time"].dt.tz) == "UTC"
    assert "_id" not in out.columns
    assert out["aqi"].tolist() == [100.0, 110.0, 120.0, 130.0, 140.0]


def test_read_range_is_bounded(mongo_db, features_df):
    mongo_client.insert_features(mongo_db, "aqi_historical_features", features_df)
    out = mongo_client.get_batch_data(
        mongo_db, "2026-01-01 01:00:00", "2026-01-01 02:00:00"
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class _DummyModel:
    def predict(self, X):
        return [100.0] * len(X)


@pytest.fixture
def models_dir(tmp_path):
    with open(tmp_path / "lightgbm.pkl", "wb") as f:
        pickle.dump(_DummyModel(), f)
    (tmp_path / "feature_names.json").write_text(json.dumps(["aqi_lag_1h", "temp"]))
    (tmp_path / "metrics.json").write_text(json.dumps({"val_rmse": 5.5}))
    return str(tmp_path)


def test_register_and_load_roundtrip(real_mongo_db, models_dir):
    """A model published on one machine must load on another via GridFS."""
    mongo_client.register_models_to_mongo(
        real_mongo_db,
        best_model_name="lightgbm",
        all_models_metrics={"lightgbm": {"val_rmse": 5.5}},
        feature_names=["aqi_lag_1h", "temp"],
        models_dir=models_dir,
    )

    info = mongo_client.load_model_from_registry_mongo(real_mongo_db, "lightgbm")
    assert info is not None
    # Artifact came back from GridFS, not the original directory.
    assert info["path"] != models_dir
    assert os.path.exists(os.path.join(info["path"], "lightgbm.pkl"))
    assert info["registry_metrics"]["val_rmse"] == 5.5

    with open(os.path.join(info["path"], "lightgbm.pkl"), "rb") as f:
        assert isinstance(pickle.load(f), _DummyModel)


def test_register_bumps_version(real_mongo_db, models_dir):
    for _ in range(3):
        mongo_client.register_models_to_mongo(
            real_mongo_db, "lightgbm", {"lightgbm": {"val_rmse": 5.5}},
            ["a"], models_dir=models_dir,
        )
    versions = sorted(
        d["version"] for d in real_mongo_db["model_registry"].find({"name": "lightgbm"})
    )
    assert versions == [1, 2, 3]

    # The loader must return the newest version.
    assert mongo_client.load_model_from_registry_mongo(real_mongo_db, "lightgbm")["version"] == 3


def test_only_one_best_model_flag(real_mongo_db, models_dir):
    mongo_client.register_models_to_mongo(
        real_mongo_db, "xgboost",
        {"lightgbm": {"val_rmse": 9.0}, "xgboost": {"val_rmse": 4.0}},
        ["a"], models_dir=models_dir,
    )
    best = list(real_mongo_db["model_registry"].find({"best_model": True}))
    assert len(best) == 1
    assert best[0]["name"] == "xgboost"
    assert mongo_client.get_best_model_name(real_mongo_db) == "xgboost"


def test_load_missing_model_returns_none(mongo_db):
    assert mongo_client.load_model_from_registry_mongo(mongo_db, "nonexistent") is None


def test_display_names_are_normalised(real_mongo_db, models_dir):
    mongo_client.register_models_to_mongo(
        real_mongo_db, "Random Forest", {"Random Forest": {"val_rmse": 6.0}},
        ["a"], models_dir=models_dir,
    )
    doc = real_mongo_db["model_registry"].find_one({"name": "random_forest"})
    assert doc is not None
    assert doc["display_name"] == "Random Forest"


def test_all_model_metrics(real_mongo_db, models_dir):
    mongo_client.register_models_to_mongo(
        real_mongo_db, "lightgbm",
        {"lightgbm": {"val_rmse": 5.5}, "xgboost": {"val_rmse": 6.5}},
        ["a"], models_dir=models_dir,
    )
    metrics = mongo_client.get_all_model_metrics_mongo(real_mongo_db)
    assert metrics["lightgbm"]["val_rmse"] == 5.5
    assert metrics["xgboost"]["val_rmse"] == 6.5


def test_legacy_filesystem_path_entry_still_loads(mongo_db, models_dir):
    """Registry documents written by the old code path must remain readable."""
    mongo_db["model_registry"].insert_one({
        "name": "lightgbm", "version": 1,
        "filesystem_path": models_dir, "metrics": {"val_rmse": 7.0},
    })
    info = mongo_client.load_model_from_registry_mongo(mongo_db, "lightgbm")
    assert info["path"] == models_dir


# ---------------------------------------------------------------------------
# Backend priority
# ---------------------------------------------------------------------------

def test_mongo_is_preferred_by_default(clean_env):
    assert storage._backend_preference() == ["mongo", "hopsworks"]


def test_backend_preference_can_be_pinned(clean_env):
    clean_env.setenv("STORAGE_BACKEND", "mongo")
    assert storage._backend_preference() == ["mongo"]

    clean_env.setenv("STORAGE_BACKEND", "hopsworks")
    assert storage._backend_preference() == ["hopsworks"]


def test_store_reports_none_when_unconfigured(clean_env):
    store = storage.FeatureStore()
    assert store.mongo() is None
    assert store.active_backend() == "none"


def test_read_falls_back_to_hopsworks_when_mongo_empty(clean_env, monkeypatch, features_df):
    """Mongo is asked first; an empty result must not stop the Hopsworks attempt."""
    import types

    store = storage.FeatureStore()
    calls = []

    monkeypatch.setattr(store, "mongo", lambda: "mongo-db")
    monkeypatch.setattr(store, "hopsworks", lambda: "hops-fs")

    def fake_mongo_read(db, start, end, collection_name=None):
        calls.append("mongo")
        return pd.DataFrame()

    def fake_hops_read(fs, start, end):
        calls.append("hopsworks")
        return features_df

    monkeypatch.setattr(mongo_client, "get_batch_data", fake_mongo_read)

    # Stand in for the optional hopsworks package so the test does not
    # require it to be installed.
    fake_module = types.ModuleType("backend.hopsworks_client")
    fake_module.get_batch_data = fake_hops_read
    fake_module.get_forecast_features = fake_hops_read
    monkeypatch.setitem(sys.modules, "backend.hopsworks_client", fake_module)

    out = store.read_range("aqi_historical_features", "2026-01-01", "2026-01-02")
    assert calls == ["mongo", "hopsworks"]
    assert len(out) == 5


def test_read_skips_hopsworks_when_package_missing(clean_env, monkeypatch):
    """A missing hopsworks package degrades to 'unavailable', never a crash."""
    store = storage.FeatureStore()
    monkeypatch.setattr(store, "mongo", lambda: None)
    monkeypatch.setattr(store, "hopsworks", lambda: "hops-fs")

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "backend.hopsworks_client" or name.endswith("hopsworks_client"):
            raise ImportError("No module named 'hopsworks'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    out = store.read_range("aqi_historical_features", "2026-01-01", "2026-01-02")
    assert out.empty


def test_read_uses_mongo_and_skips_hopsworks_when_data_present(
    clean_env, monkeypatch, features_df
):
    store = storage.FeatureStore()
    calls = []

    monkeypatch.setattr(store, "mongo", lambda: "mongo-db")
    monkeypatch.setattr(store, "hopsworks", lambda: "hops-fs")

    def fake_mongo_read(db, start, end, collection_name=None):
        calls.append("mongo")
        return features_df

    monkeypatch.setattr(mongo_client, "get_batch_data", fake_mongo_read)
    out = store.read_range("aqi_historical_features", "2026-01-01", "2026-01-02")

    assert calls == ["mongo"]  # Hopsworks never consulted.
    assert len(out) == 5


def test_write_fans_out_to_all_backends(clean_env, monkeypatch, features_df):
    """Both stores stay in sync; a write is not a first-success fallback."""
    store = storage.FeatureStore()
    written = []

    monkeypatch.setattr(store, "mongo", lambda: "mongo-db")
    monkeypatch.setattr(store, "hopsworks", lambda: None)
    monkeypatch.setattr(
        mongo_client, "insert_features",
        lambda db, col, df: written.append(col) or len(df),
    )

    results = store.write_features("aqi_historical_features", features_df)
    assert results["mongo"] == 5
    assert written == ["aqi_historical_features"]


def test_write_records_backend_error_without_raising(clean_env, monkeypatch, features_df):
    store = storage.FeatureStore()
    monkeypatch.setattr(store, "mongo", lambda: "mongo-db")
    monkeypatch.setattr(store, "hopsworks", lambda: None)

    def boom(db, col, df):
        raise RuntimeError("atlas down")

    monkeypatch.setattr(mongo_client, "insert_features", boom)
    results = store.write_features("aqi_historical_features", features_df)
    assert "error: atlas down" in results["mongo"]


def test_storage_status_shape(clean_env):
    status = storage.storage_status()
    assert status["preference"] == ["mongo", "hopsworks"]
    assert status["mongo_configured"] is False
    assert status["active_backend"] == "none"


# ---------------------------------------------------------------------------
# Artifact bundling (GridFS payload construction, no server needed)
# ---------------------------------------------------------------------------

def test_bundle_contains_model_and_sidecars(models_dir):
    """The GridFS payload must carry everything needed to serve the model."""
    blob = mongo_client._bundle_model_files(models_dir, "lightgbm")
    assert blob is not None

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
        assert names == {"lightgbm.pkl", "feature_names.json", "metrics.json"}
        assert json.loads(zf.read("feature_names.json")) == ["aqi_lag_1h", "temp"]
        assert pickle.loads(zf.read("lightgbm.pkl")).predict([1]) == [100.0]


def test_bundle_prefers_model_specific_sidecar(tmp_path):
    with open(tmp_path / "xgboost.pkl", "wb") as f:
        pickle.dump(_DummyModel(), f)
    (tmp_path / "feature_names.json").write_text(json.dumps(["shared"]))
    (tmp_path / "feature_names_xgboost.json").write_text(json.dumps(["specific"]))

    blob = mongo_client._bundle_model_files(str(tmp_path), "xgboost")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert json.loads(zf.read("feature_names.json")) == ["specific"]


def test_bundle_returns_none_when_pickle_missing(tmp_path):
    assert mongo_client._bundle_model_files(str(tmp_path), "absent") is None


def test_bundle_unpacks_to_a_loadable_directory(models_dir, tmp_path):
    """Round-trip the bundle the way load_model_from_registry_mongo does."""
    blob = mongo_client._bundle_model_files(models_dir, "lightgbm")
    target = tmp_path / "unpacked"
    target.mkdir()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(target)

    assert (target / "lightgbm.pkl").exists()
    with open(target / "lightgbm.pkl", "rb") as f:
        assert isinstance(pickle.load(f), _DummyModel)
