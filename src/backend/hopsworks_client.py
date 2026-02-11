import hopsworks
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple


def connect_hopsworks(api_key: str, project_name: str = "aqi_predictor"):
    """
    Connect to Hopsworks feature store.
    
    Args:
        api_key: Hopsworks API key
        project_name: Project name (default: aqi_predictor)
    
    Returns:
        Tuple of (project, feature_store)
    """
    try:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        print(f"✅ Connected to Hopsworks project: {project.name}")
        return project, fs
    except Exception as e:
        print(f"❌ Failed to connect to Hopsworks: {str(e)}")
        raise


def create_feature_group(
    fs,
    name: str = "aqi_historical_features",
    version: int = 1,
    primary_key: Optional[list] = None,
    event_time: Optional[str] = None,
    online_enabled: bool = False,
):
    try:
        if primary_key is None:
            primary_key = ["time"]

        if event_time is None:
            event_time = "time"

        fg = fs.get_or_create_feature_group(
            name=name,
            version=version,
            primary_key=primary_key,
            event_time=event_time,
            online_enabled=online_enabled,
        )
        return fg
    except Exception as e:
        raise


def insert_features(fg, df: pd.DataFrame, retries: int = 3):
    """
    Insert features into Hopsworks with retry logic for network resilience.
    
    Args:
        fg: Feature group object
        df: DataFrame to insert
        retries: Number of retry attempts (default: 3)
    """
    import time
    
    for attempt in range(retries):
        try:
            fg.insert(df, write_options={"wait_for_job": True})
            print(f"✅ Successfully inserted {len(df)} rows to {fg.name}")
            return
        except Exception as e:
            if attempt < retries - 1:
                # Exponential backoff: 2^attempt seconds (2s, 4s, 8s...)
                wait_time = 2 ** attempt
                print(f"⚠️  Insert failed (attempt {attempt + 1}/{retries}): {str(e)[:100]}")
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # All retries exhausted
                print(f"❌ Failed to insert {len(df)} rows after {retries} attempts")
                raise

def get_feature_view(
    fs,
    name: str = "aqi_feature_view",
    version: int = 1,
    query=None,
    label_col: str = "aqi"
):
    try:
        if query is None:
            fg = fs.get_feature_group("aqi_historical_features", version=1)
            query = fg.select_all()
        
        fv = fs.get_or_create_feature_view(
            name=name,
            version=version,
            query=query,
            labels=[label_col],
        )
        return fv
    except ValueError as e:
        if "no longer valid" in str(e) or "Parent feature groups" in str(e):
            try:
                existing_fvs = fs.get_feature_views(name=name)
                if existing_fvs:
                    max_version = max([fv.version for fv in existing_fvs])
                    new_version = max_version + 1
                else:
                    new_version = version
            except:
                new_version = version + 1
            
            if query is None:
                fg = fs.get_feature_group("aqi_historical_features", version=1)
                query = fg.select_all()
            
            fv = fs.create_feature_view(
                name=name,
                version=new_version,
                query=query,
                labels=[label_col],
            )
            return fv
        else:
            raise
    except Exception as e:
        raise


def get_training_data(
    fv,
    training_dataset_version: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = fv.get_training_data(
            training_dataset_version=training_dataset_version
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        raise


def create_training_dataset(
    fv,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    try:
        version, job = fv.create_training_data(
            data_format="csv",
            start_time=start_time,
            end_time=end_time,
            statistics_config={"enabled": True, "histograms": True, "correlations": True},
            write_options={"wait_for_job": True},
        )
        
        return version
        
    except Exception as e:
        raise


def get_batch_data(fs, start_time: str, end_time: str) -> pd.DataFrame:
    try:
        fg = fs.get_feature_group("aqi_historical_features", version=1)
        
        df = fg.filter(
            (fg.time >= start_time) & (fg.time <= end_time)
        ).read()
        
        return df
        
    except Exception as e:
        raise


def get_latest_features(fs, n_hours: int = 72) -> pd.DataFrame:
    try:
        fg = fs.get_feature_group("aqi_historical_features", version=1)
        
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(hours=n_hours)
        
        df = fg.filter(
            fg.time >= start_time.strftime('%Y-%m-%d %H:%M:%S')
        ).read()
        
        return df
        
    except Exception as e:
        raise


def get_forecast_features(
    fs,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    try:
        fg = fs.get_feature_group("weather_forecast_features", version=1)

        df = fg.filter(
            (fg.time >= start_time) &
            (fg.time <= end_time)
        ).read()

        return df
    except Exception as e:
        raise


def load_model_from_registry(
    mr,
    model_name: str = "lightgbm",
    metric: str = None,
    sort_by: str = None
):
    try:
        model = None
        normalized_name = (model_name or "").strip().lower().replace(" ", "_")
        name_candidates = [normalized_name]
        alias_map = {
            "lightgbm": ["lightgbm"],
            "xgboost": ["xgboost"],
            "random_forest": ["random_forest"],
            "elasticnet": ["elasticnet"],
            "tensorflow_nn": ["tensorflow_nn"],
        }
        name_candidates = alias_map.get(normalized_name, name_candidates)

        if metric:
            for name in name_candidates:
                try:
                    model = mr.get_best_model(name, metric, sort_by or "min")
                    if model is not None:
                        break
                except Exception:
                    continue

        if model is None:
            for name in name_candidates:
                try:
                    candidates = mr.get_models(name=name)
                    if candidates:
                        candidates = sorted(
                            candidates,
                            key=lambda m: getattr(m, "version", 0),
                            reverse=True
                        )
                        model = candidates[0]
                        break
                except Exception:
                    continue

        if model is None:
            try:
                all_models = mr.get_models()
                if all_models:
                    matched = [
                        m for m in all_models
                        if getattr(m, "name", "").strip().lower().replace(" ", "_") in name_candidates
                    ]
                    if matched:
                        matched = sorted(
                            matched,
                            key=lambda m: getattr(m, "version", 0),
                            reverse=True
                        )
                        model = matched[0]
            except Exception:
                pass

        if model is None:
            try:
                model = mr.get_model(model_name)
            except Exception:
                return None
        
        if model is None:
            return None
        
        try:
            model_dir = model.download()
        except AttributeError:
            return None
        except Exception:
            return None
        
        if not model_dir:
            return None

        registry_metrics = getattr(model, "metrics", None) or getattr(model, "model_metrics", None) or {}
        return {"path": model_dir, "registry_metrics": registry_metrics}
        
    except Exception:
        return None


def load_model_metadata(mr, model_name: str = "lightgbm_aqi_predictor"):
    try:
        model = mr.get_model(model_name, version=1)
        
        metadata = {
            "feature_names": model.training_input_example,
            "model_metrics": model.model_metrics or {},
        }
        
        return metadata
        
    except Exception as e:
        raise


def get_all_model_metrics(mr) -> dict:
    """Fetch metrics for all five models from Hopsworks registry."""
    try:
        all_metrics = {}
        model_names = ["lightgbm", "xgboost", "random_forest", "elasticnet", "tensorflow_nn"]

        for model_name in model_names:
            try:
                candidates = mr.get_models(name=model_name)
                if candidates:
                    latest = sorted(
                        candidates,
                        key=lambda m: getattr(m, "version", 0),
                        reverse=True
                    )[0]
                    
                    metrics = {}
                    # Try multiple attribute names where metrics might be stored
                    raw_metrics = (
                        getattr(latest, "metrics", None) 
                        or getattr(latest, "model_metrics", None) 
                        or getattr(latest, "training_metrics", None)
                        or {}
                    )

                    if isinstance(raw_metrics, dict):
                        for key, val in raw_metrics.items():
                            try:
                                metrics[key] = float(val) if isinstance(val, (int, float)) else val
                            except (ValueError, TypeError):
                                metrics[key] = val
                    
                    if metrics:
                        all_metrics[model_name] = metrics
            except Exception as e:
                pass

        return all_metrics
    except Exception as e:
        return {}

