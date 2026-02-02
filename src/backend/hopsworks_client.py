import hopsworks
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple


def connect_hopsworks(api_key: str, project_name: str = "aqi_predictor"):
    try:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        return project, fs
    except Exception as e:
        raise


def create_feature_group(
    fs,
    name: str = "aqi_features",
    version: int = 1,
):
    try:
        fg = fs.get_or_create_feature_group(
            name=name,
            version=version,
            primary_key=["time"],
            event_time="time",
            online_enabled=False,
        )
        return fg
    except Exception as e:
        raise


def insert_features(fg, df: pd.DataFrame):
    try:
        fg.insert(df, write_options={"wait_for_job": True})
    except Exception as e:
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
            fg = fs.get_feature_group("aqi_features", version=1)
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
                fg = fs.get_feature_group("aqi_features", version=1)
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
        fg = fs.get_feature_group("aqi_features", version=1)
        
        df = fg.filter(
            (fg.time >= start_time) & (fg.time <= end_time)
        ).read()
        
        return df
        
    except Exception as e:
        raise


def get_latest_features(fs, n_hours: int = 72) -> pd.DataFrame:
    try:
        fg = fs.get_feature_group("aqi_features", version=1)
        
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(hours=n_hours)
        
        df = fg.filter(
            fg.time >= start_time.strftime('%Y-%m-%d %H:%M:%S')
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

        if metric:
            try:
                model = mr.get_best_model(model_name, metric, sort_by or "min")
            except Exception:
                pass

        if model is None:
            try:
                candidates = mr.get_models(name=model_name)
                if candidates:
                    candidates = sorted(
                        candidates,
                        key=lambda m: getattr(m, "version", 0),
                        reverse=True
                    )
                    model = candidates[0]
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
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        all_metrics = {}
        model_names = ["lightgbm", "xgboost", "random_forest", "elasticnet", "tensorflow_nn"]
        
        logger.info(f"Attempting to fetch metrics from Hopsworks for models: {model_names}")
        
        for model_name in model_names:
            try:
                logger.info(f"Fetching metrics for model: {model_name}")
                candidates = mr.get_models(name=model_name)
                logger.info(f"Found {len(candidates) if candidates else 0} candidates for {model_name}")
                
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
                    logger.info(f"Raw metrics for {model_name}: {raw_metrics}")
                    
                    if isinstance(raw_metrics, dict):
                        for key, val in raw_metrics.items():
                            try:
                                metrics[key] = float(val) if isinstance(val, (int, float)) else val
                            except (ValueError, TypeError):
                                metrics[key] = val
                    
                    if metrics:
                        all_metrics[model_name] = metrics
                        logger.info(f"Successfully extracted metrics for {model_name}: {list(metrics.keys())}")
                    else:
                        logger.warning(f"No metrics found for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to fetch {model_name}: {str(e)}")
        
        logger.info(f"Total models with metrics: {len(all_metrics)}, models: {list(all_metrics.keys())}")
        return all_metrics
    except Exception as e:
        logger.error(f"Error in get_all_model_metrics: {str(e)}")
        return {}

