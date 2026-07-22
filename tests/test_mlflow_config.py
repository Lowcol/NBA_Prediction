import mlflow_config
from mlflow_config import (
    PRODUCTION_ALIAS,
    PRODUCTION_MODEL_URI,
    REGISTERED_MODEL_NAME,
    tracking_uri,
)


def test_production_model_uri_matches_name_and_alias():
    # Regression guard: training registers under REGISTERED_MODEL_NAME/PRODUCTION_ALIAS
    # and serving loads via PRODUCTION_MODEL_URI. If these drift, the batch job
    # silently falls back to the local pkl instead of the registered model.
    assert PRODUCTION_MODEL_URI == f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"


def test_tracking_uri_defaults_to_local_sqlite(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = tracking_uri()
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_tracking_uri_honors_env_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking-server:5000")
    assert tracking_uri() == "http://tracking-server:5000"


def test_tracking_uri_default_points_at_project_root(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    expected = f"sqlite:///{(mlflow_config.PROJECT_ROOT / 'mlflow.db').as_posix()}"
    assert tracking_uri() == expected
