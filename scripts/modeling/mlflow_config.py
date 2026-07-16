"""MLflow registry settings shared by training and serving, so the two can't
disagree on which model/alias/tracking store they use (same drift-prevention
reason `features.py` is shared)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTERED_MODEL_NAME = "nba-win-predictor"
PRODUCTION_ALIAS = "production"
PRODUCTION_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"


def tracking_uri() -> str:
    """Local sqlite store by default; override via the MLFLOW_TRACKING_URI env
    var (e.g. to point at a tracking server when running in a container)."""
    return os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
    )
