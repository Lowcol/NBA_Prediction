"""Shared inference code used by both the nightly batch job and the real-time API.

Loading the model and turning team stats into a prediction lives here in one place,
so the batch job (serving/batch/run_nightly_predictions.py) and the API
(serving/api/main.py) can't drift apart in how they call the model. Same idea as the
shared feature contract in features.py.
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "modeling"))

from decision_tree_training import collect_monthly_files  # noqa: E402
from features import SELECTED_FEATURES  # noqa: E402
from mlflow_config import PRODUCTION_MODEL_URI, tracking_uri  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "NBAdata" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "NBAdata" / "scaler.pkl"


def season_label_for_date(target: date) -> str:
    start_year = target.year if target.month >= 7 else target.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def load_team_stats(season_key: str) -> pd.DataFrame:
    monthly_files = collect_monthly_files()
    if season_key not in monthly_files:
        raise FileNotFoundError(
            f"No monthly stats file found for season {season_key}. "
            "Run the data_pull + merge_advanced_base_stats.py scripts for this season first."
        )
    stats_df = pd.read_csv(monthly_files[season_key])
    stats_df["TEAM_NAME"] = stats_df["TEAM_NAME"].str.strip().str.lower()
    stats_df["Month"] = pd.to_numeric(stats_df["Month"], errors="coerce")
    return stats_df


def latest_team_stat_row(stats_df: pd.DataFrame, team: str, month: int, resolved_cols: list[str]):
    team_rows = stats_df[stats_df["TEAM_NAME"] == team]
    if team_rows.empty:
        return None

    exact = team_rows[team_rows["Month"] == month].dropna(subset=resolved_cols, how="all")
    if not exact.empty:
        return exact.iloc[-1]

    # Fall back to the most recent month on file with usable stats for this team.
    # NBA seasons cross the calendar year (Oct=10..Dec=12, then Jan=1..Jun=6), so a plain
    # numeric sort on Month puts December last. Sort on a season-relative ordinal instead
    # ((month - 10) % 12) so Oct < Nov < ... < Jun and iloc[-1] is chronologically latest.
    candidates = team_rows.dropna(subset=resolved_cols, how="all").copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(by="Month", key=lambda m: (m - 10) % 12)
    return candidates.iloc[-1]


def load_predictor():
    """Return a fitted estimator that maps raw features -> prediction.

    Prefers the ``@production`` pipeline from the MLflow registry. Falls back to
    the local scaler+model pkls (wrapped in the same kind of pipeline) when the
    registry isn't reachable — e.g. a container runs without the sqlite tracking
    store mounted, so it uses the pkls baked alongside the data.
    """
    mlflow.set_tracking_uri(tracking_uri())
    try:
        predictor = mlflow.sklearn.load_model(PRODUCTION_MODEL_URI)
        print(f"Loaded model from registry: {PRODUCTION_MODEL_URI}")
        return predictor
    except Exception as exc:
        print(f"Registry model unavailable ({exc}); falling back to local pkl artifacts.")
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise RuntimeError(
                "No model available: the MLflow registry is unreachable and the local pkl "
                f"artifacts are missing ({MODEL_PATH}, {SCALER_PATH}). Either make the registry "
                "reachable, or mount NBAdata/ (with best_model.pkl + scaler.pkl) into the "
                "container at runtime, e.g. `docker run -v \"$(pwd)/NBAdata:/app/NBAdata\" ...`."
            ) from exc
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return Pipeline([("scaler", scaler), ("model", model)])


def build_feature_row(home_row: pd.Series, away_row: pd.Series, resolved_map: dict[str, str]) -> dict:
    row = {"Team1Home": 1}
    for model_col, source_col in resolved_map.items():
        row[f"Team1_{model_col}"] = home_row[source_col]
        row[f"Team2_{model_col}"] = away_row[source_col]
    return row


def assemble_features(
    stats_df: pd.DataFrame,
    resolved_map: dict[str, str],
    resolved_cols: list[str],
    home: str,
    away: str,
    month: int,
) -> dict | None:
    """Look up both teams' latest stats and build one feature row.

    Returns None when either team has no usable stats on file. The caller decides
    what that means (the batch job skips the game; the API returns a 404).
    """
    home_row = latest_team_stat_row(stats_df, home, month, resolved_cols)
    away_row = latest_team_stat_row(stats_df, away, month, resolved_cols)
    if home_row is None or away_row is None:
        return None
    return build_feature_row(home_row, away_row, resolved_map)


def predict_from_features(predictor, features: dict) -> tuple[int, float | None]:
    """Run the model on one feature row.

    Returns (predicted_home_win, home_win_probability). The probability is None when
    the estimator has no predict_proba (some models don't expose one).
    """
    X = pd.DataFrame([features])[SELECTED_FEATURES]
    pred = int(predictor.predict(X)[0])
    home_win_prob = (
        float(predictor.predict_proba(X)[0][1]) if hasattr(predictor, "predict_proba") else None
    )
    return pred, home_win_prob
