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
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_prep"))

from build_rolling_team_stats import REST_DAYS_CAP  # noqa: E402
from features import SELECTED_FEATURES  # noqa: E402
from mlflow_config import PRODUCTION_MODEL_URI, tracking_uri  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "NBAdata" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "NBAdata" / "scaler.pkl"
ROLLING_DIR = PROJECT_ROOT / "NBAdata" / "rolling_stats"


def season_label_for_date(target: date) -> str:
    start_year = target.year if target.month >= 7 else target.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def collect_rolling_snapshot_files() -> dict[str, Path]:
    """season_key ("2024_25") -> path, for every current-form snapshot on disk."""
    prefix, suffix = "nba_team_current_rolling_stats_", ".csv"
    season_files: dict[str, Path] = {}
    if not ROLLING_DIR.exists():
        return season_files
    for file_path in ROLLING_DIR.glob(f"{prefix}*{suffix}"):
        season_files[file_path.name[len(prefix):-len(suffix)]] = file_path
    return season_files


def load_team_stats(season_key: str) -> pd.DataFrame:
    snapshot_files = collect_rolling_snapshot_files()
    if season_key not in snapshot_files:
        raise FileNotFoundError(
            f"No current rolling-stats snapshot found for season {season_key}. "
            "Run build_rolling_team_stats.py + build_current_rolling_snapshot.py for this season first."
        )
    stats_df = pd.read_csv(snapshot_files[season_key])
    stats_df["TEAM_NAME"] = stats_df["TEAM_NAME"].str.strip().str.lower()
    return stats_df


def latest_team_stat_row(stats_df: pd.DataFrame, team: str, resolved_cols: list[str]):
    """The snapshot already holds one current row per team, so this is a plain lookup."""
    team_rows = stats_df[stats_df["TEAM_NAME"] == team].dropna(subset=resolved_cols, how="all")
    if team_rows.empty:
        return None
    return team_rows.iloc[-1]


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


def rest_days_and_b2b(last_game_date, target: date) -> tuple[float, float]:
    """Days of rest before `target`, given a team's snapshotted last known game
    date. Same cap/semantics as build_rolling_team_stats.py's training-time
    RestDays/B2B, just evaluated against the actual game being predicted
    instead of a historical row's own next game -- RestDays isn't a team
    property that can be snapshotted like the other stats, since it depends on
    which game is being predicted (see build_current_rolling_snapshot.py).
    """
    last_date = pd.to_datetime(last_game_date).date()
    rest_days = max(0, min((target - last_date).days - 1, REST_DAYS_CAP))
    return float(rest_days), float(rest_days == 0)


def build_feature_row(
    home_row: pd.Series, away_row: pd.Series, resolved_map: dict[str, str], target: date
) -> dict:
    row = {"Team1Home": 1}
    for model_col, source_col in resolved_map.items():
        row[f"Team1_{model_col}"] = home_row[source_col]
        row[f"Team2_{model_col}"] = away_row[source_col]
    row["Team1_RestDays"], row["Team1_B2B"] = rest_days_and_b2b(home_row["GAME_DATE"], target)
    row["Team2_RestDays"], row["Team2_B2B"] = rest_days_and_b2b(away_row["GAME_DATE"], target)
    return row


def assemble_features(
    stats_df: pd.DataFrame,
    resolved_map: dict[str, str],
    resolved_cols: list[str],
    home: str,
    away: str,
    target: date,
) -> dict | None:
    """Look up both teams' latest stats and build one feature row.

    Returns None when either team has no usable stats on file. The caller decides
    what that means (the batch job skips the game; the API returns a 404).
    """
    home_row = latest_team_stat_row(stats_df, home, resolved_cols)
    away_row = latest_team_stat_row(stats_df, away, resolved_cols)
    if home_row is None or away_row is None:
        return None
    return build_feature_row(home_row, away_row, resolved_map, target)


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
