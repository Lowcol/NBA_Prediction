import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from curl_cffi import requests as cr
from nba_api.stats.endpoints import scheduleleaguev2
from nba_api.stats.library.http import NBAStatsHTTP
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "modeling"))

from decision_tree_training import collect_monthly_files  # noqa: E402
from features import SELECTED_FEATURES, resolve_stat_columns  # noqa: E402
from mlflow_config import PRODUCTION_MODEL_URI, tracking_uri  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "NBAdata" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "NBAdata" / "scaler.pkl"
PREDICTIONS_DIR = PROJECT_ROOT / "NBAdata" / "predictions"


def season_label_for_date(target: date) -> str:
    start_year = target.year if target.month >= 7 else target.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_schedule(season: str) -> pd.DataFrame:
    session = cr.Session(impersonate="chrome120")
    print("Warming up Akamai cookies...")
    session.get("https://www.nba.com/stats/", timeout=20)
    NBAStatsHTTP.get_session = lambda self: session

    sched = scheduleleaguev2.ScheduleLeagueV2(season=season)
    return sched.get_data_frames()[0]


def games_on_date(schedule_df: pd.DataFrame, target: date) -> pd.DataFrame:
    schedule_df = schedule_df.copy()
    schedule_df["parsed_date"] = pd.to_datetime(schedule_df["gameDate"]).dt.date
    day_games = schedule_df[schedule_df["parsed_date"] == target].copy()
    day_games["Team1"] = (
        (day_games["homeTeam_teamCity"] + " " + day_games["homeTeam_teamName"]).str.strip().str.lower()
    )
    day_games["Team2"] = (
        (day_games["awayTeam_teamCity"] + " " + day_games["awayTeam_teamName"]).str.strip().str.lower()
    )
    return day_games


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
    registry isn't reachable — e.g. the batch container runs without the
    sqlite tracking store mounted, so it uses the pkls baked alongside the data.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NBA game winners for a given slate date.")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date as YYYY-MM-DD (default: tomorrow). Also accepts past dates for testing.",
    )
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today() + timedelta(days=1)
    season = season_label_for_date(target)
    season_key = season.replace("-", "_")

    print(f"Fetching {season} schedule...")
    schedule_df = fetch_schedule(season)
    day_games = games_on_date(schedule_df, target)
    if day_games.empty:
        print(f"No games scheduled on {target.isoformat()}.")
        return

    stats_df = load_team_stats(season_key)
    resolved_map, selected_cols = resolve_stat_columns(stats_df.columns)
    resolved_cols = [c for c in selected_cols if c not in ("TEAM_NAME", "Season", "Month")]

    predictor = load_predictor()

    results = []
    for _, game in day_games.iterrows():
        home, away = game["Team1"], game["Team2"]

        home_row = latest_team_stat_row(stats_df, home, target.month, resolved_cols)
        away_row = latest_team_stat_row(stats_df, away, target.month, resolved_cols)
        if home_row is None or away_row is None:
            print(f"Skipping {home} vs {away}: no usable stats on file for one or both teams.")
            continue

        features = build_feature_row(home_row, away_row, resolved_map)
        missing = [f for f in SELECTED_FEATURES if f not in features]
        if missing:
            print(f"Skipping {home} vs {away}: missing features {missing}.")
            continue

        X = pd.DataFrame([features])[SELECTED_FEATURES]
        pred = predictor.predict(X)[0]
        home_win_prob = (
            predictor.predict_proba(X)[0][1] if hasattr(predictor, "predict_proba") else None
        )

        results.append({
            "GameDate": target.isoformat(),
            "GameId": game.get("gameId"),
            "HomeTeam": home,
            "AwayTeam": away,
            "PredictedWinner": home if pred == 1 else away,
            "HomeWinProbability": home_win_prob,
        })

    if not results:
        print("No predictions produced (no games had usable stats).")
        return

    predictions_df = pd.DataFrame(results)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / f"predictions_{target.isoformat()}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved {len(predictions_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
