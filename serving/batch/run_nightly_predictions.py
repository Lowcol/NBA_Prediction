import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from curl_cffi import requests as cr
from nba_api.stats.endpoints import scheduleleaguev2
from nba_api.stats.library.http import NBAStatsHTTP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "modeling"))
sys.path.insert(0, str(PROJECT_ROOT / "serving" / "inference"))

from features import SELECTED_FEATURES, resolve_stat_columns  # noqa: E402
from injury_report import fetch_latest_injury_counts  # noqa: E402
from predictor import (  # noqa: E402
    MODEL_PATH,
    SCALER_PATH,
    assemble_features,
    build_feature_row,
    latest_team_stat_row,
    load_predictor,
    load_team_stats,
    predict_from_features,
    season_label_for_date,
)

PREDICTIONS_DIR = PROJECT_ROOT / "NBAdata" / "predictions"

# Re-exported so callers/tests importing these names from this module keep working
# after the inference logic moved to serving/inference/predictor.py.
__all__ = [
    "MODEL_PATH",
    "SCALER_PATH",
    "assemble_features",
    "build_feature_row",
    "fetch_latest_injury_counts",
    "fetch_schedule",
    "games_on_date",
    "latest_team_stat_row",
    "load_predictor",
    "load_team_stats",
    "predict_from_features",
    "season_label_for_date",
]


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

    # Fetched once per run (not once per game): it's the same live report for every
    # game on this slate, and each fetch downloads + parses a multi-page PDF.
    injury_counts = fetch_latest_injury_counts()

    results = []
    for _, game in day_games.iterrows():
        home, away = game["Team1"], game["Team2"]

        features = assemble_features(
            stats_df, resolved_map, resolved_cols, home, away, target, injury_counts
        )
        if features is None:
            print(f"Skipping {home} vs {away}: no usable stats on file for one or both teams.")
            continue

        missing = [f for f in SELECTED_FEATURES if f not in features]
        if missing:
            print(f"Skipping {home} vs {away}: missing features {missing}.")
            continue

        pred, home_win_prob = predict_from_features(predictor, features)

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
