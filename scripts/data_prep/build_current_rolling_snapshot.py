"""Build a "current form" snapshot: one row per team, holding that team's
most recent valid trailing rolling-average stats.

This is what serving/inference/predictor.py reads at prediction time. It
replaces the role NBAdata/monthly_stats/nba_team_combined_stats_<season>.csv
used to play, now that team stats come from per-game rolling averages
(scripts/data_prep/build_rolling_team_stats.py) instead of month-granularity
averages.

Season-boundary carryover (an intentional, documented asymmetry from
training): training (scripts/modeling/decision_tree_training.py) drops any
row whose rolling window isn't full enough yet (games_in_window < 3) and
never lets a team's rolling stats cross a season boundary -- a team's early
games in a new season just get dropped from the training set. Serving can't
do that: it must return *something* for every team even a few games into a
new season, before anyone has 3 valid games on the board yet. So this is the
one place in the whole rebuild where a team's stats are allowed to cross a
season boundary -- if a team has zero valid rows yet in the target season,
this script falls back to that team's last valid row from the *previous*
season's rolling-stats file. See documentation/TRAINING.md for more on this
asymmetry.

Output: NBAdata/rolling_stats/nba_team_current_rolling_stats_<season>.csv --
one row per team, with TEAM_NAME, the stat columns, GAME_DATE (the date of
that snapshotted row's own game), and Season (recording which season's data
the row actually came from: the target season, or a carried-over prior one,
for debuggability).

Note: RestDays/B2B are deliberately NOT snapshotted here, unlike the other
10 stats. They're a fact about the gap before a specific upcoming game, not
a team property that holds steady until the next refresh -- so serving
computes them dynamically at prediction time from this row's GAME_DATE and
the actual game being predicted (see serving/inference/predictor.py). This
is exactly why GAME_DATE is carried through instead of being dropped like
GAME_ID/games_in_window are.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NBAdata"
ROLLING_DIR = DATA_ROOT / "rolling_stats"

STAT_COLS = [
    "W_PCT", "PIE", "EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "FTR",
    "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE",
]

NBA_TEAMS = [
    "atlanta hawks", "boston celtics", "brooklyn nets", "charlotte hornets", "chicago bulls",
    "cleveland cavaliers", "dallas mavericks", "denver nuggets", "detroit pistons", "golden state warriors",
    "houston rockets", "indiana pacers", "la clippers", "los angeles lakers", "memphis grizzlies",
    "miami heat", "milwaukee bucks", "minnesota timberwolves", "new orleans pelicans", "new york knicks",
    "oklahoma city thunder", "orlando magic", "philadelphia 76ers", "phoenix suns", "portland trail blazers",
    "sacramento kings", "san antonio spurs", "toronto raptors", "utah jazz", "washington wizards",
]


def rolling_stats_path(season_key: str) -> Path:
    return ROLLING_DIR / f"nba_team_rolling_stats_{season_key}.csv"


def previous_season_key(season_key: str) -> str:
    start_year = int(season_key.split("_")[0])
    prev_start = start_year - 1
    return f"{prev_start}_{str(prev_start + 1)[-2:]}"


def latest_season_key() -> str:
    """Season key (e.g. "2025_26") of the most recent rolling-stats file on disk."""
    files = sorted(ROLLING_DIR.glob("nba_team_rolling_stats_*.csv"))
    if not files:
        raise FileNotFoundError(f"No rolling-stats files found in {ROLLING_DIR}")
    keys = [f.name[len("nba_team_rolling_stats_"):-len(".csv")] for f in files]
    return max(keys)


def last_valid_row_per_team(df: pd.DataFrame) -> pd.DataFrame:
    """Each team's last row (rows are pre-sorted per-team by GAME_DATE
    ascending) that has non-NaN stats, indexed by TEAM_NAME."""
    valid = df.dropna(subset=STAT_COLS, how="any")
    return valid.drop_duplicates(subset=["TEAM_NAME"], keep="last").set_index("TEAM_NAME")


def build_snapshot(target_season_key: str) -> pd.DataFrame:
    target_path = rolling_stats_path(target_season_key)
    if not target_path.exists():
        raise FileNotFoundError(f"No rolling-stats file for season {target_season_key}: {target_path}")

    target_df = pd.read_csv(target_path)
    target_latest = last_valid_row_per_team(target_df)

    missing_teams = [team for team in NBA_TEAMS if team not in target_latest.index]
    prev_latest = pd.DataFrame()
    if missing_teams:
        prev_key = previous_season_key(target_season_key)
        prev_path = rolling_stats_path(prev_key)
        if prev_path.exists():
            prev_df = pd.read_csv(prev_path)
            prev_latest = last_valid_row_per_team(prev_df)

    rows = []
    for team in NBA_TEAMS:
        if team in target_latest.index:
            row = target_latest.loc[team]
        elif team in prev_latest.index:
            row = prev_latest.loc[team]
        else:
            continue
        rows.append({
            "TEAM_NAME": team,
            **{col: row[col] for col in STAT_COLS},
            "GAME_DATE": row["GAME_DATE"],
            "Season": row["Season"],
        })

    return pd.DataFrame(rows, columns=["TEAM_NAME"] + STAT_COLS + ["GAME_DATE", "Season"])


def build_and_save(season_key: str) -> Path:
    result = build_snapshot(season_key)
    output_path = ROLLING_DIR / f"nba_team_current_rolling_stats_{season_key}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(result)} rows)")
    return output_path


def main() -> None:
    season_key = latest_season_key()
    build_and_save(season_key)


if __name__ == "__main__":
    main()
