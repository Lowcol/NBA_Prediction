"""Build trailing rolling-window team stats from per-game team logs.

Reads NBAdata/team_game_logs/team_game_logs_<season>.csv (one row per team
per game, built by scripts/data_pull/team_game_logs_pull.py) and computes,
per team, trailing rolling averages over each team's own game history
(Regular Season + PlayIn + Playoffs, interleaved chronologically by
GAME_DATE -- Pre Season games are excluded from the history that feeds the
window, see INCLUDE_PRESEASON_IN_WINDOW).

The core correctness requirement: each row's rolling average must use only
that team's games strictly BEFORE its own GAME_DATE. This is done with
`.shift(1)` before `.rolling(...)` -- a plain `.rolling(N).mean()` would
include the current row and reintroduce leakage.

Output columns are named to match scripts/modeling/features.py's STAT_MAP
source options (W_PCT, PIE, EFG_PCT, TM_TOV_PCT, OREB_PCT, FTR, NET_RATING,
OFF_RATING, DEF_RATING, PACE, RestDays, B2B) so that resolve_stat_columns()
needs zero changes to consume this file.

Note on W_PCT semantics: in the source per-game logs, W_PCT doesn't exist
as a column at all -- it's derived here from WL ('W' -> 1, 'L' -> 0) as a
trailing win rate over the rolling window. This is a change in meaning from
the old monthly_stats pipeline's W_PCT, which was season-cumulative win
percentage. Same column name, different semantics -- intentional.

Note on FTR: also derived, not a source column. FTR is free-throw RATE
(FTA/FGA -- how often a team gets to the line), one of basketball's "Four
Factors". The old monthly_stats pipeline sourced FTR from FT_PCT (free-throw
shooting PERCENTAGE, i.e. FTM/FTA) instead, which measures something
different -- a known approximation, not a bug introduced here. Fixed as of
this rolling-window rebuild since FTA/FGA are already in the per-game data.

RestDays/B2B are a different kind of column from everything else here: not a
rolling average of past performance, but a single fact about the gap before
THIS game (days since the team's previous game, and whether it's a
back-to-back). Unlike the rolling-window history, this lookback intentionally
DOES count Pre Season games -- rest is a real calendar/physical fact, so a
team's actual previous game (of any SeasonType) is what determines it, even
though Pre Season games still aren't output as rows. A team's first game on
file (no previous game at all, or a season-opener's multi-month offseason
gap) is treated as fully rested (REST_DAYS_CAP) rather than dropped -- unlike
the rolling stats, this needs no prior-games threshold, so there's no reason
to lose rows over it.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NBAdata"
INPUT_DIR = DATA_ROOT / "team_game_logs"
OUTPUT_DIR = DATA_ROOT / "rolling_stats"

ROLLING_WINDOW = 10
MIN_GAMES_IN_WINDOW = 3
INCLUDE_PRESEASON_IN_WINDOW = False
REST_DAYS_CAP = 5

# model-facing rolling stat name -> column in `working` to roll (source column
# in team_game_logs_<season>.csv, except FTR which is derived -- see docstring)
ROLLING_STAT_SOURCE = {
    "PIE": "PIE",
    "EFG_PCT": "EFG_PCT",
    "TM_TOV_PCT": "TM_TOV_PCT",
    "OREB_PCT": "OREB_PCT",
    "FTR": "FTR",
    "NET_RATING": "NET_RATING",
    "OFF_RATING": "OFF_RATING",
    "DEF_RATING": "DEF_RATING",
    "PACE": "PACE",
}

PASSTHROUGH_COLS = ["TEAM_NAME", "GAME_ID", "GAME_DATE", "Season", "games_in_window"]


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing rolling-average stats per team from a per-game log.

    `df` is expected to look like one season's team_game_logs_<season>.csv
    (any mix of SeasonType rows for one or more teams). Returns one row per
    non-Pre-Season game with the rolling stat columns, `games_in_window`,
    and passthrough identifying columns. Rows with fewer than
    MIN_GAMES_IN_WINDOW valid prior games get NaN for the rolling stats.
    """
    working = df.copy()
    working["GAME_DATE"] = pd.to_datetime(working["GAME_DATE"])
    working = working.sort_values(["TEAM_NAME", "GAME_DATE"]).reset_index(drop=True)
    working["W_PCT"] = (working["WL"] == "W").astype(float)
    working["FTR"] = working["FTA"] / working["FGA"]

    # Rest days since this team's true previous game (any SeasonType -- see
    # docstring). A team's first game on file has nothing to diff against, so
    # it's treated as fully rested rather than dropped.
    prev_date = working.groupby("TEAM_NAME")["GAME_DATE"].shift(1)
    rest_days = (working["GAME_DATE"] - prev_date).dt.days - 1
    working["RestDays"] = rest_days.clip(upper=REST_DAYS_CAP).fillna(REST_DAYS_CAP)
    working["B2B"] = (working["RestDays"] == 0).astype(float)

    # History used to build windows excludes Pre Season (if configured), but
    # the *output* rows are still restricted to non-Pre-Season games below --
    # this filter only controls what counts as prior history.
    if INCLUDE_PRESEASON_IN_WINDOW:
        history = working
    else:
        history = working[working["SeasonType"] != "Pre Season"].copy()

    history = history.sort_values(["TEAM_NAME", "GAME_DATE"]).reset_index(drop=True)

    grouped = history.groupby("TEAM_NAME", group_keys=False)

    source_cols = ["W_PCT"] + list(ROLLING_STAT_SOURCE.values())
    shifted = grouped[source_cols].shift(1)

    rolling_means = shifted.groupby(history["TEAM_NAME"]).rolling(
        ROLLING_WINDOW, min_periods=MIN_GAMES_IN_WINDOW
    ).mean()
    rolling_means = rolling_means.reset_index(level=0, drop=True)

    # Number of prior games available for this team, capped at the window
    # size -- e.g. a team's 1st game has 0 prior games, its 12th game has 10
    # (the window is full by then).
    games_in_window = history.groupby("TEAM_NAME").cumcount().clip(upper=ROLLING_WINDOW)

    result = history[PASSTHROUGH_COLS[:-1] + ["SeasonType", "RestDays", "B2B"]].copy()
    result["games_in_window"] = games_in_window.astype(int)
    for model_col, source_col in ROLLING_STAT_SOURCE.items():
        result[source_col] = rolling_means[source_col]
    result["W_PCT"] = rolling_means["W_PCT"]

    result = result[result["SeasonType"] != "Pre Season"].drop(columns=["SeasonType"])
    result = result.sort_values(["TEAM_NAME", "GAME_DATE"]).reset_index(drop=True)
    result["GAME_DATE"] = result["GAME_DATE"].dt.strftime("%Y-%m-%d")

    ordered_cols = PASSTHROUGH_COLS + ["W_PCT"] + list(ROLLING_STAT_SOURCE.values()) + ["RestDays", "B2B"]
    return result[ordered_cols]


def build_season(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    result = compute_rolling_stats(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(result)} rows)")


def main() -> None:
    for input_path in sorted(INPUT_DIR.glob("team_game_logs_*.csv")):
        suffix = input_path.name[len("team_game_logs_"):]
        output_path = OUTPUT_DIR / f"nba_team_rolling_stats_{suffix}"
        build_season(input_path, output_path)


if __name__ == "__main__":
    main()
