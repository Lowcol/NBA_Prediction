"""Build per-team daily unavailable-player counts from injury reports.

Reads NBAdata/injury_reports/injury_reports_<season>.csv (one row per
player per daily injury report) and computes, per team per actual game
date, how many of that team's players were reported "Out" or "Doubtful".

Each report's `target_game_date` and `Game Date` are not always the same --
a single report is a rolling 1-2 day window covering multiple game dates, so
rows are first filtered to `Game Date == target_game_date` before counting;
trusting every row under a `target_game_date` to belong to it would double
count players across the window.

Only seasons with an injury_reports_<season>.csv on disk get an output
file. The NBA's injury-report archive only goes back to 2021-22, so
2019-20 and 2020-21 (two of this project's seven training seasons) never
get one -- a known, permanent coverage gap, not a bug.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NBAdata"
INJURY_DIR = DATA_ROOT / "injury_reports"

UNAVAILABLE_STATUSES = {"Out", "Doubtful"}


def compute_injury_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count each team's Out/Doubtful players per actual game date.

    `df` is expected to look like one season's injury_reports_<season>.csv.
    Returns columns GAME_DATE, TEAM_NAME, PlayersOut -- one row per
    (date, team) with at least one Out/Doubtful player that date.
    """
    working = df.copy()
    working["Game Date"] = pd.to_datetime(working["Game Date"], format="%m/%d/%Y").dt.normalize()
    working["target_game_date"] = pd.to_datetime(working["target_game_date"]).dt.normalize()

    # A report covers a rolling window of game dates; only rows whose "Game
    # Date" matches the report's own target date are that date's real
    # snapshot -- see module docstring.
    working = working[working["Game Date"] == working["target_game_date"]]

    unavailable = working[working["Current Status"].isin(UNAVAILABLE_STATUSES)]
    counts = (
        unavailable.groupby(["Game Date", "Team"])
        .size()
        .rename("PlayersOut")
        .reset_index()
        .rename(columns={"Game Date": "GAME_DATE", "Team": "TEAM_NAME"})
    )
    counts["GAME_DATE"] = counts["GAME_DATE"].dt.strftime("%Y-%m-%d")
    return counts.sort_values(["GAME_DATE", "TEAM_NAME"]).reset_index(drop=True)


def build_season(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    result = compute_injury_counts(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(result)} rows)")


def main() -> None:
    for input_path in sorted(INJURY_DIR.glob("injury_reports_*.csv")):
        suffix = input_path.name[len("injury_reports_"):]
        output_path = INJURY_DIR / f"team_injury_counts_{suffix}"
        build_season(input_path, output_path)


if __name__ == "__main__":
    main()
