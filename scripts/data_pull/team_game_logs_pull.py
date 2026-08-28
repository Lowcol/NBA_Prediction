"""Pulls per-game team box scores (Base + Advanced) for every season, via
nba_api's TeamGameLogs -- one bulk call per (season, season_type, measure_type),
not one call per game. This is the per-game-granularity counterpart to
monthly_stats_pull.py (which pulls month-granularity aggregates instead); it
feeds scripts/data_prep/build_rolling_team_stats.py's trailing-window features.

Output: NBAdata/team_game_logs/team_game_logs_<season>.csv, one row per team
per game, Base + Advanced stats merged, all season types interleaved and
sorted chronologically.
"""
import os
import time

import pandas as pd
from curl_cffi import requests as cr
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.library.http import NBAStatsHTTP

# --- THE NBA TLS BYPASS (same pattern as nbaPull_19-25_matchups.py) ---
session = cr.Session(impersonate="chrome120")
print("Warming up Akamai cookies...")
session.get("https://www.nba.com/stats/", timeout=20)
NBAStatsHTTP.get_session = lambda self: session
# ------------------------------------------------------------------

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPES = ["Pre Season", "Regular Season", "PlayIn", "Playoffs"]

BASE_COLUMNS = [
    "TEAM_ID", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS", "PLUS_MINUS",
]
ADVANCED_COLUMNS = [
    "TEAM_ID", "GAME_ID",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "AST_TO", "AST_RATIO",
    "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT",
    "EFG_PCT", "TS_PCT", "PACE", "POSS", "PIE",
]


def fetch(season: str, season_type: str, measure_type: str) -> pd.DataFrame:
    logs = teamgamelogs.TeamGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        measure_type_player_game_logs_nullable=measure_type,
    )
    time.sleep(2)
    return logs.get_data_frames()[0]


def fetch_season(season: str) -> pd.DataFrame:
    season_type_frames = []
    for season_type in SEASON_TYPES:
        print(f"   - {season_type}...")
        try:
            base_df = fetch(season, season_type, "Base")
            adv_df = fetch(season, season_type, "Advanced")
        except Exception as e:
            print(f"     Error: {e}")
            continue
        if base_df.empty:
            print("     No data")
            continue

        base_df = base_df[BASE_COLUMNS].copy()
        adv_df = adv_df[ADVANCED_COLUMNS].copy()
        merged = base_df.merge(adv_df, on=["TEAM_ID", "GAME_ID"], how="left")
        merged["SeasonType"] = season_type
        season_type_frames.append(merged)

    if not season_type_frames:
        return pd.DataFrame()

    season_df = pd.concat(season_type_frames, ignore_index=True)
    season_df["TEAM_NAME"] = season_df["TEAM_NAME"].str.strip().str.lower()
    season_df["GAME_DATE"] = pd.to_datetime(season_df["GAME_DATE"])
    season_df["Season"] = season
    season_df = season_df.sort_values(["TEAM_NAME", "GAME_DATE"]).reset_index(drop=True)
    return season_df


def main() -> None:
    os.makedirs("NBAdata/team_game_logs", exist_ok=True)
    for season in SEASONS:
        print(f"\nFetching {season} team game logs...")
        season_df = fetch_season(season)
        if season_df.empty:
            print(f"   Warning: no data collected for {season}")
            continue
        filename = f"NBAdata/team_game_logs/team_game_logs_{season.replace('-', '_')}.csv"
        season_df.to_csv(filename, index=False)
        print(f"Saved: {filename} ({len(season_df)} rows, {season_df['GAME_ID'].nunique()} games)")


if __name__ == "__main__":
    main()
