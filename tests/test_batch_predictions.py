from datetime import date

import pandas as pd

from run_nightly_predictions import (
    build_feature_row,
    games_on_date,
    latest_team_stat_row,
    season_label_for_date,
)


def test_season_label_for_date_within_season():
    assert season_label_for_date(date(2024, 11, 15)) == "2024-25"
    assert season_label_for_date(date(2025, 3, 1)) == "2024-25"
    assert season_label_for_date(date(2025, 6, 20)) == "2024-25"


def test_season_label_for_date_crosses_july_boundary():
    assert season_label_for_date(date(2025, 7, 1)) == "2025-26"
    assert season_label_for_date(date(2025, 6, 30)) == "2024-25"


def test_games_on_date_filters_and_builds_home_away_names():
    schedule_df = pd.DataFrame({
        "gameDate": ["04/01/2025 00:00:00", "04/02/2025 00:00:00"],
        "gameId": ["001", "002"],
        "homeTeam_teamCity": ["Denver", "Boston"],
        "homeTeam_teamName": ["Nuggets", "Celtics"],
        "awayTeam_teamCity": ["Miami", "New York"],
        "awayTeam_teamName": ["Heat", "Knicks"],
    })

    result = games_on_date(schedule_df, date(2025, 4, 1))

    assert len(result) == 1
    row = result.iloc[0]
    assert row["Team1"] == "denver nuggets"
    assert row["Team2"] == "miami heat"


def test_latest_team_stat_row_prefers_exact_month():
    stats_df = pd.DataFrame({
        "TEAM_NAME": ["denver nuggets", "denver nuggets"],
        "Month": [10, 11],
        "PIE": [0.5, 0.6],
    })
    row = latest_team_stat_row(stats_df, "denver nuggets", 11, ["PIE"])
    assert row["PIE"] == 0.6


def test_latest_team_stat_row_falls_back_when_exact_month_missing():
    stats_df = pd.DataFrame({
        "TEAM_NAME": ["denver nuggets", "denver nuggets"],
        "Month": [10, 11],
        "PIE": [0.5, None],
    })
    row = latest_team_stat_row(stats_df, "denver nuggets", 11, ["PIE"])
    assert row["Month"] == 10
    assert row["PIE"] == 0.5


def test_latest_team_stat_row_returns_none_for_unknown_team():
    stats_df = pd.DataFrame({"TEAM_NAME": ["denver nuggets"], "Month": [10], "PIE": [0.5]})
    assert latest_team_stat_row(stats_df, "phantom team", 10, ["PIE"]) is None


def test_build_feature_row_prefixes_home_and_away_correctly():
    home_row = pd.Series({"W_PCT_base": 0.7, "PIE": 0.55})
    away_row = pd.Series({"W_PCT_base": 0.4, "PIE": 0.45})
    resolved_map = {"W_PCT": "W_PCT_base", "PIE": "PIE"}

    features = build_feature_row(home_row, away_row, resolved_map)

    assert features == {
        "Team1Home": 1,
        "Team1_W_PCT": 0.7,
        "Team2_W_PCT": 0.4,
        "Team1_PIE": 0.55,
        "Team2_PIE": 0.45,
    }
