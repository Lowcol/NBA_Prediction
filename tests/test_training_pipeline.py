from pathlib import Path

import pandas as pd

import decision_tree_training
from decision_tree_training import (
    collect_matchup_files,
    collect_monthly_files,
    key_to_season_label,
    parse_matchup_season_key,
    parse_monthly_season_key,
)
from features import SELECTED_FEATURES

EXPECTED_SEASON_KEYS = {"2019_20", "2020_21", "2021_22", "2022_23", "2023_24", "2024_25"}


def test_parse_matchup_season_key():
    assert parse_matchup_season_key(Path("NBA_2024_25_Matchups.csv")) == "2024_25"
    assert parse_matchup_season_key(Path("not_a_matchup_file.csv")) is None


def test_parse_monthly_season_key():
    assert parse_monthly_season_key(Path("nba_team_combined_stats_2024_25.csv")) == "2024_25"
    assert parse_monthly_season_key(Path("something_else.csv")) is None


def test_key_to_season_label():
    assert key_to_season_label("2024_25") == "2024-25"


def test_collect_matchup_files_finds_all_seasons():
    season_files = collect_matchup_files()
    assert EXPECTED_SEASON_KEYS.issubset(season_files.keys())


def test_collect_monthly_files_finds_all_seasons():
    season_files = collect_monthly_files()
    assert EXPECTED_SEASON_KEYS.issubset(season_files.keys())


def test_monthly_files_have_correct_internal_season_label():
    # Regression guard: every combined-stats file used to internally claim
    # Season == "2019-20" regardless of its filename (see PROGRESS.md).
    season_files = collect_monthly_files()
    for season_key, path in season_files.items():
        expected_label = key_to_season_label(season_key)
        stats_df = pd.read_csv(path, usecols=["Season"])
        actual_labels = set(stats_df["Season"].astype(str).unique())
        assert actual_labels == {expected_label}, (
            f"{path} claims Season {actual_labels}, expected {{{expected_label!r}}}"
        )


def test_training_dataset_includes_all_seasons_with_no_missing_features(tmp_path, monkeypatch):
    # build_historical_training_dataset() writes the combined dataset to disk
    # as a side effect; redirect that to a temp path so the test doesn't
    # overwrite the real NBAdata/NBA_Training_Matchups_2019_2025.csv.
    monkeypatch.setattr(
        decision_tree_training, "TRAINING_DATASET_PATH", tmp_path / "training_dataset.csv"
    )

    dataset = decision_tree_training.build_historical_training_dataset()

    assert EXPECTED_SEASON_KEYS.issubset(set(dataset["SeasonKey"].unique()))

    non_null = dataset.dropna(subset=SELECTED_FEATURES + ["Team1Win"])
    rows_per_season = non_null.groupby("SeasonKey").size()

    # Regression guard: the season-mislabeling bug caused every season
    # except 2019-20 to be silently dropped here by the dropna above (i.e.
    # rows_per_season == 0 for 5 of 6 seasons). Each season has 1,100+ raw
    # matchups, so 100 is a conservative floor: comfortably below normal
    # season-to-season variation in how many rows survive dropna, but high
    # enough that a season being wiped out again would still fail this.
    for season_key in EXPECTED_SEASON_KEYS:
        assert rows_per_season.get(season_key, 0) > 100, (
            f"season {season_key} contributed too few usable training rows "
            "(possible join/labeling regression)"
        )
