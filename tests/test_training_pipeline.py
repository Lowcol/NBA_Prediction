from pathlib import Path

import pandas as pd

import decision_tree_training
from decision_tree_training import (
    build_training_frame,
    chronological_train_test_split,
    collect_matchup_files,
    collect_monthly_files,
    collect_rolling_files,
    key_to_season_label,
    parse_matchup_season_key,
    parse_monthly_season_key,
    parse_rolling_season_key,
)
from features import SELECTED_FEATURES

EXPECTED_SEASON_KEYS = {"2019_20", "2020_21", "2021_22", "2022_23", "2023_24", "2024_25"}


def test_parse_matchup_season_key():
    assert parse_matchup_season_key(Path("NBA_2024_25_Matchups.csv")) == "2024_25"
    assert parse_matchup_season_key(Path("not_a_matchup_file.csv")) is None


def test_parse_monthly_season_key():
    assert parse_monthly_season_key(Path("nba_team_combined_stats_2024_25.csv")) == "2024_25"
    assert parse_monthly_season_key(Path("something_else.csv")) is None


def test_parse_rolling_season_key():
    assert parse_rolling_season_key(Path("nba_team_rolling_stats_2024_25.csv")) == "2024_25"
    assert parse_rolling_season_key(Path("something_else.csv")) is None


def test_key_to_season_label():
    assert key_to_season_label("2024_25") == "2024-25"


def test_collect_matchup_files_finds_all_seasons():
    season_files = collect_matchup_files()
    assert EXPECTED_SEASON_KEYS.issubset(season_files.keys())


def test_collect_monthly_files_finds_all_seasons():
    season_files = collect_monthly_files()
    assert EXPECTED_SEASON_KEYS.issubset(season_files.keys())


def test_collect_rolling_files_finds_all_seasons():
    season_files = collect_rolling_files()
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


def test_chronological_train_test_split_orders_within_each_season():
    # Two seasons, 10 games each, dates strictly increasing within a season.
    # A row's "value" is its position in the season so leakage is obvious.
    df = pd.DataFrame({
        "SeasonKey": ["2019_20"] * 10 + ["2020_21"] * 10,
        "DATE": pd.to_datetime(
            [f"2019-10-{d:02d}" for d in range(1, 11)]
            + [f"2020-10-{d:02d}" for d in range(1, 11)]
        ),
        "value": list(range(10)) + list(range(10)),
    })

    train_df, test_df = chronological_train_test_split(df, test_size=0.2)

    # 80/20 per season -> 8 train + 2 test per season = 16 train, 4 test total.
    assert len(train_df) == 16
    assert len(test_df) == 4

    # Both seasons must contribute to both splits (not just the newest one).
    assert set(train_df["SeasonKey"].unique()) == {"2019_20", "2020_21"}
    assert set(test_df["SeasonKey"].unique()) == {"2019_20", "2020_21"}

    # Within each season, every train row's date is earlier than every test row's.
    for season_key in ("2019_20", "2020_21"):
        train_dates = train_df.loc[train_df["SeasonKey"] == season_key, "DATE"]
        test_dates = test_df.loc[test_df["SeasonKey"] == season_key, "DATE"]
        assert train_dates.max() < test_dates.min()


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

    # 2019-20/2020-21 have no injury-report coverage (the NBA's injury-report
    # archive only goes back to 2021-22), so Team1_PlayersOut/Team2_PlayersOut
    # are NaN for their rows and dropna() above correctly drops them entirely.
    SEASONS_WITHOUT_INJURY_DATA = {"2019_20", "2020_21"}

    # Regression guard: with the (Team, GAME_ID) rolling-window join, dropna
    # only removes each team's own first MIN_GAMES_IN_WINDOW-1 games of a
    # season (no window yet), not a whole month's worth of games as with the
    # old (Team, Season, Month) join -- so drop rates are much lower than
    # they used to be. A real run against the live data (2026-08-28) landed
    # at 1,095-1,275 rows/season; 1,000 is a floor comfortably below that
    # range but high enough that a join/labeling regression (e.g. back to
    # month-level drop rates) would still fail this.
    for season_key in EXPECTED_SEASON_KEYS - SEASONS_WITHOUT_INJURY_DATA:
        assert rows_per_season.get(season_key, 0) > 1000, (
            f"season {season_key} contributed too few usable training rows "
            "(possible join/labeling regression)"
        )

    for season_key in SEASONS_WITHOUT_INJURY_DATA:
        assert rows_per_season.get(season_key, 0) == 0, (
            f"season {season_key} has no injury-counts file and should be fully "
            "dropped by dropna() once PlayersOut is a selected feature"
        )


def _write_minimal_matchup_and_rolling(tmp_path, season_key, game_id, date, team1, team2):
    matchup_path = tmp_path / f"matchups_{season_key}.csv"
    rolling_path = tmp_path / f"rolling_{season_key}.csv"

    pd.DataFrame({
        "GAME_ID": [game_id],
        "DATE": [date],
        "Team1": [team1],
        "Team2": [team2],
        "Team1Home": [1],
        "Team1Win": [1],
    }).to_csv(matchup_path, index=False)

    pd.DataFrame({
        "TEAM_NAME": [team1, team2],
        "GAME_ID": [game_id, game_id],
        "GAME_DATE": [date, date],
        "Season": [key_to_season_label(season_key)] * 2,
        "PIE": [0.1, 0.2],
    }).to_csv(rolling_path, index=False)

    return matchup_path, rolling_path


def test_build_training_frame_merges_injury_counts_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_tree_training, "INJURY_DIR", tmp_path)

    matchup_path, rolling_path = _write_minimal_matchup_and_rolling(
        tmp_path, "2024_25", game_id=1, date="2024-11-12",
        team1="atlanta hawks", team2="boston celtics",
    )

    # atlanta has 2 Out/Doubtful players reported on 2024-11-12; boston has
    # no report row at all that day, which should come back as 0, not NaN.
    pd.DataFrame({
        "GAME_DATE": ["2024-11-12"],
        "TEAM_NAME": ["atlanta hawks"],
        "PlayersOut": [2],
    }).to_csv(tmp_path / "team_injury_counts_2024_25.csv", index=False)

    result = build_training_frame(matchup_path, rolling_path, "2024_25")

    row = result.iloc[0]
    assert row["Team1_PlayersOut"] == 2
    assert row["Team2_PlayersOut"] == 0
    assert "DATE_ONLY" not in result.columns


def test_build_training_frame_leaves_playersout_absent_when_no_injury_file(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_tree_training, "INJURY_DIR", tmp_path)

    # No team_injury_counts_2019_20.csv is written to INJURY_DIR (tmp_path).
    matchup_path, rolling_path = _write_minimal_matchup_and_rolling(
        tmp_path, "2019_20", game_id=1, date="2019-11-12",
        team1="atlanta hawks", team2="boston celtics",
    )

    result = build_training_frame(matchup_path, rolling_path, "2019_20")

    assert "Team1_PlayersOut" not in result.columns
    assert "Team2_PlayersOut" not in result.columns
    assert "DATE_ONLY" not in result.columns
