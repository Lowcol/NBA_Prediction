from datetime import date

import pandas as pd
from sklearn.pipeline import Pipeline

import run_nightly_predictions
from run_nightly_predictions import (
    build_feature_row,
    games_on_date,
    latest_team_stat_row,
    load_predictor,
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
    # Months cross the season boundary (Dec=12 then Mar=3); the chronologically-latest
    # usable month is March, not December, so a plain numeric sort would pick the wrong row.
    stats_df = pd.DataFrame({
        "TEAM_NAME": ["denver nuggets", "denver nuggets", "denver nuggets"],
        "Month": [12, 3, 4],
        "PIE": [0.5, 0.6, None],
    })
    row = latest_team_stat_row(stats_df, "denver nuggets", 4, ["PIE"])
    assert row["Month"] == 3
    assert row["PIE"] == 0.6


def test_latest_team_stat_row_returns_none_for_unknown_team():
    stats_df = pd.DataFrame({"TEAM_NAME": ["denver nuggets"], "Month": [10], "PIE": [0.5]})
    assert latest_team_stat_row(stats_df, "phantom team", 10, ["PIE"]) is None


def test_season_label_for_date_january_belongs_to_prior_start_year():
    # January is month < 7, so it belongs to the season that started the prior year.
    assert season_label_for_date(date(2025, 1, 15)) == "2024-25"


def test_games_on_date_returns_empty_when_no_games_scheduled():
    schedule_df = pd.DataFrame({
        "gameDate": ["04/02/2025 00:00:00"],
        "gameId": ["002"],
        "homeTeam_teamCity": ["Boston"],
        "homeTeam_teamName": ["Celtics"],
        "awayTeam_teamCity": ["New York"],
        "awayTeam_teamName": ["Knicks"],
    })

    result = games_on_date(schedule_df, date(2025, 4, 1))

    assert result.empty


def test_latest_team_stat_row_returns_none_when_all_rows_are_nan():
    # Team is present but every month's stat is missing -> no usable row.
    stats_df = pd.DataFrame({
        "TEAM_NAME": ["denver nuggets", "denver nuggets"],
        "Month": [10, 11],
        "PIE": [None, None],
    })
    assert latest_team_stat_row(stats_df, "denver nuggets", 11, ["PIE"]) is None


def test_load_predictor_returns_registry_model_when_available(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(run_nightly_predictions.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        run_nightly_predictions.mlflow.sklearn, "load_model", lambda uri: sentinel
    )
    # If the registry loads, the local pkls must not be touched.
    monkeypatch.setattr(
        run_nightly_predictions.joblib,
        "load",
        lambda path: (_ for _ in ()).throw(AssertionError("pkl fallback should not run")),
    )

    assert load_predictor() is sentinel


def test_load_predictor_falls_back_to_local_pkls_when_registry_unavailable(monkeypatch):
    fake_scaler = object()
    fake_model = object()

    def raise_unavailable(uri):
        raise RuntimeError("registry not reachable")

    def fake_joblib_load(path):
        return fake_scaler if path == run_nightly_predictions.SCALER_PATH else fake_model

    monkeypatch.setattr(run_nightly_predictions.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(run_nightly_predictions.mlflow.sklearn, "load_model", raise_unavailable)
    monkeypatch.setattr(run_nightly_predictions.joblib, "load", fake_joblib_load)

    predictor = load_predictor()

    assert isinstance(predictor, Pipeline)
    assert predictor.named_steps["scaler"] is fake_scaler
    assert predictor.named_steps["model"] is fake_model


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
