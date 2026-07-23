"""Offline tests for the real-time prediction API.

No network: the model and the team-stats lookup are replaced with in-memory fakes,
mirroring how test_batch_predictions.py mocks the batch job. The rest of the pipeline
(feature resolution, feature assembly) runs for real against a tiny DataFrame.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main


class FakePredictor:
    """Always predicts a home win with 0.7 probability."""

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.3, 0.7]]


class FakePredictorNoProba:
    def predict(self, X):
        return [1]


class FakePredictorAwayWin:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return [[0.6, 0.4]]


def make_stats_df(include_all_stats: bool = True) -> pd.DataFrame:
    """Two teams with April (month 4) stats. Column names are the source names
    resolve_stat_columns expects."""
    row = {
        "TEAM_NAME": ["denver nuggets", "miami heat"],
        "Season": ["2024-25", "2024-25"],
        "Month": [4, 4],
        "W_PCT": [0.7, 0.4],
        "PIE": [0.55, 0.45],
        "EFG_PCT": [0.54, 0.51],
        "TM_TOV_PCT": [0.12, 0.14],
        "OREB_PCT": [0.25, 0.22],
    }
    if include_all_stats:
        row["FT_PCT"] = [0.20, 0.18]  # source column for the FTR feature
    return pd.DataFrame(row)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_predictor", lambda: FakePredictor())
    monkeypatch.setattr(main, "load_team_stats", lambda season_key: make_stats_df())
    monkeypatch.setattr(main, "collect_monthly_files", lambda: {})
    with TestClient(main.app) as c:
        yield c


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(client):
    resp = client.post(
        "/predict",
        json={"home_team": "Denver Nuggets", "away_team": "Miami Heat", "date": "2025-04-01"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_winner"] == "denver nuggets"
    assert body["home_win_probability"] == 0.7


def test_predict_unknown_team_returns_404(client):
    resp = client.post(
        "/predict",
        json={"home_team": "Phantom Team", "away_team": "Miami Heat", "date": "2025-04-01"},
    )
    assert resp.status_code == 404


def test_predict_missing_features_returns_422(monkeypatch):
    # Stats file is missing the FT_PCT column -> the FTR feature can't be built.
    monkeypatch.setattr(main, "load_predictor", lambda: FakePredictor())
    monkeypatch.setattr(main, "load_team_stats", lambda season_key: make_stats_df(include_all_stats=False))
    monkeypatch.setattr(main, "collect_monthly_files", lambda: {})
    with TestClient(main.app) as c:
        resp = c.post(
            "/predict",
            json={"home_team": "Denver Nuggets", "away_team": "Miami Heat", "date": "2025-04-01"},
        )
    assert resp.status_code == 422


def test_predict_without_proba_returns_null_probability(monkeypatch):
    monkeypatch.setattr(main, "load_predictor", lambda: FakePredictorNoProba())
    monkeypatch.setattr(main, "load_team_stats", lambda season_key: make_stats_df())
    monkeypatch.setattr(main, "collect_monthly_files", lambda: {})
    with TestClient(main.app) as c:
        resp = c.post(
            "/predict",
            json={"home_team": "Denver Nuggets", "away_team": "Miami Heat", "date": "2025-04-01"},
        )
    assert resp.status_code == 200
    assert resp.json()["home_win_probability"] is None


def test_predict_malformed_body_returns_422(client):
    resp = client.post(
        "/predict",
        json={"home_team": "", "away_team": "Miami Heat", "date": "2025-04-01"},
    )
    assert resp.status_code == 422


def test_predict_away_team_win(monkeypatch):
    monkeypatch.setattr(main, "load_predictor", lambda: FakePredictorAwayWin())
    monkeypatch.setattr(main, "load_team_stats", lambda season_key: make_stats_df())
    monkeypatch.setattr(main, "collect_monthly_files", lambda: {})
    with TestClient(main.app) as c:
        resp = c.post(
            "/predict",
            json={"home_team": "Denver Nuggets", "away_team": "Miami Heat", "date": "2025-04-01"},
        )
    assert resp.status_code == 200
    assert resp.json()["predicted_winner"] == "miami heat"


def test_predict_missing_season_returns_503(monkeypatch):
    def raise_missing(season_key):
        raise FileNotFoundError("no stats for season")

    monkeypatch.setattr(main, "load_predictor", lambda: FakePredictor())
    monkeypatch.setattr(main, "load_team_stats", raise_missing)
    monkeypatch.setattr(main, "collect_monthly_files", lambda: {})
    with TestClient(main.app) as c:
        resp = c.post(
            "/predict",
            json={"home_team": "Denver Nuggets", "away_team": "Miami Heat", "date": "2025-04-01"},
        )
    assert resp.status_code == 503


def test_health_reports_degraded_when_model_fails_to_load(monkeypatch):
    def raise_no_model():
        raise RuntimeError("no model available")

    monkeypatch.setattr(main, "load_predictor", raise_no_model)
    monkeypatch.setattr(main, "collect_monthly_files", lambda: {})
    with TestClient(main.app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
