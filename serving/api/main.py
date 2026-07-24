"""Real-time prediction API.

Serves single-game win predictions from the same @production model and feature
contract the nightly batch job uses. Given two team names and a date, it looks up
each team's latest stats server-side, builds the model's features, and returns the
predicted winner plus a win probability.
"""

import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "modeling"))
sys.path.insert(0, str(PROJECT_ROOT / "serving" / "inference"))

from features import SELECTED_FEATURES, resolve_stat_columns  # noqa: E402
from predictor import (  # noqa: E402
    assemble_features,
    collect_monthly_files,
    load_predictor,
    load_team_stats,
    predict_from_features,
    season_label_for_date,
)

from models import PredictRequest, PredictResponse  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup (a registry/pkl load is expensive; doing it
    # per request would re-hit MLflow on every call). If it fails, start anyway
    # and report it via /health (degraded) rather than refusing to boot.
    try:
        app.state.predictor = load_predictor()
    except Exception as exc:
        app.state.predictor = None
        print(f"Model load failed at startup: {exc}")
    yield


app = FastAPI(title="NBA win predictor", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    model_loaded = getattr(app.state, "predictor", None) is not None
    stats_available = season_label_for_date(date.today()).replace("-", "_") in collect_monthly_files()
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "stats_available": stats_available,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    home = req.home_team.strip().lower()
    away = req.away_team.strip().lower()
    target = req.date or date.today()
    season_key = season_label_for_date(target).replace("-", "_")

    # If the target date's season has no stats file (e.g. the current season isn't
    # provisioned yet), fall back to the most recent season on file so the UI's
    # date-less "just pick two teams" request still returns a prediction.
    available = collect_monthly_files()
    if season_key not in available:
        if not available:
            raise HTTPException(
                status_code=503,
                detail="No monthly stats files are provisioned. Run the data_pull + "
                "merge_advanced_base_stats.py scripts (or `dvc pull`) first.",
            )
        season_key = max(available)  # keys look like "2024_25"; max() is the latest season

    try:
        stats_df = load_team_stats(season_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    resolved_map, selected_cols = resolve_stat_columns(stats_df.columns)
    resolved_cols = [c for c in selected_cols if c not in ("TEAM_NAME", "Season", "Month")]

    features = assemble_features(stats_df, resolved_map, resolved_cols, home, away, target.month)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No usable stats on file for '{home}' and/or '{away}' in season {season_key}.",
        )

    missing = [f for f in SELECTED_FEATURES if f not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Stats present but missing required features: {missing}.",
        )

    model = getattr(app.state, "predictor", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    pred, home_win_prob = predict_from_features(model, features)
    return PredictResponse(
        home_team=home,
        away_team=away,
        date=target,
        predicted_winner=home if pred == 1 else away,
        home_win_probability=home_win_prob,
    )
