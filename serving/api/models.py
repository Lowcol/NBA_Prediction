"""Request/response shapes for the real-time prediction API."""

from datetime import date as _date  # aliased so the `date` field name can't shadow the type

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    home_team: str = Field(min_length=1)  # full "City Name", e.g. "Denver Nuggets"
    away_team: str = Field(min_length=1)
    date: _date | None = None  # slate date; defaults to today when omitted


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    date: _date
    predicted_winner: str
    home_win_probability: float | None  # null when the model has no predict_proba
