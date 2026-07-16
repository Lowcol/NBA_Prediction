from pathlib import Path

import joblib
import pandas as pd

from features import SELECTED_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "NBAdata" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "NBAdata" / "scaler.pkl"


def test_model_and_scaler_files_exist():
    assert MODEL_PATH.exists(), "best_model.pkl is missing — run decision_tree_training.py"
    assert SCALER_PATH.exists(), "scaler.pkl is missing — run decision_tree_training.py"


def test_scaler_expects_exactly_the_selected_features():
    scaler = joblib.load(SCALER_PATH)
    assert scaler.n_features_in_ == len(SELECTED_FEATURES)


def test_model_predicts_on_a_well_formed_feature_row():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    row = pd.DataFrame([{feature: 0.5 for feature in SELECTED_FEATURES}])
    row["Team1Home"] = 1

    X_scaled = scaler.transform(row[SELECTED_FEATURES])
    prediction = model.predict(X_scaled)

    assert prediction[0] in (0, 1)
