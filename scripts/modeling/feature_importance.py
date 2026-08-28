"""SHAP feature-importance audit for the current @production model.

Ad hoc analysis, not part of the training/serving pipeline (same category as
data_exploration.py). Treats the model as a black box (predict_proba) via
shap.Explainer's permutation backend, so it works regardless of which of the
6 candidate models is currently winning.

Needs `shap` installed separately -- NOT a pinned project dependency. It
pulls in numba/llvmlite and forces numpy>=2, which conflicts with
requirements.txt's numpy==1.26.4 pin (needed for the rest of the stack), so
adding it to requirements.txt would destabilize training/serving. Run
`pip install shap` yourself before using this script, then reinstall
numpy==1.26.4 afterward to restore the pinned environment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "modeling"))
sys.path.insert(0, str(PROJECT_ROOT / "serving" / "inference"))

from decision_tree_training import build_historical_training_dataset, chronological_train_test_split  # noqa: E402
from features import SELECTED_FEATURES  # noqa: E402
from predictor import load_predictor  # noqa: E402

BACKGROUND_SIZE = 100
SAMPLE_SIZE = 300


def main() -> None:
    df = build_historical_training_dataset()
    df = df.dropna(subset=SELECTED_FEATURES + ["Team1Win"])
    train_df, test_df = chronological_train_test_split(df, test_size=0.2)
    X_train = train_df[SELECTED_FEATURES]
    X_test = test_df[SELECTED_FEATURES]

    model = load_predictor()

    background = shap.sample(X_train, BACKGROUND_SIZE, random_state=42)
    sample = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=42)

    explainer = shap.Explainer(model.predict_proba, background)
    shap_values = explainer(sample)

    # predict_proba has 2 output columns (P(loss), P(win)); explain the win column.
    win_values = shap_values[..., 1].values
    importance = pd.Series(np.abs(win_values).mean(axis=0), index=SELECTED_FEATURES)
    importance = importance.sort_values(ascending=False)

    print(f"Mean |SHAP value| per feature (home-win probability output, n={len(sample)}):\n")
    print(importance.to_string())


if __name__ == "__main__":
    main()
