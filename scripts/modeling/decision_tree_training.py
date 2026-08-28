from pathlib import Path
import re

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import SELECTED_FEATURES, resolve_stat_columns
from mlflow_config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, tracking_uri

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NBAdata"
HISTORICAL_ROOT = DATA_ROOT / "archive" / "historical"

MATCHUP_DIRS = [
    DATA_ROOT / "matchups",
    HISTORICAL_ROOT / "matchups",
]
MONTHLY_DIRS = [
    DATA_ROOT / "monthly_stats",
    HISTORICAL_ROOT / "monthly_stats",
]
ROLLING_DIRS = [
    DATA_ROOT / "rolling_stats",
]
INJURY_DIR = DATA_ROOT / "injury_reports"

TRAINING_DATASET_PATH = DATA_ROOT / "NBA_Training_Matchups_2019_2025.csv"
MODEL_OUTPUT_PATH = DATA_ROOT / "best_model.pkl"
SCALER_OUTPUT_PATH = DATA_ROOT / "scaler.pkl"

# MLflow experiment name (registry model name/alias + tracking store live in
# mlflow_config, shared with the batch job).
MLFLOW_EXPERIMENT_NAME = "nba-win-predictor"


def parse_matchup_season_key(path: Path) -> str | None:
    match = re.search(r"NBA_(\d{4})_(\d{2})_Matchups", path.name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def parse_monthly_season_key(path: Path) -> str | None:
    match = re.search(r"nba_team_combined_stats_(\d{4})_(\d{2})\.csv$", path.name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def parse_rolling_season_key(path: Path) -> str | None:
    match = re.search(r"nba_team_rolling_stats_(\d{4})_(\d{2})\.csv$", path.name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def key_to_season_label(key: str) -> str:
    start, end = key.split("_")
    return f"{start}-{end}"


def collect_matchup_files() -> dict[str, Path]:
    season_files: dict[str, Path] = {}
    for directory in MATCHUP_DIRS:
        if not directory.exists():
            continue
        for file_path in directory.glob("NBA_*_Matchups*.csv"):
            season_key = parse_matchup_season_key(file_path)
            if season_key is None:
                continue
            season_files[season_key] = file_path
    return season_files


def collect_monthly_files() -> dict[str, Path]:
    season_files: dict[str, Path] = {}
    for directory in MONTHLY_DIRS:
        if not directory.exists():
            continue
        for file_path in directory.glob("nba_team_combined_stats_*.csv"):
            season_key = parse_monthly_season_key(file_path)
            if season_key is None:
                continue
            season_files[season_key] = file_path
    return season_files


def collect_rolling_files() -> dict[str, Path]:
    season_files: dict[str, Path] = {}
    for directory in ROLLING_DIRS:
        if not directory.exists():
            continue
        for file_path in directory.glob("nba_team_rolling_stats_*.csv"):
            season_key = parse_rolling_season_key(file_path)
            if season_key is None:
                continue
            season_files[season_key] = file_path
    return season_files


def build_training_frame(matchup_path: Path, rolling_path: Path, season_key: str) -> pd.DataFrame:
    matchups_df = pd.read_csv(matchup_path)
    stats_df = pd.read_csv(rolling_path)

    matchups_df["Team1"] = matchups_df["Team1"].str.strip().str.lower()
    matchups_df["Team2"] = matchups_df["Team2"].str.strip().str.lower()
    stats_df["TEAM_NAME"] = stats_df["TEAM_NAME"].str.strip().str.lower()

    matchups_df["DATE"] = pd.to_datetime(matchups_df["DATE"], errors="coerce")
    matchups_df["DATE_ONLY"] = matchups_df["DATE"].dt.normalize()
    matchups_df["Season"] = key_to_season_label(season_key)

    # resolve_stat_columns() also returns a selected_cols list that unconditionally
    # includes "Month" (a monthly-stats artifact); the rolling-stats file has no
    # Month column, so only resolved_map (model_col -> source_col) is used here.
    resolved_map, _selected_cols = resolve_stat_columns(stats_df.columns)

    team1_rename = {"TEAM_NAME": "Team1"}
    team2_rename = {"TEAM_NAME": "Team2"}
    for model_col, source_col in resolved_map.items():
        team1_rename[source_col] = f"Team1_{model_col}"
        team2_rename[source_col] = f"Team2_{model_col}"

    select_cols = ["TEAM_NAME", "GAME_ID"] + list(resolved_map.values())
    team1_stats = stats_df[select_cols].rename(columns=team1_rename)
    team2_stats = stats_df[select_cols].rename(columns=team2_rename)

    merged_df = matchups_df.merge(team1_stats, on=["Team1", "GAME_ID"], how="left")
    merged_df = merged_df.merge(team2_stats, on=["Team2", "GAME_ID"], how="left")

    # Injury coverage only exists for some seasons (the NBA's injury-report
    # archive starts 2021-22); seasons without a counts file simply don't get
    # this merge, leaving Team1_PlayersOut/Team2_PlayersOut absent from that
    # season's frame -- pd.concat() across seasons then introduces NaN for
    # those rows automatically, which main()'s dropna(subset=selected_features)
    # already handles like any other missing feature.
    injury_path = INJURY_DIR / f"team_injury_counts_{season_key}.csv"
    if injury_path.exists():
        injury_df = pd.read_csv(injury_path)
        injury_df["TEAM_NAME"] = injury_df["TEAM_NAME"].str.strip().str.lower()
        injury_df["GAME_DATE"] = pd.to_datetime(injury_df["GAME_DATE"]).dt.normalize()

        team1_injury = injury_df.rename(
            columns={"TEAM_NAME": "Team1", "GAME_DATE": "DATE_ONLY", "PlayersOut": "Team1_PlayersOut"}
        )
        team2_injury = injury_df.rename(
            columns={"TEAM_NAME": "Team2", "GAME_DATE": "DATE_ONLY", "PlayersOut": "Team2_PlayersOut"}
        )

        merged_df = merged_df.merge(team1_injury, on=["Team1", "DATE_ONLY"], how="left")
        merged_df = merged_df.merge(team2_injury, on=["Team2", "DATE_ONLY"], how="left")
        # A team with a real game that day but no Out/Doubtful report rows
        # genuinely has 0 known-unavailable players -- fillna(0), not NaN.
        merged_df[["Team1_PlayersOut", "Team2_PlayersOut"]] = (
            merged_df[["Team1_PlayersOut", "Team2_PlayersOut"]].fillna(0)
        )

    merged_df = merged_df.drop(columns=["DATE_ONLY"])
    return merged_df


def build_historical_training_dataset() -> pd.DataFrame:
    matchup_files = collect_matchup_files()
    rolling_files = collect_rolling_files()
    common_seasons = sorted(set(matchup_files).intersection(rolling_files))
    if not common_seasons:
        raise FileNotFoundError(
            "No overlapping season files found between matchup and rolling stats directories."
        )

    season_frames: list[pd.DataFrame] = []
    for season_key in common_seasons:
        frame = build_training_frame(
            matchup_path=matchup_files[season_key],
            rolling_path=rolling_files[season_key],
            season_key=season_key,
        )
        frame["SeasonKey"] = season_key
        season_frames.append(frame)
        print(f"Loaded season {season_key}: {len(frame)} rows")

    dataset = pd.concat(season_frames, ignore_index=True)
    TRAINING_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(TRAINING_DATASET_PATH, index=False)
    print(f"Saved combined training dataset: {TRAINING_DATASET_PATH} ({len(dataset)} rows)")
    return dataset


def chronological_train_test_split(
    df: pd.DataFrame, season_col: str = "SeasonKey", date_col: str = "DATE", test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each season's games by date instead of randomly: a season's earlier
    games go to train, its later games (including playoffs) go to test.

    Splitting per season -- rather than across the full multi-season date range
    -- keeps train and test balanced across eras (every season contributes to
    both), and matches how the model is actually used in-season: given a team's
    form through some point in the season, predict its remaining games.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, season_df in df.groupby(season_col, sort=False):
        season_df = season_df.sort_values(date_col)
        split_idx = round(len(season_df) * (1 - test_size))
        train_parts.append(season_df.iloc[:split_idx])
        test_parts.append(season_df.iloc[split_idx:])
    return pd.concat(train_parts), pd.concat(test_parts)


def build_model_grids() -> dict[str, tuple]:
    """Each model paired with a small hyperparameter grid for GridSearchCV.

    Grids are deliberately small: ~6k rows of noisy data can't support fine-grained
    tuning, and SVC-based fits are expensive. The former hand-picked settings are
    all included in their model's grid, so tuning can only match or beat them.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 8], "min_samples_leaf": [1, 20, 50]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=42),
            {"max_depth": [5, 10, None], "min_samples_leaf": [1, 10]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss"),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3, 6],
            },
        ),
        "SVC (RBF Kernel)": (
            SVC(class_weight="balanced"),
            {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]},
        ),
        "Bagging SVC": (
            BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0),
            {"estimator__C": [0.1, 1, 10], "estimator__gamma": ["scale", 0.1]},
        ),
    }


# The SVC-based models are tuned with probability=False: accuracy scoring only uses
# predict(), which calibration doesn't change, while probability=True adds an internal
# 5-fold calibration to every grid fit (~5x cost). The winning params are refit with
# probability=True below so serving gets real predict_proba output (without it,
# BaggingClassifier.predict_proba degrades to counting the 10 models' hard votes ->
# near-0/1 "probabilities").
PROBABILITY_OVERRIDES = {
    "SVC (RBF Kernel)": {"probability": True},
    "Bagging SVC": {"estimator__probability": True},
}


def main() -> None:
    df = build_historical_training_dataset()
    target = "Team1Win"
    selected_features = SELECTED_FEATURES

    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required features in merged dataset: {missing_features}")

    df = df.dropna(subset=selected_features + [target])
    if df.empty:
        raise ValueError("No rows available after dropping missing values for selected features.")

    X = df[selected_features]
    y = df[target]

    train_df, test_df = chronological_train_test_split(df, test_size=0.2)
    X_train, y_train = train_df[selected_features], train_df[target]
    X_test, y_test = test_df[selected_features], test_df[target]
    print(
        f"Chronological split: train={len(X_train)} rows "
        f"({train_df['DATE'].min().date()} -> {train_df['DATE'].max().date()}), "
        f"test={len(X_test)} rows "
        f"({test_df['DATE'].min().date()} -> {test_df['DATE'].max().date()})"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Class distribution (Team1Win):")
    print(y.value_counts(normalize=True))

    majority_baseline = max(y.mean(), 1 - y.mean())
    home_baseline = (X["Team1Home"] == y).mean()
    print(f"Majority-class baseline accuracy: {majority_baseline:.4f}")
    print(f"Home-team-always-wins baseline accuracy: {home_baseline:.4f}")

    models = build_model_grids()

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_model_name = ""
    best_model = None
    best_cv_score = -1.0

    with mlflow.start_run(run_name="training-run"):
        mlflow.log_param("n_features", len(selected_features))
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))
        mlflow.log_metric("majority_baseline_accuracy", majority_baseline)
        mlflow.log_metric("home_court_baseline_accuracy", home_baseline)

        for name, (estimator, param_grid) in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                # refit=False: we only need the scores/params here; the winning
                # config is refit once below (with the probability override).
                search = GridSearchCV(
                    estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1, refit=False
                )
                search.fit(X_train_scaled, y_train)
                cv_mean = search.best_score_
                cv_std = search.cv_results_["std_test_score"][search.best_index_]
                print(
                    f"{name} best CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f}) "
                    f"with {search.best_params_}"
                )
                mlflow.log_param("model_type", name)
                mlflow.log_param("n_grid_candidates", len(search.cv_results_["params"]))
                mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
                mlflow.log_metric("cv_accuracy_mean", cv_mean)
                mlflow.log_metric("cv_accuracy_std", cv_std)
            if cv_mean > best_cv_score:
                best_cv_score = cv_mean
                best_model_name = name
                best_model = clone(estimator).set_params(**search.best_params_)

        if best_model is None:
            raise RuntimeError("No model was trained successfully.")

        if best_model_name in PROBABILITY_OVERRIDES:
            best_model.set_params(**PROBABILITY_OVERRIDES[best_model_name])
        best_model.fit(X_train_scaled, y_train)
        test_preds = best_model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        print(f"\nBest model: {best_model_name} (CV accuracy: {best_cv_score:.4f})")
        print(f"Held-out test accuracy: {test_acc:.4f}")

        joblib.dump(best_model, MODEL_OUTPUT_PATH)
        joblib.dump(scaler, SCALER_OUTPUT_PATH)
        print(f"Saved best model ({best_model_name}) to {MODEL_OUTPUT_PATH}")
        print(f"Saved scaler to {SCALER_OUTPUT_PATH}")

        # Package the already-fit scaler + model as one raw-features -> prediction
        # pipeline and register it. Serving loads this single object, so scaling
        # can't drift from the model it was fit alongside.
        pipeline = Pipeline([("scaler", scaler), ("model", best_model)])
        signature = infer_signature(X_test, pipeline.predict(X_test))

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("cv_accuracy", best_cv_score)
        mlflow.log_metric("test_accuracy", test_acc)

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_test.iloc[:2],
            registered_model_name=REGISTERED_MODEL_NAME,
            # MLflow's skops serialization only trusts sklearn types by default; when
            # XGBoost wins the bake-off, its classes must be trust-listed or the save is
            # refused. Safe here (we just trained this model ourselves), and the list is
            # stored in the model's flavor config so load_model reuses it automatically.
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
        )

        MlflowClient().set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, model_info.registered_model_version
        )
        print(
            f"Registered {REGISTERED_MODEL_NAME} v{model_info.registered_model_version} "
            f"and set alias @{PRODUCTION_ALIAS}"
        )


if __name__ == "__main__":
    main()
