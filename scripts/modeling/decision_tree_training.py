from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import SELECTED_FEATURES, resolve_stat_columns

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

TRAINING_DATASET_PATH = DATA_ROOT / "NBA_Training_Matchups_2019_2025.csv"
MODEL_OUTPUT_PATH = DATA_ROOT / "best_model.pkl"
SCALER_OUTPUT_PATH = DATA_ROOT / "scaler.pkl"


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
            previous = season_files.get(season_key)
            if previous is None:
                season_files[season_key] = file_path
                continue
            if "withMonth" in previous.name and "withMonth" not in file_path.name:
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


def build_training_frame(matchup_path: Path, monthly_path: Path, season_key: str) -> pd.DataFrame:
    matchups_df = pd.read_csv(matchup_path)
    stats_df = pd.read_csv(monthly_path)

    matchups_df["Team1"] = matchups_df["Team1"].str.strip().str.lower()
    matchups_df["Team2"] = matchups_df["Team2"].str.strip().str.lower()
    stats_df["TEAM_NAME"] = stats_df["TEAM_NAME"].str.strip().str.lower()

    matchups_df["DATE"] = pd.to_datetime(matchups_df["DATE"], errors="coerce")
    matchups_df["Month"] = matchups_df["DATE"].dt.month
    matchups_df["Season"] = key_to_season_label(season_key)

    stats_df["Month"] = pd.to_numeric(stats_df["Month"], errors="coerce")
    stats_df["Season"] = stats_df["Season"].astype(str)

    resolved_map, selected_cols = resolve_stat_columns(stats_df.columns)

    team1_rename = {"TEAM_NAME": "Team1"}
    team2_rename = {"TEAM_NAME": "Team2"}
    for model_col, source_col in resolved_map.items():
        team1_rename[source_col] = f"Team1_{model_col}"
        team2_rename[source_col] = f"Team2_{model_col}"

    team1_stats = stats_df[selected_cols].rename(columns=team1_rename)
    team2_stats = stats_df[selected_cols].rename(columns=team2_rename)

    merged_df = matchups_df.merge(team1_stats, on=["Team1", "Season", "Month"], how="left")
    merged_df = merged_df.merge(team2_stats, on=["Team2", "Season", "Month"], how="left")
    return merged_df


def build_historical_training_dataset() -> pd.DataFrame:
    matchup_files = collect_matchup_files()
    monthly_files = collect_monthly_files()
    common_seasons = sorted(set(matchup_files).intersection(monthly_files))
    if not common_seasons:
        raise FileNotFoundError(
            "No overlapping season files found between matchup and monthly stats directories."
        )

    season_frames: list[pd.DataFrame] = []
    for season_key in common_seasons:
        frame = build_training_frame(
            matchup_path=matchup_files[season_key],
            monthly_path=monthly_files[season_key],
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
        ),
        "SVC (RBF Kernel)": SVC(gamma="scale", probability=True, class_weight="balanced"),
        "Bagging SVC": BaggingClassifier(estimator=SVC(gamma="scale"), n_estimators=10, random_state=0),
    }

    best_model_name = ""
    best_model = None
    best_cv_score = -1.0

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        if cv_scores.mean() > best_cv_score:
            best_cv_score = cv_scores.mean()
            best_model_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    best_model.fit(X_train_scaled, y_train)
    test_preds = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nBest model: {best_model_name} (CV accuracy: {best_cv_score:.4f})")
    print(f"Held-out test accuracy: {test_acc:.4f}")

    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"Saved best model ({best_model_name}) to {MODEL_OUTPUT_PATH}")
    print(f"Saved scaler to {SCALER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
