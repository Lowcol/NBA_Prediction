# How the model is trained (and what it learns from)

This explains, in plain language, how the NBA win predictor is built: where the
data comes from, the exact features fed to the model, and each step of the
training process. It reflects the code in
`scripts/modeling/decision_tree_training.py` and the shared feature contract in
`scripts/modeling/features.py`.

---

## 1. What the model predicts

One thing: **given two teams in a specific game, will the home team win?**

It's a yes/no (binary) prediction. The model also returns a probability — e.g.
"home team wins with 61% confidence" — not just the winner.

---

## 2. The data

- **Source:** historical NBA games and monthly team stats pulled from the NBA
  API, stored as CSVs under `NBAdata/` (tracked with DVC, not committed to git —
  run `dvc pull` to fetch them).
- **Seasons:** seven, from **2019-20 through 2025-26**.
- **Size:** about **8,900 games** total (roughly 1,100–1,300 per season).
- **One row = one game.** Each row pairs the two teams' season-to-date stats
  (as of that game's month) with the actual outcome.

The combined training table is assembled by
`build_historical_training_dataset()`, which joins each season's game list
(`NBAdata/matchups/`) to that season's monthly team-stats file
(`NBAdata/monthly_stats/`) and saves the result to
`NBAdata/NBA_Training_Matchups_2019_2025.csv`.

### The "Team1 / Team2" convention

In the training data, the two teams in a game are labeled **Team1** and
**Team2** *alphabetically* — that ordering carries no meaning about who is home.
Home-court is encoded separately by a single feature, **`Team1Home`** (1 if
Team1 is the home team, 0 otherwise).

> Note: in live serving (`serving/`), the code always assigns **Team1 = the home
> team**, so `Team1Home` is always 1 and the model's output reads directly as
> "probability the home team wins." Same model, simpler convention at prediction
> time.

---

## 3. The features (the 13 inputs)

Every prediction is based on **13 numbers**. For each team we take 6 season-level
stats, plus the one home-court flag: 6 + 6 + 1 = 13. These are defined in
`features.py` as `SELECTED_FEATURES`.

### The 6 per-team stats (each provided for Team1 and Team2)

| Feature name | What it measures (plain language) | Source column |
|---|---|---|
| `W_PCT` | **Win percentage** — the team's win rate so far. | `W_PCT` |
| `PIE` | **Player Impact Estimate** — the share of everything good in a game (points, rebounds, assists, steals…) the team accounts for. A single "how good overall" number. | `PIE` |
| `eFG%` | **Effective field-goal %** — shooting accuracy that gives extra credit for 3-pointers being worth more. | `EFG_PCT` |
| `TOV%` | **Turnover %** — how often possessions are lost to turnovers (lower is better). | `TM_TOV_PCT` |
| `ORB%` | **Offensive rebound %** — share of available offensive rebounds grabbed (second chances). | `OREB_PCT` |
| `FTR` | **Free-throw factor** — how much the team gets to (and converts) the free-throw line. | `FT_PCT` |

The four shooting/possession stats (`eFG%`, `TOV%`, `ORB%`, `FTR`) are the
well-known **"Four Factors"** of basketball — the aspects of play most tied to
winning. `W_PCT` and `PIE` add overall quality.

> Implementation detail: the "source column" names above can vary slightly
> between stat files (e.g. `W_PCT` vs `W_PCT_base`). `resolve_stat_columns()` in
> `features.py` maps each model-facing name to whichever column actually exists
> in a given file, so the contract stays stable even if the raw data's column
> names differ. `FTR` is currently sourced from the `FT_PCT` column.

### The 1 game-context feature

| Feature name | Meaning |
|---|---|
| `Team1Home` | 1 if Team1 is playing at home, else 0. This is how home-court advantage enters the model. |

### Full list (as fed to the model, in order)

```
Team1_W_PCT, Team2_W_PCT, Team1Home,
Team1_PIE,  Team1_eFG%, Team1_TOV%, Team1_ORB%, Team1_FTR,
Team2_PIE,  Team2_eFG%, Team2_TOV%, Team2_ORB%, Team2_FTR
```

**What's deliberately NOT included:** anything only known *after* the game
(final score, plus/minus, points scored). Including those would be "leakage" —
the model would cheat by peeking at the result. The test
`test_selected_features_excludes_postgame_leakage` guards against them sneaking
back in.

---

## 4. The training process, step by step

All of this runs when you execute:

```
python scripts/modeling/decision_tree_training.py
```

### Step 1 — Build & clean the dataset
Join games to team stats for all six seasons (above), then **drop any row
missing one of the 13 features or the outcome** (`dropna`). The prediction
target is **`Team1Win`** (1 = Team1 won).

### Step 2 — Split into train and test
Hold out **20%** of games as a **test set** the model never sees during
training, keeping the other **80%** for training. The split is *stratified*
(keeps the same win/loss ratio in both halves) and seeded (`random_state=42`) so
it's reproducible.

### Step 3 — Scale the features
Fit a **`StandardScaler`** on the training data — it rescales every feature to a
comparable range (mean 0, spread 1). Some models (especially the SVM) need this
to work well. The scaler is **fit on the training data only**, then applied to
the test data, so no test information leaks into training.

### Step 4 — Try 6 models, each with hyperparameter tuning
Six different algorithms compete. For each one, we don't just train it once — we
try several **hyperparameter** settings (a model's configuration knobs, set
before training) and keep the best. This is done with **`GridSearchCV`**:
exhaustively try every combination in a small grid, scoring each by **5-fold
cross-validation** (split the training data into 5 parts, train on 4, test on
the 5th, rotate, average — a robust accuracy estimate).

| Model | What it is | Tuned settings (grid) |
|---|---|---|
| Logistic Regression | A weighted linear formula → probability. | `C` (regularization strength): 0.01, 0.1, 1, 10 |
| Decision Tree | A flowchart of yes/no splits. | `max_depth`: 3, 5, 8 · `min_samples_leaf`: 1, 20, 50 |
| Random Forest | Many decision trees averaged. | `max_depth`: 5, 10, none · `min_samples_leaf`: 1, 10 |
| XGBoost | Trees built in sequence, each fixing the last's mistakes. | `n_estimators`: 100, 300 · `learning_rate`: 0.03, 0.1 · `max_depth`: 2, 3, 6 |
| SVC (RBF) | Support-vector machine — finds a smooth separating boundary. | `C`: 0.1, 1, 10 · `gamma`: scale, 0.01, 0.1 |
| Bagging SVC | 10 SVMs trained on random data samples, averaged. | inner `C`: 0.1, 1, 10 · inner `gamma`: scale, 0.1 |

Each model's best score and winning settings are logged to **MLflow** (view them
with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

### Step 5 — Pick the winner and refit
The model with the highest cross-validation accuracy wins. It's refit on the
full training set. (For the SVM-based models, real probability output is only
switched on here for the final winner — it's skipped during the grid search
because it's slow and doesn't affect which settings win.)

### Step 6 — Final evaluation
The winner predicts on the **held-out 20% test set** it never saw. That
`test_accuracy` is the honest estimate of real-world performance.

### Step 7 — Save & register
- Save `NBAdata/best_model.pkl` and `NBAdata/scaler.pkl` (the serving fallback).
- Package scaler + model as one **Pipeline** (so scaling can't drift from the
  model) and register it in the **MLflow model registry** as `nba-win-predictor`,
  with the alias **`@production`** — which is exactly what the live API loads.

---

## 5. How to read the accuracy

Accuracy only means something compared to a baseline. The script logs two:

- **Majority-class baseline (~0.53):** always guess the more common outcome.
- **Home-team-always-wins baseline (~0.55):** always pick the home team.

The trained models land around **0.60–0.61**. So the model beats "just pick the
home team" by roughly **5 percentage points**. That gap is real but modest —
single NBA games are inherently close to a coin flip, and month-level team
averages only carry so much signal. All six models cluster near 0.60, which
tells us we're bumping against an **information ceiling in the features**, not a
weakness of any one algorithm.

> The current production winner is **Random Forest** (~0.611 cross-val, ~0.600
> test), but the winner changes from run to run — across recent retrains it's been
> Bagging SVC, XGBoost, and Random Forest, all separated by less than the
> run-to-run noise.

---

## 6. Where everything lives

| Thing | Path |
|---|---|
| Training script | `scripts/modeling/decision_tree_training.py` |
| Feature contract (shared with serving) | `scripts/modeling/features.py` |
| MLflow / registry settings | `scripts/modeling/mlflow_config.py` |
| Raw data (DVC-tracked) | `NBAdata/matchups/`, `NBAdata/monthly_stats/` |
| Combined training set (generated) | `NBAdata/NBA_Training_Matchups_2019_2025.csv` |
| Saved model + scaler (serving fallback) | `NBAdata/best_model.pkl`, `NBAdata/scaler.pkl` |
| Experiment tracking UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000 |
