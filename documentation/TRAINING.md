# How the model is trained (and what it learns from)

Plain-language explanation of how the NBA win predictor is built: the data, the
features, and the training steps. Reflects `scripts/modeling/decision_tree_training.py`
and `scripts/modeling/features.py`.

---

## 1. What the model predicts

**Given two teams in a specific game, will the home team win?** A yes/no
prediction plus a win probability (e.g. "home team wins, 61% confidence").

---

## 2. The data

- **Source:** per-game NBA team box scores pulled from the NBA API
  (`NBAdata/team_game_logs/`). Bulk history is DVC-tracked; a small per-team
  "current form" snapshot is git-tracked so serving works without `dvc pull`.
- **Seasons:** 7, 2019-20 through 2025-26 (~8,900 games).
- **One row = one game**, paired with each team's stats **as of that game**.

Each team's stats are a **trailing 10-game rolling average**, computed only
from games strictly before the one being predicted (`build_rolling_team_stats.py`,
`.shift(1)` before `.rolling(10)` — a game's own stats can never leak into its
own average). Needs 3+ prior games or the row is dropped. This also makes the
join exact: matchups and rolling stats share the same `GAME_ID`, so training
joins on that instead of a fuzzy `(Team, Season, Month)` match.

### Team1 / Team2

Teams are labeled **Team1**/**Team2** alphabetically in training data — home
court is its own feature, `Team1Home`. In live serving, Team1 is always the
home team, so the model's output reads directly as "probability home team wins."

---

## 3. The features (21 inputs)

10 per-team stats × 2 teams + 1 home-court flag. Each stat is a trailing
10-game average.

| Feature | Meaning | Source |
|---|---|---|
| `W_PCT` | Win rate, last 10 games | `W_PCT` |
| `PIE` | Player Impact Estimate — overall "how good" | `PIE` |
| `eFG%` | Effective field-goal % | `EFG_PCT` |
| `TOV%` | Turnover rate | `TM_TOV_PCT` |
| `ORB%` | Offensive rebound % | `OREB_PCT` |
| `FTR` | Free-throw rate, `FTA/FGA` | `FTR` |
| `NetRtg` | Net rating (pts scored − allowed per 100 poss.) | `NET_RATING` |
| `OffRtg` | Offensive rating | `OFF_RATING` |
| `DefRtg` | Defensive rating | `DEF_RATING` |
| `Pace` | Possessions per 48 min | `PACE` |
| `RestDays` | Days of rest before this game (capped at 5) | `RestDays`* |
| `B2B` | 1 if back-to-back (0 rest) | `B2B`* |

`eFG%`/`TOV%`/`ORB%`/`FTR` are the classic "Four Factors." `RestDays`/`B2B`
are the one pair that isn't a rolling average — they're computed dynamically
at prediction time from the target game's date vs. the team's last known
game (`predictor.py`'s `rest_days_and_b2b()`), not snapshotted like the rest.

Full ordered list lives in `features.py`'s `SELECTED_FEATURES`. Post-game
info (final score, plus/minus) is deliberately excluded — that would be
leakage, guarded by `test_selected_features_excludes_postgame_leakage`.

---

## 4. Training, step by step

Run: `python scripts/modeling/decision_tree_training.py`

1. **Build & clean.** Join matchups to rolling stats on `GAME_ID`; drop rows
   missing any feature (mostly each team's first couple of season games).
2. **Split.** Chronological, per season: each season's earliest ~80% of
   games train, latest ~20% (incl. playoffs) test. Keeps every era in both
   splits and matches real usage (predict games not yet seen).
3. **Scale.** `StandardScaler`, fit on train only.
4. **Train + tune 6 models** via `GridSearchCV` (5-fold CV on the training
   split): Logistic Regression, Decision Tree, Random Forest, XGBoost,
   SVC-RBF, Bagging SVC. Logged to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
5. **Pick the winner** by CV accuracy, refit on full training set.
6. **Evaluate** on the held-out test set — the honest accuracy number.
7. **Save & register**: `best_model.pkl`/`scaler.pkl`, plus an MLflow
   `Pipeline` registered as `nba-win-predictor` with the `@production` alias.

---

## 5. Accuracy

Two baselines: majority-class (~0.52) and home-team-always-wins (~0.55).

| Change | CV | Test |
|---|---|---|
| Month-to-date averages (pre-2026-08-28) | 0.61 | 0.60 |
| + rolling 10-game windows | 0.625 | 0.631 |
| + FTR fix (true `FTA/FGA`) | 0.628 | 0.626 |
| + chronological split | 0.627 | 0.624 |
| + rest days / back-to-back | 0.628 | 0.631 |

The rolling-window switch was the one real, above-noise-band gain (+3pt
test). Everything since has landed inside the run-to-run noise band
(±1–1.5pt) — expected for correctness fixes and eval-methodology changes,
which aren't trying to add new signal. All changes compared against the
same-period home-court baseline (~0.55).

Current winner: **Logistic Regression**, though the winner shifts across
retrains (all 6 models cluster tightly — see §6).

---

## 6. How to improve accuracy further

All 6 models cluster within ~1pt of each other — the limit is feature
signal, not algorithm choice. Ordered by expected payoff:

**Lever 1 — richer features.**
- ✅ Recent form (rolling windows)
- ✅ Rest / back-to-backs
- Player availability / injuries
- Opponent-adjusted stats & strength of schedule
- Real home/away splits (vs. the single `Team1Home` flag)
- Head-to-head history, pace/style matchups

**Lever 2 — better use of existing features.**
- Feed `Team1_stat − Team2_stat` differences instead of raw pairs
- League-relative stats (normalize by season/era)

**Lever 3 — evaluation honesty.**
- ✅ Chronological split
- ✅ `FTR` corrected to true free-throw rate
- Early-season noise: currently a hard 3-game cutoff, not down-weighting

**Lever 4 — model-side (smaller payoff, models already tie).**
- Probability calibration (`CalibratedClassifierCV`)
- Stacking, or a wider hyperparameter search

**Ceiling:** pro models using betting-market data top out ~65–70%; single
games are close to coin flips. 0.63 on box-score-derived stats is solid.
Betting odds are deliberately excluded as a feature — that's tracking the
market, a different problem than predicting from team performance.

---

## 7. Where everything lives

| Thing | Path |
|---|---|
| Training script | `scripts/modeling/decision_tree_training.py` |
| Feature contract | `scripts/modeling/features.py` |
| MLflow / registry settings | `scripts/modeling/mlflow_config.py` |
| Per-game box score pull | `scripts/data_pull/team_game_logs_pull.py` |
| Rolling-window builder | `scripts/data_prep/build_rolling_team_stats.py` |
| Current-form snapshot builder | `scripts/data_prep/build_current_rolling_snapshot.py` |
| Raw data (DVC) | `NBAdata/matchups/`, `NBAdata/team_game_logs/`, `NBAdata/rolling_stats/nba_team_rolling_stats_*.csv` |
| Current-form snapshot (git) | `NBAdata/rolling_stats/nba_team_current_rolling_stats_*.csv` |
| Combined training set (generated) | `NBAdata/NBA_Training_Matchups_2019_2025.csv` |
| Saved model + scaler | `NBAdata/best_model.pkl`, `NBAdata/scaler.pkl` |
| Experiment tracking UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → localhost:5000 |

`NBAdata/monthly_stats/` (old month-granularity data) is unused but left on
disk rather than deleted.
