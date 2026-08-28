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

## 3. The features (25 inputs)

12 per-team stats × 2 teams + 1 home-court flag. Each stat is a trailing
10-game average unless noted.

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
| `B2B` | 1 if back-to-back (0 rest) | `B2B`* |
| `PlayersOut` | Count of that team's players listed `Out`/`Doubtful` on the NBA's injury report | `PlayersOut`* |

`eFG%`/`TOV%`/`ORB%`/`FTR` are the classic "Four Factors." `B2B` and
`PlayersOut` (marked `*`) aren't rolling averages like the rest — both are
facts about *this specific upcoming game*, computed dynamically at
prediction time rather than snapshotted (`predictor.py`'s
`is_back_to_back()` / `injury_report.py`'s `fetch_latest_injury_counts()`).
A SHAP audit (2026-08-28) found the continuous rest-day count carried
almost no weight vs. `B2B`'s much larger effect, so the count was dropped
and only the binary flag kept — see §6.

`PlayersOut` (2026-08-28) only has training coverage for **5 of 7 seasons**
(2021-22 onward — the NBA's public injury-report archive doesn't go back
further). Rows for 2019-20/2020-21 get `NaN` for this feature and are
dropped by the existing `dropna`, same mechanism as any other missing
feature — so the effective training set shrinks to ~6,600 rows for this
one feature's sake. Live serving fetches and parses the NBA's official
injury-report PDF at prediction time (pure Python, no Java — see
`serving/inference/injury_report.py`); if the fetch fails or no report is
found, it degrades gracefully to 0 rather than breaking a prediction.

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
| + SHAP audit: drop `RestDays`, keep `B2B` | 0.629 | 0.630 |
| + `PlayersOut` (injury availability, 5-season subset) | 0.636 | 0.636 |

The rolling-window switch was the one real, above-noise-band gain (+3pt
test). Everything since has landed inside the run-to-run noise band
(±1–1.5pt) — expected for correctness fixes and eval-methodology changes,
which aren't trying to add new signal. All changes compared against the
same-period home-court baseline (~0.55). **Caveat on the last row:**
`PlayersOut` also shrinks the training set to the 5 seasons with injury
coverage (~6,600 rows vs. ~8,300) — a controlled offline A/B on that same
5-season subset (identical row count with/without the feature) showed a
cleaner +1.1pt test gain, so some of this row's apparent movement is the
smaller/different dataset, not purely the feature. Directionally positive
either way, just not as clean a comparison as the other rows.

Current winner: **SVC (RBF Kernel)**, though the winner shifts across
retrains (all 6 models cluster tightly — see §6).

---

## 6. How to improve accuracy further

All 6 models cluster within ~1pt of each other — the limit is feature
signal, not algorithm choice. Ordered by expected payoff:

**Lever 1 — richer features.**
- ✅ Recent form (rolling windows)
- ✅ Rest / back-to-backs
- ✅ Player availability / injuries (`PlayersOut`, 5-season coverage only)
- Opponent-adjusted stats & strength of schedule
- Real home/away splits (vs. the single `Team1Home` flag)
- Head-to-head history, pace/style matchups

**Lever 2 — better use of existing features.**
- Feed `Team1_stat − Team2_stat` differences instead of raw pairs
- League-relative stats (normalize by season/era)
- ✅ SHAP audit (2026-08-28, `scripts/modeling/feature_importance.py`,
  needs `pip install shap` separately — not a pinned dependency, conflicts
  with `requirements.txt`'s `numpy` pin): dropped the continuous `RestDays`
  (negligible importance, `B2B` alone carries the rest-fatigue signal).
  `NetRtg` flagged as a possible prune too (redundant with `OffRtg`/`DefRtg`
  by definition) but kept — worth another look if features keep growing.

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
| Historical injury-report pull | `scripts/data_pull/injury_report_pull.py` (needs `nbainjuries`, not pinned) |
| Injury-count feature builder | `scripts/data_prep/build_injury_features.py` |
| Live injury-report fetch (serving) | `serving/inference/injury_report.py` |
| Raw data (DVC) | `NBAdata/matchups/`, `NBAdata/team_game_logs/`, `NBAdata/rolling_stats/nba_team_rolling_stats_*.csv`, `NBAdata/injury_reports/` |
| Current-form snapshot (git) | `NBAdata/rolling_stats/nba_team_current_rolling_stats_*.csv` |
| Combined training set (generated) | `NBAdata/NBA_Training_Matchups_2019_2025.csv` |
| Saved model + scaler | `NBAdata/best_model.pkl`, `NBAdata/scaler.pkl` |
| Experiment tracking UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → localhost:5000 |

`NBAdata/monthly_stats/` (old month-granularity data) is unused but left on
disk rather than deleted.
