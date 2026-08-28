# Project log

Snapshot of where the project stands, updated as major changes land. Not a full changelog — see git history for that.

## 2026-08-28 — Injury availability (`PlayersOut`) feature

Added a count of each team's `Out`/`Doubtful` players from the NBA's official
injury report — 23 → 25 features. Historical pull (`nbainjuries`, needs Java,
training-only) covers 5 of 7 seasons (2021-22+, the archive doesn't go back
further); 2019-20/2020-21 rows drop via the existing `dropna`. An offline
controlled A/B (identical row counts) showed +1.1pt test on the 5-season
subset before committing to building live serving support. Live serving uses
a custom pure-Python `pdfplumber` parser instead (`serving/inference/injury_report.py`)
to keep Java out of the Docker images — fetches and parses the current NBA
injury-report PDF at prediction time, degrading gracefully to 0 on any
failure. `@production` v10 (SVC-RBF): CV 0.636 / test 0.636. Built via two
parallel background agents (training-side vs. serving-side, disjoint files) —
80/80 tests, real Docker rebuild + live smoke test, both verified after landing.

## 2026-08-28 — SHAP audit: dropped `RestDays`, kept `B2B`

Ran a SHAP feature-importance audit (`scripts/modeling/feature_importance.py`,
needs `pip install shap` separately) on `@production`. Found `RestDays`
(continuous) had negligible importance vs. `B2B`'s much larger effect —
confirms rest matters as a threshold (0 rest, yes/no), not a smooth
gradient. Dropped `RestDays` as a feature, kept `B2B`: 25 → 23 features.
`NetRtg` also flagged as redundant with `OffRtg`/`DefRtg` by the same audit
but kept (user's call). `@production` v9: CV 0.629 / test 0.630, unchanged
within noise.

## 2026-08-28 — Rest days / back-to-back feature

`RestDays` (capped at 5) + `B2B` added — 21 → 25 features. Computed from
calendar gaps in `team_game_logs`, not snapshotted like the other stats:
serving computes it dynamically from the target game's date vs. each team's
last known game (`predictor.py`'s `rest_days_and_b2b()`). `@production` v8
(Logistic Regression): CV 0.628 / test 0.631 — inside noise vs. v7, slight
positive lean. Full suite + Docker smoke test green.

## 2026-08-28 — Chronological train/test split

Replaced random 80/20 split with a per-season chronological one (each
season's earliest ~80% trains, latest ~20% incl. playoffs tests) — matches
real usage and confirms an earlier diagnostic that the random split wasn't
flattering the score. `@production` v7: CV 0.627 / test 0.624, within noise
of the random split.

## 2026-08-28 — FTR fix

`FTR` now `FTA/FGA` (true free-throw rate) instead of the `FT_PCT` proxy.
Correctness fix, not an accuracy bet — `@production` v6: CV 0.628 / test
0.626, a wash vs. v5.

## 2026-08-28 — Rolling-window rebuild (Part B)

Replaced month-to-date team stats with trailing 10-game rolling averages,
leakage-free by construction (`.shift(1)` before `.rolling(10)`). New
pipeline: `team_game_logs_pull.py` (bulk per-game box scores) →
`build_rolling_team_stats.py` (rolling averages) →
`build_current_rolling_snapshot.py` (serving snapshot). Training now joins
on exact `GAME_ID` instead of fuzzy `(Team, Season, Month)`. Fixes the
residual month-level leakage open since 2026-07-15. `@production` v5
(Logistic Regression): CV 0.625 / test 0.631, up from 0.615/0.598 — the one
real, above-noise gain this round of changes. `docker/Dockerfile.api` fixed
to bake in the new snapshot (was still copying the dead `monthly_stats/`
path). Old month-granularity scripts/data left in place, unreferenced.

## 2026-07-24 — Web UI, probability fix, tuning, 2025-26 season

Added a 2-team-dropdown web UI at `/`. Fixed a probability bug (bagged SVC
missing `probability=True` → vote-count instead of real probabilities).
Replaced hand-picked hyperparameters with `GridSearchCV` across all 6
models. Pulled the 2025-26 season, retrained on all 7 (`@production` v3,
Random Forest, CV 0.611/test 0.600). Moved current-season stats from
DVC into git so serving needs no AWS credentials. `docker/Dockerfile.api`
made self-contained (bakes in model + stats).

## 2026-07-22 — Real-time FastAPI service

Added `serving/api/` (`POST /predict`, `GET /health`), loading the same
`@production` model the batch job uses. Extracted shared model-loading/
feature code into `serving/inference/predictor.py`. Error mapping: 404
(unknown team/no stats), 422 (missing feature), 503 (not provisioned/model
down). Packaged as `docker/Dockerfile.api`.

## 2026-07-21 — MLflow + DVC

Training now logs to MLflow and registers `@production` via alias (MLflow
3.x has no stages). Bulk training data moved from git into DVC/S3. Batch
job loads from the registry, falling back to local pkls.

## 2026-07-16 — CI added; monthly-stats mislabeling fixed

`merge_advanced_base_stats.py` had hardcoded `season="2019-20"`, so the
`(Team,Season,Month)` join only ever matched 2019-20 rows — the model was
silently training on ~976 rows instead of ~7,600. Fixed; retrained on all 6
seasons (Bagging SVC, CV 0.607/test 0.611 vs. 0.554 baseline). Added
`ARCHITECTURE.md` (productionizing roadmap) and CI (`ci.yml`: pytest +
Docker build, `stats.nba.com`-dependent code excluded since it blocks CI
runner IPs).

## 2026-07-15 — Leakage fix, home-court feature, repo cleanup

Fixed the original data leakage: the model was trained on the completed
game's own `PTS`/`PLUS_MINUS` to predict that game's winner. Removed those;
added `Team1Home` (previously missing). Switched to 5-fold CV + baseline
comparisons. Best model at the time: Decision Tree, ~61% CV / 58% test vs.
~55% home-court baseline. Removed a second, also-leaky, unused pipeline and
various orphaned scripts.
