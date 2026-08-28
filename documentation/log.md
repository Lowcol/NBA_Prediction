# Project log

Snapshot of where the project stands, updated as major changes land. Not a full changelog — see git history for that.

## 2026-08-28 — FTR fixed to true free-throw rate (isolated retrain after the rolling-window rebuild)

**State: `FTR` is now `FTA/FGA` (true free-throw rate), not the `FT_PCT` proxy; `@production` (v6, SVC-RBF) — CV 62.8% / test 62.6%, a wash vs. v5's 62.5%/63.1%, both inside the noise band.**

- `FT_PCT` (free-throw shooting *percentage*) was never the right stat for the `FTR` ("free-throw rate") feature — it measured how well a team shoots free throws, not how often it gets to the line. Flagged as a known approximation since the rolling-window rebuild (same day), fixed as a deliberately separate change so its effect could be measured on its own.
- `build_rolling_team_stats.py` now derives `FTR = FTA/FGA` per game from the already-pulled raw box score columns, then rolls it the same way as the other 9 stats. `features.py`'s `STAT_MAP` and `build_current_rolling_snapshot.py` updated to match (`FT_PCT` → `FTR` throughout).
- Regenerated all 7 seasons' rolling-stats files and both current-season snapshots; retrained (`@production` v6). Result: a wash, same pattern as the `NetRtg`/`Pace` addition — a real correctness fix, not an accuracy one.
- Full test suite (54/54) and a real Docker rebuild + live `/health`/`/predict` smoke test both verified after the change.

## 2026-08-28 — Rolling-window feature rebuild (Part B): month averages replaced with trailing 10-game stats

**State: training and serving both key off per-game rolling stats now, not month-to-date averages; `@production` (v5, Logistic Regression) genuinely improved — CV 62.5% / test 63.1%, up from 61.5% / 59.8%.**

- Added `NetRtg`/`OffRtg`/`DefRtg`/`Pace` as month-level features first (13 → 21 features) to isolate whether the ceiling was stat variety — it wasn't (a wash, CV 61.5%/test 59.8% vs the prior 61.1%/60.0%). Also ran a one-off chronological-split diagnostic against the then-current model to check the random split wasn't flattering the score — it wasn't (61.4% chronological vs. 60.0% random).
- Both diagnostics pointed at time resolution, not stat variety or eval methodology, as the real remaining lever — so rebuilt the feature pipeline around it:
  - New `scripts/data_pull/team_game_logs_pull.py` pulls per-game team box scores (Base + Advanced) via `TeamGameLogs`, one bulk call per season/season-type/measure-type (~56 calls total for all 7 seasons — not the ~9,000 individually-called games originally estimated).
  - New `scripts/data_prep/build_rolling_team_stats.py` computes trailing 10-game rolling averages per team per game, `.shift(1)` before `.rolling(10)` so a game's own stats can never leak into its own average (`NBAdata/rolling_stats/`). Regression-tested directly: `test_build_rolling_team_stats.py`'s leakage test asserts this.
  - New `scripts/data_prep/build_current_rolling_snapshot.py` condenses that into a one-row-per-team "current form" snapshot for serving, with a documented season-boundary carryover exception (falls back to the previous season's tail if a team has no valid games yet this season) — training does not do this, it just drops those early rows.
  - `decision_tree_training.py`'s join moved from `["Team", "Season", "Month"]` (approximate) to `["Team", "GAME_ID"]` (exact — both matchups and the new per-game logs share the same NBA game IDs, confirmed 0 mismatches across all 7 seasons).
  - `serving/inference/predictor.py`'s `latest_team_stat_row`/`assemble_features` simplified (dropped the `month` parameter and the old month-wraparound-sort fallback entirely) since the snapshot is already "one row per team, already current." `serving/batch/run_nightly_predictions.py` and `serving/api/main.py` updated to match.
  - `docker/Dockerfile.api` updated to bake in the new snapshot files instead of the old `monthly_stats/`; rebuilt and smoke-tested end-to-end (container health check + a real `/predict` call against the baked-in data both worked).
- `merge_advanced_base_stats.py`, `monthly_stats_pull.py`/`monthly_team_stat_pull.py`, `nba_api_pull.py`, and the old `NBAdata/monthly_stats/` data are superseded and unreferenced by the pipeline as of this change — left in place rather than deleted (same "leave it, don't delete DVC-tracked/working history" pattern used elsewhere in this project).
- Fixes the "residual, smaller leakage" issue that had been open since 2026-07-15 (see `COMPONENTS.md`, "Known issues") — a trailing window ending strictly before the target game can't leak by construction, unlike a month aggregate that included the game's own contribution.
- New known limitation, documented in `COMPONENTS.md`: the serving snapshot always reflects the *most recent* data on file, not "as of the requested date" — so `run_nightly_predictions.py --date <past date>` no longer doubles as point-in-time historical backtesting the way it arguably did under the old month-keyed lookup.
- Implementation split across parallel background agents (training-join rework and serving rework touch disjoint files, ran concurrently) plus direct work for the live API pulls, DVC wiring, Docker fix, and this doc pass. Full test suite: 54/54 passing throughout.
- Not yet done, deliberately deferred as a separate follow-up so any accuracy change is attributable on its own: fixing `FTR` to be true `FTA/FGA` instead of the current `FT_PCT` proxy (cheap now that raw `FTA`/`FGA` are pulled per-game anyway). Also not done: pushing the new DVC-tracked data (`team_game_logs.dvc`, the per-game `rolling_stats` `.dvc` files) to the S3 remote, or committing any of this — both are outward/durable actions gated on explicit user say-so, same pattern as every prior phase in this project.

## 2026-07-24 — Web UI, probability fix, hyperparameter tuning, 2025-26 season, self-contained serving

**State: a two-team dropdown UI is served from the API; probabilities are real (not vote counts); all 6 models are tuned via `GridSearchCV`; training now covers 7 seasons (2019-20..2025-26) with Random Forest as the current `@production` model (v3, CV 0.611 / test 0.600); the API and batch job run straight after cloning with no AWS credentials.**

- Added a minimal web UI (`serving/api/static/index.html`, served at `GET /`): two team dropdowns, POSTs to `/predict`, no date field (defaults to today). `/predict` now falls back to the latest season on file when the target date's season has no stats, instead of 503-ing whenever "today" falls in an unplayed season.
- Fixed a probability bug: the winning model's bagged base `SVC` was missing `probability=True`, so `predict_proba` degraded to counting the 10 SVMs' hard votes (piling at 0%/100%). Fixed, and separately replaced the old hand-picked hyperparameters with `GridSearchCV` (5-fold, small per-model grids) for all 6 models — each grid includes the former hand-set config, so tuning can only match or beat it. Also fixed a latent bug where MLflow's skops serialization refused to save a winning XGBoost model until its types were added to `skops_trusted_types`.
- Pulled the 2025-26 season (via the existing TLS-bypass pull path) and retrained on all 7 seasons (~8,900 rows). New `@production` v3: **Random Forest** (CV 0.611 / test 0.600), pushed to the DVC S3 remote. `documentation/TRAINING.md` added — plain-language walkthrough of training and the 13 features.
- Moved `NBAdata/monthly_stats/` (current-season stats) out of DVC and into git, alongside the already-git-tracked `best_model.pkl`/`scaler.pkl` — so a fresh clone can serve predictions with no AWS credentials and no `dvc pull`. Bulk training data (`matchups/`, `archive/`, the combined training CSV) stays in DVC/S3.
- `docker/Dockerfile.api` now bakes in the model and current-season stats (both git-tracked), making the API image fully self-contained — `docker build` + `docker run`, no volume mount needed. `docker/Dockerfile.batch` is unchanged (still mounts `NBAdata/` at runtime). README and COMPONENTS.md rewritten to lead with the Docker quick-start.

## 2026-07-22 — Phase 4: real-time FastAPI prediction service + shared inference module

**State: a `/predict` + `/health` API serves live single-game predictions from the same `@production` model the batch job uses.**

- Built `serving/api/` (FastAPI): `POST /predict` takes `{home_team, away_team, date}`, looks up each team's latest monthly stats server-side (same feature contract as training), and returns the predicted winner + home-win probability. `GET /health` reports whether the model loaded and whether the current season's stats are on file. The model is loaded once at startup via FastAPI's `lifespan` (a registry/pkl load is expensive; per-request would re-hit MLflow every call).
- Extracted the model-loading and feature-assembly code out of `run_nightly_predictions.py` into `serving/inference/predictor.py`, now imported by both the batch job and the API — one source of truth so the two serving paths can't drift (same rationale as the shared `features.py`/`mlflow_config.py`). Added two thin shared helpers, `assemble_features` and `predict_from_features`. The batch job re-exports the moved names, so its imports/tests keep working; the two `load_predictor` tests had their mock targets retargeted to `predictor.*`.
- `/predict` error mapping: unknown team / no usable stats → 404; stats present but a required feature missing → 422; season stats file absent or model not loaded → 503; malformed body → 422 (Pydantic).
- Packaged as `docker/Dockerfile.api` (uvicorn, port 8000; `NBAdata/` mounted at runtime, same as the batch image). `requirements.txt` adds `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`; CI builds the API image alongside the batch image; `pytest.ini` gains `serving/inference` + `serving/api`. New offline `tests/test_api.py` (FastAPI `TestClient`, model + stats mocked, no network) covers the happy path and every error branch.
- **Reordered vs. `ARCHITECTURE.md` §9:** built the real-time API (phase 4) before batch Prometheus (phase 3). A short-lived batch job can't be scraped by Prometheus directly (needs a push gateway); a long-running API is naturally scrapeable, so it's the better monitoring target — phase 3's `/metrics` will target this app. Circuit breaker (5), shadow deployment (6), and load testing (7) remain deferred.

## 2026-07-21 — Phase 2: MLflow tracking/registry + DVC data on S3, batch repointed at the registry

**State: training logs to MLflow and registers `@production`; data moved to DVC/S3; batch job loads from the registry with a local-pkl fallback.**

- `decision_tree_training.py` now logs to MLflow (local sqlite `mlflow.db`, gitignored): a parent run plus a nested run per candidate model (params + CV mean/std). After selecting the best model it packages the already-fit `StandardScaler` + model as a single raw-features→prediction `sklearn.Pipeline`, logs it with an inferred signature, registers it as `nba-win-predictor`, and moves the `@production` alias to the new version. Registry name/alias/tracking URI live in the shared `scripts/modeling/mlflow_config.py` so training and serving can't drift. Still writes `best_model.pkl`/`scaler.pkl` as before. MLflow 3.x uses **aliases**, not stages (stages were removed) — `@production` is the promotion mechanism.
- `run_nightly_predictions.py` `load_predictor()` loads `models:/nba-win-predictor@production` from the registry, falling back to the local pkls (wrapped in the same Pipeline) when the registry is unreachable — which is what the container does, since `mlflow.db` isn't mounted. If both the registry and the pkls are unavailable it now raises a clear error pointing at the `NBAdata/` mount, instead of an uncaught `FileNotFoundError`.
- Training data (`matchups/`, `monthly_stats/`, `archive/`, combined training CSVs) moved out of git into DVC, stored in S3 (`s3://nba-prediction-dvc-ag`, `us-east-2`); `.dvc` pointers are committed, the data is gitignored. `best_model.pkl`/`scaler.pkl` intentionally stay in git as the batch fallback. CI gained a `dvc pull` step (AWS secrets) before pytest; `requirements.txt` adds `mlflow==3.14.0` and `dvc[s3]==3.67.1`.
- Fixed a season-boundary bug in `latest_team_stat_row`'s fallback: it sorted months numerically (1..12) and returned December instead of the chronologically-latest month across the Oct–Jun season; now sorted on a season-relative ordinal.
- Test-infra note (supersedes the 2026-07-16 entry below): `requirements-dev.txt` and the root `conftest.py` were dropped — test deps are merged into `requirements.txt` and imports are resolved via `pytest.ini`'s `pythonpath`.

## 2026-07-16 — Added CI: pytest suite + Docker build check

**State: `.github/workflows/ci.yml` runs on every push/PR to `main`.**

- Added `tests/` (pytest): feature-resolution logic, batch job's pure-logic pieces (season/date parsing, home/away parsing, stats lookup + fallback), and regression guards for the two bugs already fixed this week — leaked post-game columns reappearing in `SELECTED_FEATURES`, and any combined monthly-stats file's `Season` column not matching its filename. Also checks the full training dataset uses all 6 seasons and that `best_model.pkl`/`scaler.pkl` load and predict correctly.
- Deliberately excluded from CI: `scripts/data_pull/*` and the batch job's live schedule fetch, since `stats.nba.com` blocks cloud/datacenter IPs (GitHub-hosted runners included) — those stay manual/local-only.
- `requirements-dev.txt` added (adds `pytest` on top of `requirements.txt`); `conftest.py` at repo root wires up imports since the project has no package structure.

## 2026-07-16 — Fixed monthly-stats season mislabeling, retrained, started MLOps roadmap

**State: training pipeline now actually uses all 6 seasons; started productionizing per `ARCHITECTURE.md`.**

- While scoping the first build phase (batch prediction job, per `ARCHITECTURE.md`), found that `merge_advanced_base_stats.py` hardcoded `season = "2019-20"` regardless of which season it was merging — every combined monthly-stats file, including the other 5 seasons' archived files, had its `Season` column stuck at `"2019-20"`. Since training joins matchups to stats on `["Team", "Season", "Month"]`, only 2019-20 matchups ever matched; the other 5 seasons were silently dropped by `dropna` on every training run since this bug was introduced. The model was effectively trained on ~976 rows, not ~7,600.
- Fixed: `season` is now read from the loaded data instead of hardcoded, and the script regenerates every season's file. No re-pull from the NBA API was needed — the raw per-season base/advanced stat files were already correct, only the merge step was buggy.
- Retrained on the corrected dataset (7,595 rows across all 6 seasons). New best model: **Bagging SVC**, ~60.7% CV accuracy, ~61.1% held-out test accuracy, vs. a ~55.4% home-court-only baseline. Comparable to the previous (bugged) result, but now genuinely trained on the full multi-season dataset instead of one season.
- Added `ARCHITECTURE.md`: a plan for productionizing the model (Docker, batch + real-time serving, MLflow/DVC versioning, monitoring, circuit breaker, shadow deployment, load testing). Batch prediction is the priority; the on-demand real-time API is deferred to a later phase.

## 2026-07-15 — Leakage fix, home-court feature, repo cleanup

**State: working end to end, results are believable but not yet strong.**

- Data covers 6 seasons of NBA games: 2019-20 through 2024-25 (regular season, playoffs, play-in), ~7,600 matchups total after joining to monthly team stats.
- Fixed a data leakage bug where the model was trained on the completed game's own `PTS`/`PLUS_MINUS` to predict that same game's winner. Removed those as features.
- Added `Team1Home` (home/away) as a feature — previously missing entirely.
- Model selection now uses 5-fold cross-validation instead of a single train/test split, and is compared against two baselines (majority-class, home-team-always-wins) so results are interpretable.
- Current best model: **Decision Tree**, ~61% CV accuracy, ~58% held-out test accuracy, vs. a ~55% home-court-only baseline. Beats baseline, but not by a wide margin, and the gap between CV and test accuracy suggests the Decision Tree may be somewhat high-variance on this data — worth re-checking after any feature changes.
- Removed a large amount of dead weight from the repo: an entire second, also-leaky data pipeline that was no longer used, a synthetic-matchup generator with a circular label, and various orphaned scripts/CSVs left over from earlier iterations. See `README.md` for what remains and why.

### Known limitations (not yet addressed)
- Team stats are joined to each matchup at month granularity, which means a game's own result mildly contaminates the month-aggregate stats it's matched against (see README "Known issues"). Fixing this needs trailing rolling stats computed strictly before each game's date, not a monthly merge.
- No player-level data (injuries, rest days, star player availability) is used at all — only team-level monthly aggregates. This is likely the biggest lever for accuracy beyond fixing the residual leakage.
- `stats.nba.com` blocks some cloud/sandboxed networks; data pulls need to run from a normal network.

### Possible next steps
- Rebuild team stats as pregame rolling averages to close the residual leakage gap.
- Add player-availability signals (injuries, rest).
- Investigate why the Decision Tree's CV score doesn't hold up as well on the held-out test set.
