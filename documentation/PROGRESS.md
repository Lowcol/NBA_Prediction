# Progress notes (working scratchpad)

Purpose: a running record of *how* work got done — task specs, design
decisions made along the way, things ruled out, open questions — not just
*what* changed. `log.md` is the terse changelog; this is the reasoning
behind it. Read this first when resuming work after a context gap.

---

## Web UI + probability bug + hyperparameter tuning (out-of-band, 2026-07-24)

**Status: code + tests done; NOT retrained yet — the tuning + probability fixes only take effect when the user re-runs training (which also re-promotes `@production`). Nothing committed.**

### Web UI (user request: "simple interface, choose 2 teams, get prediction")
- `serving/api/static/index.html` — single self-contained page (inline CSS/JS, no deps), two dropdowns with the 30 NBA teams hardcoded (user's choice over a `/teams` endpoint), POSTs `/predict` same-origin (no CORS needed). No date field (user's choice) — API defaults to today.
- `serving/api/main.py` — `GET /` serves the page via `FileResponse`. No `StaticFiles` mount needed for one file. Dockerfile untouched (`COPY serving/` already ships it).
- **Season fallback (user's choice among 3 options):** the UI's date-less request maps today (off-season 2026) to season `2026_27`, which has no stats file → every request 503'd. `/predict` now falls back to the latest season on file (`max()` on keys like `"2024_25"`) when the target season is missing; 503 only when *nothing* is provisioned. Caveat flagged to user: response echoes the requested date even when stats come from an older season. Tests updated (mocks now need the season present in `collect_monthly_files`) + 2 new tests (fallback picks latest; empty store → 503).

### Probability bug: "everything is 100%"
- User noticed most matchups predicted at 100%. Root cause: production model is `BaggingClassifier(estimator=SVC(gamma="scale"))` — base SVC **without** `probability=True`, so `predict_proba` degrades to counting the 10 SVMs' hard votes (all values multiples of 0.1, piling at 0/1; verified empirically: 28 of 56 test matchups at exactly 1.0). The standalone SVC on the line above *had* the flag; the bagged one didn't.
- Fix: `probability=True` on the bagged base SVC (user chose "fix code only, I'll retrain"). Doesn't change predictions/accuracy — only makes probabilities real (Platt-scaled, averaged).

### Model training visibility (user asked for "TensorBoard")
- Pushed back: models are sklearn (one-shot fits, no epoch curves) and MLflow tracking already exists — TensorFlow would be a heavy dep for bar charts. User accepted MLflow UI instead: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → :5000.
- Gotcha for next time: the 6 model runs are **nested** under the `training-run` parent — the UI collapses them (user thought only Bagging SVC was logged; all 6 were there behind the ▸ expander; Models tab shows only the registered winner).

### Hyperparameter tuning (user request, after "why did SVC beat XGBoost" → answer: it didn't meaningfully — 0.7pt gap vs ±1-1.7pt fold noise, all models ~0.60 vs 0.554 home-court baseline, none tuned)
- `decision_tree_training.py`: `cross_val_score` → `GridSearchCV` (5-fold, accuracy, `n_jobs=-1`, `refit=False` since the winner is refit once at the end). New module-level `build_model_grids()` (testable) — small grids per model (4-18 candidates, 46 total), each containing the former hand-picked config so tuning can only match-or-beat.
- **SVC probability trick:** grids tune with `probability=False` (accuracy uses `predict`, unaffected; `probability=True` adds internal 5-fold calibration ≈5x cost per fit). `PROBABILITY_OVERRIDES` applies `probability=True` / `estimator__probability=True` to the winner only, before its single final fit. This preserves the probability bugfix above.
- MLflow per-model runs now log `best_<param>` + `n_grid_candidates` instead of full `get_params()`.

### Verification performed
- Smoke test (scratchpad): all 6 grids fit on tiny synthetic data (validates grid param names, incl. `estimator__` prefixes) + both overrides apply cleanly.
- Full suite: 44 passed. UI verified via TestClient (200, HTML served) + real end-to-end predict after `dvc pull` (date-less request now 200 via fallback; unknown team still 404).

### Retrain done (same day) — new winner XGBoost, plus a registration bug fixed
- First retrain crashed at `mlflow.sklearn.log_model`: MLflow 3.x serializes sklearn models via **skops**, which trust-lists sklearn types only. Tuning changed the winner to **XGBoost** (first non-pure-sklearn winner), whose `XGBClassifier`/`Booster` types got refused at the save audit. Latent bug — only fires when a non-sklearn model wins the bake-off.
- Fix: `skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]` on `log_model`. Verified in MLflow source that the list is stored in the flavor config and reused by `load_model` automatically → serving needed no change.
- Rerun succeeded: **XGBoost wins (CV 0.6079, test 0.6073)** with tuned `{lr 0.1, max_depth 3, n_estimators 100}` — shallower than the old hand-picked depth-6. Old winner Bagging SVC unchanged at 0.6070 (its grid re-picked the former hand-set config). Registered **v2**, `@production` re-aliased; pkls overwritten.
- Verified end-to-end: API loads v2 from registry, probabilities now realistic (e.g. 0.615/0.583/0.686/0.208 vs the old vote-count 0/1) — XGBoost has native `predict_proba`, so the SVC probability override wasn't even needed this time. Full suite 44 passed.

### Pulled 2025-26 season + retrained on 7 seasons (same day)
- **API reachability:** plain `nba_api` to stats.nba.com times out from this env (datacenter-IP block, as README documents). The repo's `nbaPull_19-25_matchups.py` has a working **TLS bypass** (`curl_cffi` Chrome-120 impersonation + `nba.com/stats` Akamai cookie warmup, then `NBAStatsHTTP.get_session` override). Reusing that bypass, the API **is** reachable — so the pull ran here, not just locally.
- **Pull:** one-off scratchpad script (`scratchpad/pull_2025_26.py`, not committed) mirroring the existing pull format exactly: `LeagueDashTeamStats` Base + Advanced (PerGame, all 4 season types, months 1-12) → `nba_team_{base,advanced}_stats_2025_26.csv`; `LeagueGameFinder` → `NBA_2025_26_Matchups.csv`. Sanity: **1230 regular-season games = 30×82/2 exactly**, 282 monthly rows each measure.
- **Merge:** called the repo's own `merge_season()` for 2025-26 → `nba_team_combined_stats_2025_26.csv`. Byte-structure identical to 2024-25 (270 rows, 103 cols, same columns); all 6 model stats resolve; 30 teams have month-4 rows. Existing merge `main()` still hardcodes 2024_25 only — didn't touch it; `collect_monthly_files()` globs the dir so the new file is picked up regardless.
- **Retrain (v3):** all 7 seasons load (2025_26 = 1321 rows, ~8916 total). New winner **Random Forest** (CV 0.6111, test 0.5999), tuned `{max_depth 10, min_samples_leaf 10}`. Note winner has now been Bagging SVC (v1-ish) → XGBoost (v2) → Random Forest (v3) across retrains — all within noise, consistent with the ~0.60 information ceiling. Registered v3, `@production` re-aliased.
- **Verified:** API loads v3; realistic probs (0.593 / 0.810 / 0.566). `stats_available: False` in /health is correct — today (Jul 2026) maps to season 2026-27 which hasn't been played; predict falls back to the newest season on file, now 2025-26 (was 2024-25). 44 tests pass. `documentation/TRAINING.md` (new this session, explains process + the 13 features) updated to 7 seasons / RF winner.

### Open / waiting on user
- **New 2025-26 data not yet persisted to the DVC remote.** `dvc status` shows `matchups`, `monthly_stats`, and the combined training CSV modified. Needs `dvc add`/`dvc commit` + `dvc push` (S3), then git-commit the updated `.dvc` pointers + `best_model.pkl`/`scaler.pkl` (v3). Not done — pushing to S3 is an outward action, gated on user say-so.
- Whether to add a committed, season-parameterized pull script (the scratchpad one is throwaway) so next season is one command.
- Whether to flatten MLflow nested runs to top-level (offered, not decided).
- Nothing committed yet — waiting on explicit "commit this."

---

## Phase 4 — real-time FastAPI API (`ARCHITECTURE.md`), built ahead of phase 3

**Status: code + tests done, docs updated; not committed yet — waiting on explicit "commit this."**

### Spec / why reordered
`ARCHITECTURE.md` §9 lists batch Prometheus (phase 3) before the real-time API (phase 4). User chose to build the API first, reasoning it's more useful to have a live prediction endpoint and to monitor *that*. Technically it's also the cleaner order: a short-lived batch job can't be scraped by Prometheus (needs a push gateway / textfile collector); a long-running API is natively scrapeable via a `/metrics` route — so phase 3's instrumentation should target the API. Prometheus, circuit breaker (5), shadow (6), and load testing (7) all stay deferred and out of this phase.

### Decisions confirmed with the user (AskUserQuestion)
- **`/predict` input = `{home_team, away_team, date}`** with server-side stat lookup (same as batch), not a raw feature vector. More useful; needs `NBAdata/` mounted (already the pattern).
- **Shared inference EXTRACTED, not duplicated** — moved `load_predictor` + feature-assembly out of `run_nightly_predictions.py` into `serving/inference/predictor.py`, imported by both batch and API. One source of truth, same rationale as shared `features.py`/`mlflow_config.py`.
- Defaults taken (not separately asked): single `requirements.txt`; per-request `load_team_stats` (no caching yet); HTTP mapping 404 (unknown team/no stats) / 422 (missing feature) / 503 (season file absent or model unloaded).

### What got built
- `serving/inference/predictor.py` — moved `season_label_for_date`, `load_predictor` (registry@production + local-pkl fallback, unchanged), `load_team_stats`, `latest_team_stat_row`, `build_feature_row`, plus new thin helpers `assemble_features` (both teams' latest stats → one feature row, None if unusable) and `predict_from_features` (select SELECTED_FEATURES, return `(home_win, prob)`; `predict_proba` guarded with `hasattr`).
- `serving/api/main.py` — FastAPI app; model loaded once at startup via `lifespan` into `app.state.predictor`; `GET /health` (model_loaded + stats_available), `POST /predict`.
- `serving/api/models.py` — Pydantic v2 `PredictRequest`/`PredictResponse`.
- `serving/batch/run_nightly_predictions.py` — now imports the moved fns from `predictor`, keeps `fetch_schedule`/`games_on_date`/`PREDICTIONS_DIR`, loop rewired through `assemble_features`/`predict_from_features`. Re-exports the moved names (`__all__`) so existing imports keep resolving.
- `docker/Dockerfile.api` (uvicorn, 8000), `requirements.txt` (+fastapi, uvicorn[standard], pydantic, httpx), `pytest.ini` (+serving/inference, +serving/api), `ci.yml` (+API image build), `tests/test_api.py` (offline TestClient: happy path + 404/422/503/malformed).

### Two shadowing bugs caught by tests before review (both fixed)
- `models.py`: a Pydantic field named `date` with annotation `date | None` shadowed the `datetime.date` type → `TypeError: unsupported operand |: NoneType, NoneType`. Fixed by importing `from datetime import date as _date` and annotating with `_date`.
- `test_batch_predictions.py`: a test local var `predictor = load_predictor()` shadowed the `import predictor` module → `UnboundLocalError`. Renamed the local to `loaded`.

### Verification performed
- `python -m pytest tests/test_api.py tests/test_batch_predictions.py tests/test_mlflow_config.py tests/test_features.py -v` → **27 passed** offline. (The 3 `test_training_pipeline.py` tests still need `dvc pull` — pre-existing, untouched by this phase.)
- Two review agents spawned (code quality + plan conformance) — findings pending at time of writing.

### Open questions / follow-ups
- Nothing committed yet.
- Phase 3 (`/metrics` on this API + Grafana + drift) is the natural next step.

---

## Active task: CI/CD test suite (out-of-band request, not on the `ARCHITECTURE.md` roadmap)

**Status: done.**

### Spec
User asked to "create tests for a CI/CD that will test important features of our app" and to propose what's relevant to include. Note this reprioritizes ahead of `ARCHITECTURE.md` phase 2 (MLflow/DVC) — user-driven, same pattern as the earlier batch-before-real-time reprioritization.

### Key constraint that shaped the design
GitHub Actions runners are cloud/datacenter IPs. `README.md` already documents `stats.nba.com` blocking exactly that kind of traffic. So anything hitting the live NBA API (`scripts/data_pull/*`, batch job's `fetch_schedule()`) cannot be a real network call in CI — proposed and built around testing logic with fixtures instead, excluding live-API paths from CI entirely (flagged explicitly in `ci.yml`'s header comment and in README's Testing section, not silently skipped).

### What got built
- `tests/` (pytest): `test_features.py`, `test_merge_advanced_base_stats.py`, `test_batch_predictions.py`, `test_training_pipeline.py`, `test_model_artifacts.py`.
- `pytest.ini` (`pythonpath = scripts/modeling, scripts/data_prep, serving/batch`) — the project has no package/`__init__.py` structure, so tests need those directories on `sys.path` to import modules the same ad-hoc way the scripts already do.
- `pytest` added to `requirements.txt` directly (no separate dev-requirements file).
- `.github/workflows/ci.yml` — two jobs: `test` (pytest) and `docker-build` (`docker build -f docker/Dockerfile.batch`), both on push/PR to `main`.

### Revised after user review (same day)
User asked me to explain why `requirements-dev.txt` and `conftest.py` were necessary — good challenge, since neither was strictly required:
- **`conftest.py` → `pytest.ini`'s built-in `pythonpath` option.** Same effect (verified: reran the full suite with `pytest.ini` instead of the hand-rolled `sys.path` code in `conftest.py`, all 24 still passed), but `pythonpath` is pytest's standard mechanism for this — recognizable to anyone who knows pytest, vs. having to read custom `sys.path`-mutating code to understand why imports resolve. User picked this option.
- **`requirements-dev.txt` → merged into `requirements.txt`.** The original split was "don't make the Docker image carry test tooling it doesn't use," which is a real but minor concern for a project this size (pytest is a small dependency). User chose to merge — one file, simpler, at the cost of the batch image now installing pytest unnecessarily. Worth revisiting if the image ever needs to be genuinely minimal (e.g. a hard size budget).

### Repo cleanup (same day)
- User asked to move the "extra" markdown docs into a `documentation/` folder to keep the repo root clean. Moved `ARCHITECTURE.md`, `log.md`, `PROGRESS.md` (this file) via `git mv`. **Deliberately did not move** `README.md` (GitHub convention expects it at repo root to render on the repo homepage) or `CLAUDE.md` (Claude Code auto-loads project instructions from that exact root path — moving it would silently stop it from being picked up). Updated the one cross-reference that crossed the new folder boundary (`README.md`'s mention of `ARCHITECTURE.md`); the docs' mentions of each other needed no changes since they're now co-located.
- While investigating an unrelated user question ("why do we have 2 NBAdata folders now?"), found and removed a stray empty `NBAdata;C` directory at repo root. Root cause: an earlier `docker run -v "$(pwd)/NBAdata:/app/NBAdata" ...` call, run *before* discovering the `MSYS_NO_PATHCONV=1` fix for Git Bash's path mangling — the container-side path got corrupted into something containing a literal `C:`, and Docker Desktop auto-created that garbled path as an empty host directory. Untracked (git doesn't track empty dirs), harmless, but confusing to find sitting in the repo — removed with `rmdir`.

### Two judgment calls made without asking (flagged to user, not silently decided)
- Model-performance ("beats baseline") is **not** a hard CI gate — proposed as out of scope entirely for now rather than a soft/logged check, since accuracy isn't tested at all yet. If wanted later, it'd be a separate advisory (non-blocking) check, not a merge-blocking one — ML metrics are noisier than code correctness.
- Skipped lint/formatting (ruff etc.) — wasn't asked for, would be scope creep beyond "test important features."

### The tests double as regression guards for both bugs fixed this week
- `test_selected_features_excludes_postgame_leakage` — leaked `PTS`/`PLUS_MINUS` columns can't silently reappear in `SELECTED_FEATURES`.
- `test_merge_season_uses_season_from_input_data_not_hardcoded` / `test_merge_season_output_differs_by_input_season` — unit-level guard for the season-mislabeling bug, using fixture CSVs.
- `test_monthly_files_have_correct_internal_season_label` — same guard but against the actual committed `NBAdata/` files, i.e. would catch the bug's real-world symptom directly, not just the code path.
- `test_training_dataset_includes_all_seasons_with_no_missing_features` — guards the actual failure mode (5 of 6 seasons silently dropped by `dropna`), asserting each season contributes >100 usable training rows.

### Verification performed
- `python -m pytest tests/ -v` locally: **24 passed**, 0 failed (a few pre-existing pandas `DeprecationWarning`s from `merge_advanced_base_stats.py`'s `.apply(ffill)`, unrelated to this change, not touched).
- Did not re-verify by deliberately re-breaking the season-labeling code to confirm the guard tests fail red-then-green — skipped since the bug's failure mode was already directly observed and reproduced earlier this session before the fix landed (see the Phase 1 section below); considered redundant given time cost.
- Did not re-run `docker build` for `ci.yml`'s `docker-build` job in this pass — `docker/Dockerfile.batch` itself is unchanged since it was last built and verified working (see Phase 1 section).

### Open questions / follow-ups
- Nothing committed yet — waiting on explicit "commit this."
- Whether to eventually add an advisory (non-blocking) model-performance check once there's a stable baseline to compare against.

---

## Phase 2 — MLflow tracking + registry (`ARCHITECTURE.md`)

**Status: done.** Merged/verified via PR `phase2-mlflow-dvc` — all CI checks
green, including a from-scratch `dvc pull` from S3 in the `test` job (the real
proof a fresh machine can restore the data). Roadmap item 2 was "Introduce
MLflow (training logs runs) and DVC (data snapshots), and point the batch job
at the registry's Production model instead of a local pickle." All three
sub-tasks done: (a) MLflow training instrumentation + registry; (b) DVC data
snapshots on an AWS S3 remote; (c) batch job repointed at the registry.

### Decisions locked (user chose Pipeline; the other two are recommended defaults, easily reversible)
- **Model + scaler packaged as one `sklearn.Pipeline`** (user's call).
  Training wraps the *already-fit* `scaler` + `best_model` in
  `Pipeline([("scaler", scaler), ("model", best_model)])` purely as a
  packaging wrapper — it does **not** change the CV/selection math (still
  scales once up front, then CVs on scaled data). So the registered artifact
  takes raw features → prediction in one object, and serving can't drift the
  scaler from the model. Same-day verification confirmed the selected model
  and accuracy are unchanged (Bagging SVC, CV 0.6070 / test 0.6109).
- **sqlite tracking backend** (`sqlite:///mlflow.db` at repo root). The
  MLflow *model registry* (registered models + aliases) does **not** work
  with the default filesystem store — it needs a DB-backed tracking URI.
  `mlflow.db` + `mlruns/` are gitignored.
- **Registry aliases, not stages.** MLflow 3.x (pinned 3.14.0) has *removed*
  model-version stages entirely, so the roadmap's "Production" is modeled as
  an alias named `production` (`models:/nba-win-predictor@production`) rather
  than the deprecated `Production` stage.

### What got built
- `requirements.txt` — added `mlflow==3.14.0`, `dvc==3.67.1` (dvc pinned now
  but unused until sub-task (b)).
- `.gitignore` — added `mlflow.db`, `mlruns/`.
- `scripts/modeling/decision_tree_training.py` — `main()` now: sets the
  sqlite tracking URI + `nba-win-predictor` experiment; opens a parent run
  logging feature/row counts + both baselines; logs one **nested run per
  candidate model** (params via `get_params()` + CV mean/std); after picking
  the best, logs `best_model`/`cv_accuracy`/`test_accuracy`, builds the
  Pipeline, logs it with an inferred signature + input example, registers it
  as `nba-win-predictor`, and sets the `@production` alias to the new
  version. The existing local `best_model.pkl`/`scaler.pkl` dump is
  **untouched** — kept as the batch job's fallback artifacts (see sub-task c).
- `scripts/modeling/mlflow_config.py` — new. Single source of truth for the
  registry model name (`nba-win-predictor`), the `production` alias, the
  `models:/…@production` URI, and the `tracking_uri()` default (local sqlite,
  overridable via the `MLFLOW_TRACKING_URI` env var). Both training and the
  batch job import it, so they can't drift on model identity — same reasoning
  as the shared `features.py`. Training was refactored to import from it
  rather than hold its own copies.
- `serving/batch/run_nightly_predictions.py` (sub-task c) — new
  `load_predictor()` loads `models:/nba-win-predictor@production` from the
  registry and returns the pipeline; on any failure it prints why and falls
  back to wrapping the local pkls in the same kind of `Pipeline`. Callers use
  `predict`/`predict_proba` on **raw** features either way — the manual
  `scaler.transform` step is gone (the pipeline scales internally).
  - Why the fallback isn't over-engineering: the batch **container** mounts
    `NBAdata/` but not the host's `mlflow.db`, so the registry is genuinely
    unreachable there and the job would otherwise break. Fallback = it uses
    the pkls baked alongside the mounted data. Local host runs hit the real
    registry. A tracking *server* (future) would make the registry reachable
    from the container too, via `MLFLOW_TRACKING_URI`.

### Verification performed
- Ran the full training script: exit 0, model selection identical to Phase 1,
  registered `nba-win-predictor` v1 with `@production` set.
- Loaded `models:/nba-win-predictor@production` back and predicted on a
  synthetic feature row — works, and the pipeline exposes `predict_proba`
  (which the batch job needs for `HomeWinProbability`).
- `git status`: the retrained `.pkl` files are byte-identical to the
  committed ones (deterministic training), so the working-tree diff is purely
  the 3 intended files. No pkl restore needed.
- `pytest tests/ -q`: 24 passed (the training module now imports mlflow;
  nothing broke). Re-ran after the batch repoint — still 24 passed.
- Sub-task (c): ran the batch job locally, `--date 2025-04-01` — logged
  "Loaded model from registry: models:/nba-win-predictor@production" and saved
  7 predictions. Then forced the fallback (bogus `MLFLOW_TRACKING_URI`):
  logged "Registry model unavailable … falling back to local pkl artifacts"
  and still predicted. Verification CSV + throwaway sqlite deleted after.

### Benign warnings seen at log time (noted, not acted on)
- MLflow warns the inferred schema has integer column(s) — that's
  `Team1Home` (always 1). Harmless here; would only bite if that column
  could be missing at inference.
- MLflow's env-var recorder noticed a stray `IINGO_API_TOKEN` in the shell
  env during logging. The sklearn pipeline doesn't use it; it's just MLflow
  cataloguing process env vars. Ignored.

### Open questions / blockers before continuing Phase 2
- **DVC (sub-task b) is blocked on a decision.** The repo currently commits
  *all* `NBAdata/` CSVs to git. DVC-tracking them means
  `git rm -r --cached` the data and moving it behind a DVC remote — a real
  change to the repo's data workflow, and it needs a remote target (local
  dir? cloud?). Not started; needs the user's call on whether to pull data
  out of git and where the DVC remote lives.
### Sub-task (b): DVC on AWS S3 — what got set up
- **Remote:** S3 bucket `nba-prediction-dvc-ag` in `us-east-2`, dedicated to
  DVC. Config in `.dvc/config` (committed): `url = s3://nba-prediction-dvc-ag`,
  `region = us-east-2`. Bucket is fully private (all public access blocked);
  DVC authenticates with an IAM user.
- **Auth:** dedicated least-privilege IAM user `nba-dvc` with a customer policy
  (`nba-dvc-s3-access`) granting only `s3:ListBucket` + object
  `Get/Put/Delete` on that one bucket. The access keys live in the user's
  `~/.aws/credentials` (`[default]`), **outside the repo** — never committed.
  Connectivity verified via s3fs (List/Write/Read/Delete all OK) before moving
  data.
- **What's tracked by DVC (moved out of git):** `archive/`, `matchups/`,
  `monthly_stats/`, `NBA_Team_Boxscores_2024_25.csv`,
  `NBA_Training_Matchups_2019_2025.csv` — via per-path `.dvc` pointer files +
  a DVC-managed `NBAdata/.gitignore`. **Kept in git:** `best_model.pkl`,
  `scaler.pkl` (tiny, they're the batch fallback artifacts and already in the
  registry — keeping them in git means the model-artifact tests need no
  `dvc pull`). `dvc push` uploaded 29 objects; `dvc status -c` reports cache
  and remote in sync.
- **Deps gotcha:** `dvc-s3` (s3fs/aiobotocore) was enough for the s3fs probe,
  but `dvc push` itself also needs `boto3`. `requirements.txt` now pins
  `dvc[s3]==3.67.1` (pulls boto3 + s3fs), so CI gets the full S3 stack.
- **CI:** `ci.yml`'s test job gained a `dvc pull` step (before pytest) that
  reads `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from GitHub Actions
  secrets. The `docker-build` job is untouched (it doesn't need data).

### Closed out
- GitHub repo secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` added.
- CI on the `phase2-mlflow-dvc` PR went green — the `test` job's `dvc pull`
  from S3 succeeded on a clean runner, proving the fresh-machine restore path.
  This was the outstanding end-to-end proof; Phase 2 is fully verified.
- `test_model_artifacts.py` left as-is: it still asserts the local
  `best_model.pkl`/`scaler.pkl` exist + scaler shape, which now doubles as a
  guard on the batch job's fallback artifacts. No registry-loading test was
  added to CI on purpose — a fresh checkout has no `mlflow.db` and nothing
  registered, so such a test couldn't pass there (same reasoning that keeps
  live-API paths out of CI).
- Nothing committed yet — waiting on explicit "commit this."

---

## Phase 1 — batch prediction job (`ARCHITECTURE.md`)

**Status: done.**

### Spec (from `ARCHITECTURE.md`, phase 1 of the roadmap)
> Package the existing best_model.pkl/scaler.pkl into a Docker image and
> build the nightly batch job (same feature pipeline, predicts the next
> slate of games, writes results to a predictions store). Get batch working
> end to end first.

### What got built
- `scripts/modeling/features.py` — new. Pulls the feature contract
  (`STAT_MAP`, `SELECTED_FEATURES`, `resolve_stat_columns`) out of
  `decision_tree_training.py` into a shared module, so the batch job can't
  silently drift from what the model was actually trained on.
- `serving/batch/run_nightly_predictions.py` — new. Fetches the schedule for
  a target date via `nba_api`'s `ScheduleLeagueV2` (gives home/away directly,
  unlike the `LeagueGameFinder` + `MATCHUP`-string parsing used for
  historical pulls), builds features per game using the same monthly-stats
  join logic as training, predicts with the saved model + scaler, writes
  `NBAdata/predictions/predictions_<date>.csv`.
- `docker/Dockerfile.batch` — new. `python:3.12-slim`, code baked in,
  `NBAdata/` mounted as a volume at runtime (not baked in) so the container
  always sees current data without a rebuild.

### Decisions made along the way (and why)
- **Home team = Team1 always, in batch.** Training's `Team1`/`Team2` are
  alphabetical, not home/away; `Team1Home` is what actually encodes home
  status. For batch it's simpler and equally valid to always assign
  `Team1 = home team`, so `Team1Home = 1` for every row and the model's
  output is directly "probability the home team wins."
- **Stats lookup falls back to the team's latest available month** if the
  target game's exact (season, month) has no data yet — needed because a
  nightly job predicting an upcoming game can't join to that same month's
  completed-game aggregate the way historical training does.
- **Predictions land in `NBAdata/predictions/` as one CSV per date**,
  matching the project's existing CSV-based-everything convention rather
  than introducing a database.
- **Docker image excludes `NBAdata/`** — keeps the image small and reusable
  across model retrains without rebuilding; data is a runtime volume mount.

### A bug found and fixed along the way (not originally in scope)
While building the batch job's "join upcoming game to latest team stats"
logic, found `merge_advanced_base_stats.py` hardcoded `season = "2019-20"`
regardless of which season it was actually merging. Every combined
monthly-stats file (all 6 seasons, current + archived) had its `Season`
column stuck at `"2019-20"`, which meant `decision_tree_training.py`'s
season-matched join only ever matched the true 2019-20 matchups — the other
5 seasons were silently dropped by `dropna` on every training run. The model
was effectively trained on ~976 rows, not ~7,600. Fixed by deriving `season`
from the loaded data instead of hardcoding it, regenerated all 6 season
files (no re-pull from the API needed — the raw per-season files were
already correctly labeled), and retrained. New best model: Bagging SVC,
~60.7% CV / ~61.1% test accuracy vs. a ~55.4% home-court baseline. Full
detail in `README.md` ("Known issues") and `log.md`.

### Verification performed
- Ran `serving/batch/run_nightly_predictions.py --date 2025-04-01` and
  `--date 2025-04-02` locally against real 2024-25 season game dates
  (stand-in for "tomorrow," since today is mid-off-season) — produced sane
  predictions for every game on both dates.
- Built `docker/Dockerfile.batch`, ran the container with `NBAdata/` mounted
  as a volume against the same test dates — confirmed the container reaches
  the NBA API, builds features, predicts, and writes the output CSV back to
  the host filesystem correctly. (Note for Windows/Git Bash: prefix `docker
  run` with `MSYS_NO_PATHCONV=1` or the `/app/...` container path gets
  rewritten to a Windows Git path.)
- The test-only prediction CSVs were deleted afterward; they weren't real
  nightly output, just verification artifacts.

### Open questions / follow-ups (not yet decided)
- Whether `NBAdata/predictions/` should be committed to git going forward,
  or gitignored as pure runtime output — existing project convention commits
  generated CSVs (training set, model, scaler), but a growing pile of
  per-date prediction files is a different kind of artifact. Not decided.
- The month-ordering issue noted in passing: forward-fill in
  `merge_advanced_base_stats.py` sorts months numerically (1..6 then
  10..12), not chronologically, for a season that spans two calendar years.
  Doesn't appear to affect current results since all in-season months
  already have real pulled data, but flagged here in case it matters later.
- Next phase per `ARCHITECTURE.md`: MLflow (training logs runs) + DVC (data
  snapshots), then point the batch job at the registry instead of the local
  `.pkl` files.
