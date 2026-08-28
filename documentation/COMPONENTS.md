# Components in depth

This document explains each part of the NBA prediction project in detail: the
repo layout, how data is stored and versioned, model tracking, and every stage
of the pipeline and serving layer. `README.md` has the short project description
and the commands to run things; this is the thorough version.

## Repo structure

```
scripts/
  data_pull/     # Hits the NBA stats API (via nba_api) and writes raw CSVs into NBAdata/
  data_prep/     # Merges/reshapes raw pulls into the monthly team-stat tables used for training
  modeling/      # Trains and evaluates the prediction model; features.py is the shared feature contract

serving/
  inference/     # Shared model-loading + feature-assembly code (used by both batch and api)
  batch/         # Nightly batch job: predicts an upcoming slate of games
  api/           # Real-time FastAPI service: POST /predict, GET /health

docker/
  Dockerfile.batch  # Containerizes the batch job
  Dockerfile.api    # Containerizes the real-time API

tests/           # pytest suite — see "Testing" below

documentation/
  ARCHITECTURE.md  # Plan for productionizing the model
  TRAINING.md      # Plain-language explanation of how the model is trained and what it learns from
  COMPONENTS.md    # This file
  log.md           # Dated changelog of major changes
  PROGRESS.md      # Working scratchpad: task specs, decisions, open questions

NBAdata/                           # (see "Data (DVC + S3)" below — the training data is not in git)
  matchups/                        # [DVC] One row per game, per season (2019-20 .. 2025-26)
  monthly_stats/                   # [git] Current-season base/advanced/combined team stats — the model's input at prediction time
  archive/historical/monthly_stats/# [DVC] Past-season monthly stats (2019-20 .. 2023-24), used for training
  predictions/                     # Batch job output, one CSV per predicted slate date
  NBA_Training_Matchups_2019_2025.csv  # [DVC] Combined training set built by decision_tree_training.py
  best_model.pkl, scaler.pkl       # [git] Latest trained model + StandardScaler (kept in git as the batch fallback)
```

`[DVC]` paths (the bulk **training** data) live in S3, not git — run `dvc pull` after cloning to fetch them (see below). `[git]` paths are committed, so the model and the current-season stats it needs to *serve* predictions are available straight after cloning.

See `documentation/ARCHITECTURE.md` for the plan behind productionizing this: batch first, then a real-time API, versioning, monitoring, and reliability patterns. `documentation/TRAINING.md` explains, in plain language, how the model is trained and exactly what it learns from. `documentation/log.md` is the dated changelog of major changes, and `documentation/PROGRESS.md` is the working scratchpad — task specs, decisions, and open questions behind those changes.

## Data (DVC + S3)

The bulk **training** data under `NBAdata/` (`matchups/`, `archive/`, and the combined training CSVs) is versioned with [DVC](https://dvc.org/) and stored in S3 (`s3://nba-prediction-dvc-ag`, `us-east-2`), **not** in git. A fresh clone only has the `.dvc` pointer files for that data until you pull:

```
pip install -r requirements.txt          # includes dvc[s3]
dvc pull                                 # downloads the training data from S3
```

`dvc pull` needs AWS credentials for the bucket — configure `~/.aws/credentials` (or the standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars) before running it.

What is **not** DVC-tracked, and so is available straight after cloning with no credentials: `best_model.pkl`/`scaler.pkl` (the trained model + scaler) and `monthly_stats/` (the current-season team stats the model reads as its input features). Together those are everything the batch job and the API need to *serve* predictions — only *re-training* needs the DVC data pulled. The current-season `monthly_stats/` is a snapshot, though: to keep predictions using up-to-date team form during a season, refresh it with the `data_pull` scripts (see below).

## MLflow tracking & registry

Training logs to MLflow (local sqlite store, `mlflow.db`, gitignored). Each run records the per-model CV scores, packages the best scaler+model as one `sklearn.Pipeline`, registers it as **`nba-win-predictor`**, and moves the **`@production`** alias to that new version. The batch job loads `models:/nba-win-predictor@production` from the registry, falling back to the local `best_model.pkl`/`scaler.pkl` when the registry isn't reachable. Registry name and alias live in `scripts/modeling/mlflow_config.py`, shared by training and serving so they can't drift.

## Pipeline

1. **Data pull** (`scripts/data_pull/`)

- `nbaPull_19-25_matchups.py` — pulls every game 2019-20 through 2024-25 (Regular Season, Playoffs, PlayIn) via `LeagueGameFinder`, pairs the two teams per `GAME_ID` into one row, and derives `Team1Home` from the `MATCHUP` field (`"vs."` = home, `"@"` = away). Writes to `NBAdata/matchups/`.
- `nba_api_pull.py` — pulls per-team box scores (`LeagueGameLog`) per season.
- `monthly_stats_pull.py` / `monthly_team_stat_pull.py` — pull per-team, per-month base/advanced stats (`LeagueDashTeamStats`).
- Note: `stats.nba.com` blocks requests from some cloud/datacenter networks — if a pull times out or the connection resets from a sandboxed environment, try running it from a normal home/office network first before assuming the code is broken.

2. **Data prep** (`scripts/data_prep/`)

- `merge_advanced_base_stats.py` — merges a season's base + advanced monthly stats per team/month, builds the full team x month grid, and forward-fills gaps into `nba_team_combined_stats_<season>.csv`.

3. **Modeling** (`scripts/modeling/`)

- `decision_tree_training.py` — the main entry point. Auto-discovers every season under `NBAdata/matchups/` and `NBAdata/monthly_stats/` (plus their `archive/historical/` equivalents), joins each matchup to that team/month's stats, trains and cross-validates 6 classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVC-RBF, Bagging SVC), and saves the best one.
- `data_exploration.py` — ad hoc correlation/heatmap exploration against a season's box scores; not part of the training pipeline.

For a plain-language walkthrough of what the model learns from and how training works step by step, see `documentation/TRAINING.md`.

## Serving

4. **Batch predictions** (`serving/batch/`)

- `run_nightly_predictions.py` — predicts an upcoming slate of games. Fetches that day's schedule via `nba_api`'s `ScheduleLeagueV2` (home/away comes directly from the schedule, no `MATCHUP`-string parsing needed), looks up each team's most recent monthly stats using the same feature contract as training (`scripts/modeling/features.py`), and writes `NBAdata/predictions/predictions_<date>.csv` with a predicted winner and win probability per game. It loads the `@production` model from the MLflow registry, falling back to the local `best_model.pkl`/`scaler.pkl` when the registry isn't reachable — which is what happens inside the container, since `mlflow.db` isn't mounted, so it always uses the mounted pkls.
- Run locally:
  ```
  python serving/batch/run_nightly_predictions.py                  # predicts tomorrow's slate
  python serving/batch/run_nightly_predictions.py --date 2025-04-01 # predicts a specific date (also useful for testing against a past date)
  ```
- Run via Docker (`docker/Dockerfile.batch`) — the image holds the code only; `NBAdata/` (model, scaler, stats) is mounted at runtime so the container always reads/writes the current data on disk. The model and current-season stats are in git, so a fresh clone can build and run without a `dvc pull`:
  ```
  docker build -f docker/Dockerfile.batch -t nba-batch:latest .
  docker run --rm -v "$(pwd)/NBAdata:/app/NBAdata" nba-batch:latest --date 2025-04-01
  ```
  (On Windows Git Bash, prefix with `MSYS_NO_PATHCONV=1` — otherwise Git Bash rewrites the container-side `/app/...` path.)

5. **Real-time API** (`serving/api/`)

- A FastAPI service that predicts a single game on demand. `POST /predict` with `{"home_team": "...", "away_team": "...", "date": "YYYY-MM-DD"}` (date optional, defaults to today) returns the predicted winner and the home team's win probability. It does the same stat lookup and feature-building as the batch job — the shared code lives in `serving/inference/predictor.py` — and loads the `@production` model once at startup (falling back to the local pkls, same as the batch job). `GET /health` reports whether the model loaded and whether the current season's stats are on file.
- A minimal web UI is served at `/` (`serving/api/static/index.html`): a single self-contained page with two team dropdowns that POSTs to `/predict` and shows the predicted winner and win probability.
- Error responses: unknown team or no stats on file → 404; stats present but a required feature is missing → 422; the season's stats file isn't provisioned, or the model failed to load → 503.
- Run locally — the model and current-season stats are in git, so this works straight after cloning (no `dvc pull`):
  ```
  uvicorn main:app --app-dir serving/api --reload      # serves on http://localhost:8000
  ```
  Then, e.g.:
  ```
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"home_team": "Denver Nuggets", "away_team": "Miami Heat", "date": "2025-04-01"}'
  ```
- Run via Docker (`docker/Dockerfile.api`) — unlike the batch image, this one is **self-contained**: it bakes in the trained model + current-season stats (both git-tracked), so no volume mount and no `dvc pull` are needed to serve:
  ```
  docker build -f docker/Dockerfile.api -t nba-api:latest .
  docker run --rm -p 8000:8000 nba-api:latest        # then open http://localhost:8000
  ```
  The baked-in stats are a snapshot from build time. To serve fresher stats without rebuilding, mount an updated `NBAdata/` over the image's copy: `docker run --rm -p 8000:8000 -v "$(pwd)/NBAdata:/app/NBAdata" nba-api:latest`.

## Testing

`tests/` covers the pure-logic pieces of the pipeline (feature resolution, season/date parsing, stats lookup and fallback, home/away parsing, the registry→pkl model-loading fallback) and the real-time API (`/health` and `/predict` happy path + error branches, with the model and stats mocked so the tests stay offline), plus a few regression guards for bugs that have bitten this project before — most notably that every combined monthly-stats file's `Season` column actually matches its filename, and that the full training set uses all 6 seasons instead of silently dropping five of them. The model-artifact and pure-logic tests run offline against the git-tracked `best_model.pkl`/`scaler.pkl` and `monthly_stats/`, but the data-dependent tests also read `NBAdata/matchups/`, which is DVC-tracked — run `dvc pull` first (needs AWS creds) or those tests will fail on a fresh clone.

What's deliberately **not** covered here: `scripts/data_pull/*` and the batch job's live schedule fetch. Both need `stats.nba.com`, which blocks cloud/datacenter IPs — exactly what CI runners are (see the network note above). Those stay manual/local-only.

```
pip install -r requirements.txt
dvc pull                        # fetch DVC-tracked data (needs AWS creds); skip only if running pkl/pure-logic tests
python -m pytest
```

CI (`.github/workflows/ci.yml`) runs `dvc pull` (using AWS secrets) and this test suite, plus a `docker build` of `docker/Dockerfile.batch`, on every push/PR to `main`.

## Known issues

### Data leakage (fixed, 2026-07-15)

The original `nbaPull_19-25_matchups.py` pulled `PTS` and `PLUS_MINUS` from the **completed** game itself, and the label `Team1Win` was derived from `Team1_PTS > Team2_PTS`. Those same leaked columns were then used as **input features** in training — the model was effectively being handed the final score to predict who won, which would have made any reported accuracy meaningless. This has been fixed: those columns are no longer used as features, and `Team1Home` (previously missing entirely — `Team1`/`Team2` were assigned by alphabetical sort order, not home/away) was added as a legitimate pregame feature instead.

### Residual, smaller leakage (open)

Team stats are joined to each matchup at **month granularity** (`merge_advanced_base_stats.py` / the merge logic in `decision_tree_training.py`): a game played in November is joined to November's aggregate stats, which include that same game's contribution to the month's totals. This is much less severe than the original bug (the aggregate is diluted across ~15 games instead of being the game's own score) but is not fully pregame-safe. Fixing this properly would mean rebuilding team stats as trailing rolling averages computed strictly from games before each matchup's date, which is a bigger change than the minimal fix applied so far.

### Monthly stats mislabeled by season (fixed, 2026-07-16)

`merge_advanced_base_stats.py` hardcoded `season = "2019-20"` when building its output, regardless of which season's base/advanced files it was actually merging. Every `nba_team_combined_stats_*.csv` file — current and archived, all 6 seasons — ended up with its internal `Season` column stuck at `"2019-20"`. Since `decision_tree_training.py` joins matchups to stats on `["Team", "Season", "Month"]`, only the true 2019-20 matchups ever found a match; the other 5 seasons' rows were silently dropped by `dropna`, meaning the model was training on ~976 rows instead of the ~7,600 the pipeline was supposed to produce. The script now derives `season` from the loaded data itself and regenerates every season's combined-stats file; all 6 seasons now correctly contribute rows.
