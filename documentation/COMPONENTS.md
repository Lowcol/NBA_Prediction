# Components in depth

This document explains each part of the NBA prediction project in detail: the
repo layout, how data is stored and versioned, model tracking, and every stage
of the pipeline and serving layer. `README.md` has the short project description
and the commands to run things; this is the thorough version.

## Repo structure

```
scripts/
  data_pull/     # Hits the NBA stats API (via nba_api) and writes raw CSVs into NBAdata/
  data_prep/     # Turns per-game pulls into the trailing rolling-window team stats used for training/serving
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
  team_game_logs/                  # [DVC] One row per team per game (Base+Advanced box score) — rolling-stats input
  rolling_stats/
    nba_team_rolling_stats_*.csv          # [DVC] One row per (team, game): that team's trailing 10-game averages, leakage-free
    nba_team_current_rolling_stats_*.csv  # [git] One row per team: current-form snapshot — the model's input at prediction time
  predictions/                     # Batch job output, one CSV per predicted slate date
  NBA_Training_Matchups_2019_2025.csv  # [DVC] Combined training set built by decision_tree_training.py
  best_model.pkl, scaler.pkl       # [git] Latest trained model + StandardScaler (kept in git as the batch fallback)
```

`monthly_stats/` and `archive/historical/monthly_stats/` (the pre-2026-08-28
month-granularity stats) still exist on disk but are no longer read by the
pipeline — see `documentation/TRAINING.md` §2 for why they were replaced.

`[DVC]` paths (the bulk **training** data) live in S3, not git — run `dvc pull` after cloning to fetch them (see below). `[git]` paths are committed, so the model and the current-season stats it needs to *serve* predictions are available straight after cloning.

See `documentation/ARCHITECTURE.md` for the plan behind productionizing this: batch first, then a real-time API, versioning, monitoring, and reliability patterns. `documentation/TRAINING.md` explains, in plain language, how the model is trained and exactly what it learns from. `documentation/log.md` is the dated changelog of major changes, and `documentation/PROGRESS.md` is the working scratchpad — task specs, decisions, and open questions behind those changes.

## Data (DVC + S3)

The bulk **training** data under `NBAdata/` (`matchups/`, `team_game_logs/`, the per-game `rolling_stats/nba_team_rolling_stats_*.csv` files, `archive/`, and the combined training CSV) is versioned with [DVC](https://dvc.org/) and stored in S3 (`s3://nba-prediction-dvc-ag`, `us-east-2`), **not** in git. A fresh clone only has the `.dvc` pointer files for that data until you pull:

```
pip install -r requirements.txt          # includes dvc[s3]
dvc pull                                 # downloads the training data from S3
```

`dvc pull` needs AWS credentials for the bucket — configure `~/.aws/credentials` (or the standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars) before running it.

What is **not** DVC-tracked, and so is available straight after cloning with no credentials: `best_model.pkl`/`scaler.pkl` (the trained model + scaler) and `rolling_stats/nba_team_current_rolling_stats_*.csv` (the current-form snapshot the model reads as its input features — one row per team, its trailing 10-game averages as of the last data refresh). Together those are everything the batch job and the API need to *serve* predictions — only *re-training* needs the DVC data pulled. The snapshot is exactly that, a snapshot: to keep predictions using up-to-date team form during a season, refresh it by re-running `team_game_logs_pull.py` → `build_rolling_team_stats.py` → `build_current_rolling_snapshot.py` (see below).

## MLflow tracking & registry

Training logs to MLflow (local sqlite store, `mlflow.db`, gitignored). Each run records the per-model CV scores, packages the best scaler+model as one `sklearn.Pipeline`, registers it as **`nba-win-predictor`**, and moves the **`@production`** alias to that new version. The batch job loads `models:/nba-win-predictor@production` from the registry, falling back to the local `best_model.pkl`/`scaler.pkl` when the registry isn't reachable. Registry name and alias live in `scripts/modeling/mlflow_config.py`, shared by training and serving so they can't drift.

## Pipeline

1. **Data pull** (`scripts/data_pull/`)

- `nbaPull_19-25_matchups.py` — pulls every game 2019-20 through 2025-26 (Regular Season, Playoffs, PlayIn) via `LeagueGameFinder`, pairs the two teams per `GAME_ID` into one row, and derives `Team1Home` from the `MATCHUP` field (`"vs."` = home, `"@"` = away). Writes to `NBAdata/matchups/`.
- `team_game_logs_pull.py` — pulls per-team, per-game Base + Advanced box scores (`TeamGameLogs`) for every season/season-type, one bulk call per (season, season_type, measure_type) — not one call per game. Writes to `NBAdata/team_game_logs/`. This is what `build_rolling_team_stats.py` (below) turns into the model's actual training input; it shares its `GAME_ID`s exactly with `nbaPull_19-25_matchups.py`'s output, which is what lets the two be joined precisely instead of approximately.
- `nba_api_pull.py`, `monthly_stats_pull.py` / `monthly_team_stat_pull.py` — superseded by `team_game_logs_pull.py` as of 2026-08-28 (see `TRAINING.md` §2); left in place, unreferenced by the pipeline, rather than deleted.
- Note: `stats.nba.com` blocks requests from some cloud/datacenter networks — if a pull times out or the connection resets from a sandboxed environment, try running it from a normal home/office network first before assuming the code is broken.

2. **Data prep** (`scripts/data_prep/`)

- `build_rolling_team_stats.py` — turns `team_game_logs_pull.py`'s per-game output into trailing 10-game rolling averages per team per game (`NBAdata/rolling_stats/nba_team_rolling_stats_<season>.csv`), using `.shift(1)` before `.rolling(10)` so a game's own stats can never leak into its own average. A team needs 3+ prior games before it gets real numbers; Pre Season games don't count toward any team's window. See `TRAINING.md` §2 for the full mechanics.
- `build_current_rolling_snapshot.py` — condenses the per-game rolling file down to one row per team (that team's most recent valid rolling average) for serving to read at prediction time. If a team has no valid games yet in the target season (e.g. the season just started), falls back to that team's last valid row from the *previous* season — the one place in this pipeline where a team's stats are allowed to cross a season boundary (training does not do this; see "Known issues" below).
- `merge_advanced_base_stats.py` — superseded by the two scripts above as of 2026-08-28; left in place, unreferenced by the pipeline.

3. **Modeling** (`scripts/modeling/`)

- `decision_tree_training.py` — the main entry point. Auto-discovers every season under `NBAdata/matchups/` and `NBAdata/rolling_stats/`, joins each matchup to that game's own rolling stats **on `GAME_ID`** (an exact join, not an approximate one), trains and cross-validates 6 classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVC-RBF, Bagging SVC), and saves the best one.
- `data_exploration.py` — ad hoc correlation/heatmap exploration against a season's box scores; not part of the training pipeline.

For a plain-language walkthrough of what the model learns from and how training works step by step, see `documentation/TRAINING.md`.

## Serving

4. **Batch predictions** (`serving/batch/`)

- `run_nightly_predictions.py` — predicts an upcoming slate of games. Fetches that day's schedule via `nba_api`'s `ScheduleLeagueV2` (home/away comes directly from the schedule, no `MATCHUP`-string parsing needed), looks up each team's current-form snapshot using the same feature contract as training (`scripts/modeling/features.py`), and writes `NBAdata/predictions/predictions_<date>.csv` with a predicted winner and win probability per game. It loads the `@production` model from the MLflow registry, falling back to the local `best_model.pkl`/`scaler.pkl` when the registry isn't reachable — which is what happens inside the container, since `mlflow.db` isn't mounted, so it always uses the mounted pkls.
- Note on `--date <past date>`: the snapshot lookup always returns each team's *most recent* rolling average on file, not "as of that specific past date" — so predicting a past date now reflects team form at the time of the last data refresh, not form as of that date. This is a deliberate simplification for live serving (it's always correct for "predict tomorrow's slate"), but it means this flag is no longer useful for point-in-time historical backtesting the way it arguably was under the old month-keyed lookup. Worth knowing if you use `--date` to spot-check the model against a past slate.
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

`tests/` covers the pure-logic pieces of the pipeline (feature resolution, season/date parsing, stats lookup and fallback, home/away parsing, the registry→pkl model-loading fallback, the rolling-window computation) and the real-time API (`/health` and `/predict` happy path + error branches, with the model and stats mocked so the tests stay offline), plus a few regression guards for bugs that have bitten this project before. The most important one as of 2026-08-28 is `test_build_rolling_team_stats.py`'s leakage regression test, which asserts a game's own stats can never appear in that same game's rolling average — the entire point of the rolling-window rebuild. Older guards (from when stats were month-keyed) still exist and still pass against the now-superseded `merge_advanced_base_stats.py` path: that every combined monthly-stats file's `Season` column actually matches its filename, and that the full training set uses all seasons instead of silently dropping some. The model-artifact and pure-logic tests run offline against the git-tracked `best_model.pkl`/`scaler.pkl` and the current-form snapshot, but the data-dependent tests also read `NBAdata/matchups/` and `NBAdata/team_game_logs/`, which are DVC-tracked — run `dvc pull` first (needs AWS creds) or those tests will fail on a fresh clone.

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

### Residual, smaller leakage (fixed, 2026-08-28)

Team stats used to be joined to each matchup at **month granularity** (`merge_advanced_base_stats.py` / the merge logic in `decision_tree_training.py`): a game played in November was joined to November's aggregate stats, which included that same game's own contribution to the month's totals. Fixed by rebuilding team stats as trailing 10-game rolling averages computed strictly from each team's games *before* the target game's own date (`build_rolling_team_stats.py`, joined on the game's own `GAME_ID` rather than `(Team, Season, Month)`) — see `TRAINING.md` §2. This was the "bigger change" the original note here said fixing this properly would require.

### Monthly stats mislabeled by season (fixed, 2026-07-16; path superseded 2026-08-28)

`merge_advanced_base_stats.py` hardcoded `season = "2019-20"` when building its output, regardless of which season's base/advanced files it was actually merging. Every `nba_team_combined_stats_*.csv` file — current and archived, all 6 seasons at the time — ended up with its internal `Season` column stuck at `"2019-20"`. Since `decision_tree_training.py` joined matchups to stats on `["Team", "Season", "Month"]`, only the true 2019-20 matchups ever found a match; the other 5 seasons' rows were silently dropped by `dropna`, meaning the model was training on ~976 rows instead of the ~7,600 the pipeline was supposed to produce. The script was fixed to derive `season` from the loaded data itself. As of 2026-08-28 this whole month-keyed path (`merge_advanced_base_stats.py` and the `["Team", "Season", "Month"]` join) is no longer used by the pipeline at all — superseded by the `GAME_ID`-keyed rolling-stats join above, which has no `Season`/`Month` matching step to mislabel in the first place.

### Serving's "current form" snapshot isn't date-bounded (known limitation, since 2026-08-28)

`build_current_rolling_snapshot.py` always returns each team's *most recent* valid rolling average on file — it doesn't look up "stats as of the specific date being predicted." This is correct and intentional for live serving (predicting an upcoming game should always use the freshest data), but it means `run_nightly_predictions.py --date <past date>` no longer reproduces "what the model would have seen on that date" once a season has fully played out — see the note in "Serving" above. If point-in-time historical backtesting is needed later, it would mean teaching the snapshot lookup (or a variant of it) to filter the rolling-stats file to games before a given date, rather than always taking the last row.

### Training/serving season-boundary asymmetry (intentional, since 2026-08-28)

Training (`decision_tree_training.py`) drops any game where a team has fewer than 3 prior games of rolling history — including, unavoidably, every team's first couple of games each new season. Serving's snapshot builder (`build_current_rolling_snapshot.py`) can't afford to do the same (it must return *something* for every team, even on day 2 of a new season), so it instead falls back to that team's last valid row from the *previous* season when the current season has nothing usable yet. This is the one place in the pipeline where a team's stats are allowed to cross a season boundary — deliberate, and asymmetric between training and serving on purpose.
