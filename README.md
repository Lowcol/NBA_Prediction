# NBA Prediction

A machine learning pipeline that predicts the winner of upcoming NBA games from
team-level stats. It pulls game and team data from the NBA stats API, builds a
training set, trains and cross-validates several classifiers, and serves the best
model two ways: a nightly **batch** job that predicts a whole slate of games, and
a real-time **API** (with a small web UI) that predicts a single game on demand.

The model and the current-season stats it needs are committed, so you can run the
predictor straight after cloning. The bulk training data is versioned with DVC
(stored in S3), and models are tracked and registered with MLflow so the batch job
and the API always serve the same `@production` model.

## How to run

There are two ways to run this, depending on what you want:

- **[Easy — just the prediction page (Docker)](#easy-setup-just-the-prediction-page-docker)** — one build, one run, open the browser. Best if you just want to use the predictor.
- **[Full — everything locally (Python)](#full-setup-run-everything-locally-python)** — install deps and run the pieces directly. Needed to re-train the model, run the batch job, or run the tests.

---

## Easy setup: just the prediction page (Docker)

The quickest way to get the prediction web page running.

```
docker build -f docker/Dockerfile.api -t nba-api:latest .
docker run --rm -p 8000:8000 nba-api:latest
```

Then open [http://localhost:8000](http://localhost:8000), pick two teams, and get a prediction.

---

## Full setup: run everything locally (Your going to need AWS cred)

Use this if you want to re-train the model, run the batch job, or run the tests.

### 1. Install dependencies

```
pip install -r requirements.txt          # includes dvc[s3]
```

### 2. Get the data (only needed to retrain or run the full test suite)

**To just run the API or batch predictions, skip this step.** The trained model
(`best_model.pkl` / `scaler.pkl`) and the current-form team stats the model reads
as input (`NBAdata/rolling_stats/nba_team_current_rolling_stats_*.csv`) are
committed to git, so serving works straight after cloning — no credentials,
no download.

The bulk **training** data (historical matchups + per-game team box scores,
`NBAdata/matchups/` + `NBAdata/team_game_logs/` + `archive/`)
is not in git — it's versioned with DVC and stored in S3. You only need it to
_re-train_ the model or to run the data-dependent tests. Pull it (needs AWS
credentials for the bucket, via `~/.aws/credentials` or the
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars):

```
dvc pull                                 # downloads the training data from S3
```

### 3. Train the model

```
python scripts/modeling/decision_tree_training.py    # builds the training set, trains, saves best_model.pkl
```

This also logs the run to MLflow, registers the best model as `nba-win-predictor`,
and points the `@production` alias at it.

### 4. Predict a slate of games (batch)

```
python serving/batch/run_nightly_predictions.py                  # predicts tomorrow's slate
python serving/batch/run_nightly_predictions.py --date 2025-04-01 # predicts a specific date
```

Output is written to `NBAdata/predictions/predictions_<date>.csv`.

### 5. Predict a single game (real-time API)

```
uvicorn main:app --app-dir serving/api --reload      # serves on http://localhost:8000
```

Open [http://localhost:8000](http://localhost:8000) for the web UI (pick two teams, get a prediction), or
call the API directly:

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Denver Nuggets", "away_team": "Miami Heat", "date": "2025-04-01"}'
```

`GET /health` reports whether the model loaded and whether the current season's
stats are on file.

### Run the web UI with Docker (no Python setup)

The quickest way to get the prediction page running. The API image is
**self-contained** — it bundles the trained model and the current-season stats —
so all you need is Docker. Build it once, then run:

```
docker build -f docker/Dockerfile.api -t nba-api:latest .
docker run --rm -p 8000:8000 nba-api:latest
```

Then open [http://localhost:8000](http://localhost:8000), pick two teams, and get a prediction. No
`pip install`, no `dvc pull`, no credentials, no volume mounts. Stop it with
`Ctrl+C`.

(If a prebuilt image is ever published to a registry, `docker pull <image>`
replaces the `docker build` step — the `docker run` line is the same.)

### Run the tests

```
dvc pull                        # fetch DVC-tracked data (needs AWS creds); skip only for pkl/pure-logic tests
python -m pytest
```

The batch job can also run in Docker (`docker/Dockerfile.batch`) — see the
components doc below.

## Learn more

- **[documentation/COMPONENTS.md](documentation/COMPONENTS.md)** — every part of the
  project explained in depth: repo layout, DVC/S3 data, MLflow tracking, each
  pipeline stage, both serving paths (including Docker), testing, and known issues.
- **[documentation/TRAINING.md](documentation/TRAINING.md)** — plain-language
  walkthrough of how the model is trained and exactly what it learns from.
- **[documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md)** — the plan for
  productionizing the model (batch, real-time, versioning, monitoring, reliability).
- **[documentation/log.md](documentation/log.md)** — dated changelog of major changes.
- **[documentation/PROGRESS.md](documentation/PROGRESS.md)** — working scratchpad:
  task specs, decisions, and open questions.
