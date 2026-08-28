# Components in depth

Repo layout, data storage/versioning, and every pipeline/serving stage.
`README.md` has the short version and run commands; this is the detailed one.

## Repo structure

```
scripts/
  data_pull/     # Hits the NBA API, writes raw CSVs into NBAdata/
  data_prep/     # Per-game pulls -> trailing rolling-window team stats
  modeling/      # Trains/evaluates the model; features.py is the shared feature contract

serving/
  inference/     # Shared model-loading + feature-assembly (batch + api)
  batch/         # Nightly batch job: predicts an upcoming slate
  api/           # Real-time FastAPI: POST /predict, GET /health

docker/          # Dockerfile.batch, Dockerfile.api
tests/           # pytest suite
documentation/   # ARCHITECTURE.md, TRAINING.md, COMPONENTS.md (this file), log.md, PROGRESS.md

NBAdata/
  matchups/                             # [DVC] one row per game, per season
  team_game_logs/                       # [DVC] one row per team per game (box score) -- rolling-stats input
  rolling_stats/
    nba_team_rolling_stats_*.csv        # [DVC] one row per (team, game): trailing 10-game averages
    nba_team_current_rolling_stats_*.csv # [git] one row per team: current-form snapshot, serving's input
  injury_reports/                       # [DVC] historical injury data (5 of 7 seasons, 2021-22+)
  predictions/                          # batch job output
  NBA_Training_Matchups_2019_2025.csv   # [DVC] combined training set
  best_model.pkl, scaler.pkl            # [git] latest model + scaler (batch fallback)
```

`monthly_stats/` (old month-granularity data, pre-2026-08-28) still exists on
disk but is unused — see `TRAINING.md` §2. `[DVC]` paths need `dvc pull`;
`[git]` paths are available straight after cloning.

## Data (DVC + S3)

Bulk training data (`matchups/`, `team_game_logs/`, per-game `rolling_stats/`,
`injury_reports/`, the combined training CSV) is DVC-tracked in S3 (`s3://nba-prediction-dvc-ag`).
Pull it with:

```
pip install -r requirements.txt   # includes dvc[s3]
dvc pull                          # needs AWS creds
```

Not DVC-tracked (available straight after cloning): `best_model.pkl`/`scaler.pkl`
and the current-form snapshot (`rolling_stats/nba_team_current_rolling_stats_*.csv`)
— everything serving needs. The snapshot is a point-in-time refresh; re-run
`team_game_logs_pull.py` → `build_rolling_team_stats.py` → `build_current_rolling_snapshot.py`
to update it during a season.

## MLflow tracking & registry

Training logs to MLflow (local sqlite `mlflow.db`, gitignored), packages the
best scaler+model as one `Pipeline`, registers it as `nba-win-predictor`, and
points the `@production` alias at it. Serving loads that alias, falling back
to the local pkls if the registry isn't reachable. Shared config:
`scripts/modeling/mlflow_config.py`.

## Pipeline

**1. Data pull** (`scripts/data_pull/`)
- `nbaPull_19-25_matchups.py` — every game 2019-20→2025-26 via `LeagueGameFinder`, one row per game, `Team1Home` from the `MATCHUP` field.
- `team_game_logs_pull.py` — per-team, per-game Base+Advanced box scores via `TeamGameLogs` (bulk calls, not per-game). Shares `GAME_ID`s exactly with the matchups file, enabling an exact join.
- `nba_api_pull.py`, `monthly_stats_pull.py`/`monthly_team_stat_pull.py` — superseded, left unreferenced.
- `injury_report_pull.py` — historical NBA injury reports via the `nbainjuries` package (needs a JVM, `tabula-py` — training/offline-only, kept out of `requirements.txt` and out of the serving Docker images on purpose). Only covers 2021-22 onward; the NBA's public archive doesn't go back further.
- `stats.nba.com` blocks some cloud/datacenter networks — run pulls from a normal network if they fail.

**2. Data prep** (`scripts/data_prep/`)
- `build_rolling_team_stats.py` — per-game box scores → trailing 10-game rolling averages (leakage-free by construction) + `RestDays`/`B2B` (calendar gap since the team's true previous game, any SeasonType, capped at 5 days — `RestDays` itself isn't a model feature as of a 2026-08-28 SHAP audit, only `B2B` is; kept as a data column anyway). See `TRAINING.md` §2.
- `build_current_rolling_snapshot.py` — condenses that to one "current form" row per team for serving. Falls back to the team's last valid row from the *previous* season if it has none yet this season (see "Known issues").
- `build_injury_features.py` — per-team, per-date counts of `Out`/`Doubtful` players from the historical injury reports. Only produces output for the 5 seasons with coverage.
- `merge_advanced_base_stats.py` — superseded, left unreferenced.

**3. Modeling** (`scripts/modeling/`)
- `decision_tree_training.py` — joins matchups to rolling stats on `GAME_ID`, trains/cross-validates 6 classifiers, saves the best. See `TRAINING.md` for the full walkthrough.
- `data_exploration.py` — ad hoc, not part of the pipeline.

## Serving

**4. Batch** (`serving/batch/run_nightly_predictions.py`)
- Fetches the day's schedule (`ScheduleLeagueV2`), looks up each team's current-form snapshot, fetches live injury counts, predicts, writes `NBAdata/predictions/predictions_<date>.csv`.
- `B2B` is computed dynamically from the snapshot's `GAME_DATE` vs. the target date — so `--date <past>` reflects back-to-back status as of that date, but the other 10 rolling stats still reflect the snapshot's *latest* refresh (see "Known issues"). `PlayersOut` is always fetched relative to **real wall-clock time**, not `--date` — a past-date backtest run always sees *today's* injury report (or none, off-season), never that date's actual one.
- `fetch_latest_injury_counts()` (`serving/inference/injury_report.py`) parses the NBA's live injury-report PDF with `pdfplumber` — pure Python, no Java. On any failure (network, no report found, parse error) it logs and returns `{}`, degrading `PlayersOut` to 0 rather than breaking the run.
- Run: `python serving/batch/run_nightly_predictions.py [--date YYYY-MM-DD]`
- Docker: code-only image, `NBAdata/` mounted at runtime (`MSYS_NO_PATHCONV=1` on Windows Git Bash).

**5. Real-time API** (`serving/api/`)
- `POST /predict {home_team, away_team, date?}` → winner + probability. `GET /health`. Web UI at `/`.
- Errors: unknown team/no stats → 404; missing feature → 422; not provisioned / model not loaded → 503.
- Run: `uvicorn main:app --app-dir serving/api --reload`
- Docker: self-contained (bakes in model + snapshot, both git-tracked) — no mount or `dvc pull` needed.

## Testing

`tests/` covers pure logic (feature resolution, date parsing, stats lookup,
model-loading fallback, rolling-window computation) and the API (mocked,
offline). Key regression guard: `test_build_rolling_team_stats.py`'s leakage
test, asserting a game's own stats never appear in its own rolling average.
Pkl/pure-logic tests run offline; data-dependent tests need `dvc pull`.
`scripts/data_pull/*` and the batch job's live schedule fetch aren't covered
(need `stats.nba.com`, which blocks CI runners).

```
pip install -r requirements.txt
dvc pull        # needed for data-dependent tests only
python -m pytest
```

CI (`.github/workflows/ci.yml`): `dvc pull` + pytest + `docker build` on every push/PR to `main`.

## Known issues

- **Data leakage (fixed 2026-07-15):** `PTS`/`PLUS_MINUS` were pulled from completed games and used as input features. Removed; `Team1Home` added as a real pregame feature.
- **Month-level leakage (fixed 2026-08-28):** stats used to be joined at month granularity, including the target game's own contribution. Fixed by the rolling-window rebuild (exact `GAME_ID` join, strictly-prior-games averages).
- **Season mislabeling (fixed 2026-07-16, path since superseded):** `merge_advanced_base_stats.py` once hardcoded `season="2019-20"`, silently dropping 5 of 6 seasons via the `(Team,Season,Month)` join. Fixed, then the whole path was superseded by the `GAME_ID` join, which has no such labeling step to break.
- **Snapshot isn't date-bounded (since 2026-08-28):** the current-form snapshot always holds each team's *most recent* stats, not "as of a specific past date." Correct for live serving; means `--date <past>` no longer doubles as point-in-time backtesting for the 10 rolling stats (`B2B` is the exception — computed relative to the given date; `PlayersOut` is a *different* exception — it's relative to real wall-clock time, not `--date`, at all).
- **Train/serve season-boundary asymmetry (intentional, since 2026-08-28):** training drops a team's under-3-game rows at a season's start; serving instead carries over that team's last valid row from the previous season, since it must return something for every team at all times.
- **`PlayersOut` has partial training coverage (since 2026-08-28):** the NBA's public injury-report archive only goes back to 2021-22, so 2019-20/2020-21 (2 of 7 training seasons) get `NaN` for this feature and are dropped from training entirely. Live serving is unaffected — it always fetches the current report regardless of season.
