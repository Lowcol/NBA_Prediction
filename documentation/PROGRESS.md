# Progress notes (working scratchpad)

Purpose: a running record of *how* work got done — task specs, design
decisions made along the way, things ruled out, open questions — not just
*what* changed. `log.md` is the terse changelog; this is the reasoning
behind it. Read this first when resuming work after a context gap.

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
