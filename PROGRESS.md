# Progress notes (working scratchpad)

Purpose: a running record of *how* work got done — task specs, design
decisions made along the way, things ruled out, open questions — not just
*what* changed. `log.md` is the terse changelog; this is the reasoning
behind it. Read this first when resuming work after a context gap.

---

## Active task: Phase 1 — batch prediction job (`ARCHITECTURE.md`)

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
