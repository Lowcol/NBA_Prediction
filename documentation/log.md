# Project log

Snapshot of where the project stands, updated as major changes land. Not a full changelog — see git history for that.

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
