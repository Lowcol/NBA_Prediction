# Project log

Snapshot of where the project stands, updated as major changes land. Not a full changelog — see git history for that.

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
