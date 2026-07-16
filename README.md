# NBA Prediction

A machine learning pipeline that predicts the winner of upcoming NBA games from team-level stats.

## Repo structure

```
scripts/
  data_pull/     # Hits the NBA stats API (via nba_api) and writes raw CSVs into NBAdata/
  data_prep/     # Merges/reshapes raw pulls into the monthly team-stat tables used for training
  modeling/      # Trains and evaluates the prediction model

NBAdata/
  matchups/                        # One row per game, per season (2019-20 .. 2024-25)
  monthly_stats/                   # Current-season (2024-25) base/advanced/combined team stats by month
  archive/historical/monthly_stats/# Same, for past seasons (2019-20 .. 2023-24)
  NBA_Training_Matchups_2019_2025.csv  # Combined training set built by decision_tree_training.py
  best_model.pkl, scaler.pkl       # Latest trained model + the StandardScaler used with it
```

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

### Running it end to end
Dependencies are pinned in `requirements.txt` — install them first:
```
pip install -r requirements.txt
```

Then run the pipeline:
```
python scripts/data_pull/nbaPull_19-25_matchups.py   # only needed to refresh matchup data
python scripts/modeling/decision_tree_training.py    # builds the training set, trains, saves best_model.pkl
```

## Known issues

### Data leakage (fixed, 2026-07-15)

The original `nbaPull_19-25_matchups.py` pulled `PTS` and `PLUS_MINUS` from the **completed** game itself, and the label `Team1Win` was derived from `Team1_PTS > Team2_PTS`. Those same leaked columns were then used as **input features** in training — the model was effectively being handed the final score to predict who won, which would have made any reported accuracy meaningless. This has been fixed: those columns are no longer used as features, and `Team1Home` (previously missing entirely — `Team1`/`Team2` were assigned by alphabetical sort order, not home/away) was added as a legitimate pregame feature instead.

### Residual, smaller leakage (open)

Team stats are joined to each matchup at **month granularity** (`merge_advanced_base_stats.py` / the merge logic in `decision_tree_training.py`): a game played in November is joined to November's aggregate stats, which include that same game's contribution to the month's totals. This is much less severe than the original bug (the aggregate is diluted across ~15 games instead of being the game's own score) but is not fully pregame-safe. Fixing this properly would mean rebuilding team stats as trailing rolling averages computed strictly from games before each matchup's date, which is a bigger change than the minimal fix applied so far.
