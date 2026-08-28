# How the model is trained (and what it learns from)

This explains, in plain language, how the NBA win predictor is built: where the
data comes from, the exact features fed to the model, and each step of the
training process. It reflects the code in
`scripts/modeling/decision_tree_training.py` and the shared feature contract in
`scripts/modeling/features.py`.

---

## 1. What the model predicts

One thing: **given two teams in a specific game, will the home team win?**

It's a yes/no (binary) prediction. The model also returns a probability — e.g.
"home team wins with 61% confidence" — not just the winner.

---

## 2. The data

- **Source:** historical NBA games and per-game team box scores pulled from
  the NBA API, stored as CSVs under `NBAdata/`. Most of this (matchups,
  per-game team logs) is tracked with DVC, not committed to git — run
  `dvc pull` to fetch it. A small per-team "current form" snapshot
  (`NBAdata/rolling_stats/nba_team_current_rolling_stats_<season>.csv`) is
  committed straight to git instead, so serving works without a DVC pull.
- **Seasons:** seven, from **2019-20 through 2025-26**.
- **Size:** about **8,900 games** total (roughly 1,100–1,300 per season).
- **One row = one game.** Each row pairs the two teams' stats **as of that
  specific game** (see below) with the actual outcome.

**2026-08-28: stats moved from month-to-date averages to trailing 10-game
rolling averages.** Previously each team's stats for a game were "this
team's average for the calendar month the game fell in" — computed from
`NBAdata/monthly_stats/`. As of this change, each team's stats for a game
are **that team's average over its own last 10 games, using only games
strictly before this one** — computed by:
1. `scripts/data_pull/team_game_logs_pull.py` — pulls per-game team box
   scores (Base + Advanced) for every season into `NBAdata/team_game_logs/`.
2. `scripts/data_prep/build_rolling_team_stats.py` — turns that into trailing
   rolling averages per team per game (`NBAdata/rolling_stats/`), using
   `.shift(1)` before `.rolling(10)` so a game's own stats can never leak
   into its own average. A team needs at least 3 prior games before a row
   gets real numbers (fewer than that → `NaN` → dropped from training, same
   as any other missing feature).

This fixes two things that were previously open issues (see
`COMPONENTS.md`, "Known issues"): the residual month-level leakage (a game's
own box score no longer contributes to the average it's compared against),
and the lack of "recent form" signal (a team's last 10 games reflects who's
hot/cold/injured/traded much better than a whole month blended together).

The combined training table is assembled by
`build_historical_training_dataset()`, which joins each season's game list
(`NBAdata/matchups/`) to that season's rolling-stats file
(`NBAdata/rolling_stats/nba_team_rolling_stats_<season>.csv`) **on the
game's own `GAME_ID`** (an exact join — both files share the same NBA game
IDs, replacing the old fuzzy `(Team, Season, Month)` join), and saves the
result to `NBAdata/NBA_Training_Matchups_2019_2025.csv`.

### The "Team1 / Team2" convention

In the training data, the two teams in a game are labeled **Team1** and
**Team2** _alphabetically_ — that ordering carries no meaning about who is home.
Home-court is encoded separately by a single feature, `Team1Home` (1 if
Team1 is the home team, 0 otherwise).

> Note: in live serving (`serving/`), the code always assigns **Team1 = the home
> team**, so `Team1Home` is always 1 and the model's output reads directly as
> "probability the home team wins." Same model, simpler convention at prediction
> time.

---

## 3. The features (the 21 inputs)

Every prediction is based on **21 numbers**. For each team we take 10 season-level
stats, plus the one home-court flag: 10 + 10 + 1 = 21. These are defined in
`features.py` as `SELECTED_FEATURES`.

### The 10 per-team stats (each provided for Team1 and Team2)

All 10 are now **trailing 10-game averages** (see §2), not season-to-date or
month-to-date averages.

| Feature name | What it measures (plain language)                                                                                                                                   | Source column |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `W_PCT`      | **Win percentage** — the team's win rate **over its last 10 games** (not season-to-date — see the callout below).                                                  | `W_PCT`       |
| `PIE`        | **Player Impact Estimate** — the share of everything good in a game (points, rebounds, assists, steals…) the team accounts for. A single "how good overall" number. | `PIE`         |
| `eFG%`       | **Effective field-goal %** — shooting accuracy that gives extra credit for 3-pointers being worth more.                                                             | `EFG_PCT`     |
| `TOV%`       | **Turnover %** — how often possessions are lost to turnovers (lower is better).                                                                                     | `TM_TOV_PCT`  |
| `ORB%`       | **Offensive rebound %** — share of available offensive rebounds grabbed (second chances).                                                                           | `OREB_PCT`    |
| `FTR`        | **Free-throw rate** — how often the team gets to the free-throw line, relative to its shot attempts (`FTA / FGA`).                                                  | `FTR`         |
| `NetRtg`     | **Net rating** — points scored minus points allowed, per 100 possessions. The single "are they outscoring opponents, pace-adjusted" number.                        | `NET_RATING`  |
| `OffRtg`     | **Offensive rating** — points scored per 100 possessions.                                                                                                           | `OFF_RATING`  |
| `DefRtg`     | **Defensive rating** — points allowed per 100 possessions (lower is better).                                                                                        | `DEF_RATING`  |
| `Pace`       | **Pace** — possessions per 48 minutes, i.e. how fast the team plays.                                                                                                | `PACE`        |

The four shooting/possession stats (`eFG%`, `TOV%`, `ORB%`, `FTR`) are the
well-known **"Four Factors"** of basketball — the aspects of play most tied to
winning. `W_PCT` and `PIE` add overall quality. `NetRtg`, `OffRtg`, `DefRtg`,
and `Pace` were added on 2026-08-28 to test whether pace/efficiency stats carry
signal the Four Factors + PIE don't; `NetRtg` is exactly `OffRtg − DefRtg`, so
it's redundant with the other two, but kept anyway since it's free (already in
the pulled data) and tree-based models handle redundant features fine. See §5
for whether this actually helped.

> **`W_PCT` semantics changed on 2026-08-28.** Before the rolling-window
> rebuild (§2), `W_PCT` was season-cumulative win percentage. It's now a
> trailing win rate over the last 10 games instead — same column name,
> different meaning. Noted here so it doesn't read as a bug later.

> Implementation detail: the "source column" names above can vary slightly
> between stat files (e.g. `W_PCT` vs `W_PCT_base`). `resolve_stat_columns()` in
> `features.py` maps each model-facing name to whichever column actually exists
> in a given file, so the contract stays stable even if the raw data's column
> names differ.
>
> **`FTR` fixed on 2026-08-28.** Previously sourced from `FT_PCT`
> (free-throw shooting *percentage*, `FTM/FTA`) as an approximation. Now
> computed as true free-throw *rate*, `FTA/FGA` (how often the team gets to
> the line at all, per shot attempt) — a genuinely different stat, one of
> basketball's actual "Four Factors." Computed per game in
> `build_rolling_team_stats.py` from the raw `FTA`/`FGA` columns (already
> pulled per-game for the rolling-window rebuild), then rolled the same way
> as the other 9 stats. Done as its own isolated retrain, right after the
> rolling-window rebuild landed — see §5 for the result.

### The 1 game-context feature

| Feature name | Meaning                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------- |
| `Team1Home`  | 1 if Team1 is playing at home, else 0. This is how home-court advantage enters the model. |

### Full list (as fed to the model, in order)

```
Team1_W_PCT, Team2_W_PCT, Team1Home,
Team1_PIE, Team1_eFG%, Team1_TOV%, Team1_ORB%, Team1_FTR, Team1_NetRtg, Team1_OffRtg, Team1_DefRtg, Team1_Pace,
Team2_PIE, Team2_eFG%, Team2_TOV%, Team2_ORB%, Team2_FTR, Team2_NetRtg, Team2_OffRtg, Team2_DefRtg, Team2_Pace
```

**What's deliberately NOT included:** anything only known _after_ the game
(final score, plus/minus, points scored). Including those would be "leakage" —
the model would cheat by peeking at the result. The test
`test_selected_features_excludes_postgame_leakage` guards against them sneaking
back in.

---

## 4. The training process, step by step

All of this runs when you execute:

```
python scripts/modeling/decision_tree_training.py
```

### Step 1 — Build & clean the dataset

Join games to each team's trailing rolling stats for all seven seasons (§2),
on the game's own `GAME_ID`, then **drop any row missing one of the 21
features or the outcome** (`dropna`). Most drops now happen at the very start
of each team's season, before it has 3 prior games to average — roughly 2-4%
of rows per season, down from a much higher month-granularity drop rate. The
prediction target is `Team1Win` (1 = Team1 won).

### Step 2 — Split into train and test

Hold out **20%** of games as a **test set** the model never sees during
training, keeping the other **80%** for training. The split is _stratified_
(keeps the same win/loss ratio in both halves) and seeded (`random_state=42`) so
it's reproducible.

### Step 3 — Scale the features

Fit a `StandardScaler` on the training data — it rescales every feature to a
comparable range (mean 0, spread 1). Some models (especially the SVM) need this
to work well. The scaler is **fit on the training data only**, then applied to
the test data, so no test information leaks into training.

### Step 4 — Try 6 models, each with hyperparameter tuning

Six different algorithms compete. For each one, we don't just train it once — we
try several **hyperparameter** settings (a model's configuration knobs, set
before training) and keep the best. This is done with `GridSearchCV`:
exhaustively try every combination in a small grid, scoring each by **5-fold
cross-validation** (split the training data into 5 parts, train on 4, test on
the 5th, rotate, average — a robust accuracy estimate).

| Model               | What it is                                                   | Tuned settings (grid)                                                        |
| ------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Logistic Regression | A weighted linear formula → probability.                     | `C` (regularization strength): 0.01, 0.1, 1, 10                              |
| Decision Tree       | A flowchart of yes/no splits.                                | `max_depth`: 3, 5, 8 · `min_samples_leaf`: 1, 20, 50                         |
| Random Forest       | Many decision trees averaged.                                | `max_depth`: 5, 10, none · `min_samples_leaf`: 1, 10                         |
| XGBoost             | Trees built in sequence, each fixing the last's mistakes.    | `n_estimators`: 100, 300 · `learning_rate`: 0.03, 0.1 · `max_depth`: 2, 3, 6 |
| SVC (RBF)           | Support-vector machine — finds a smooth separating boundary. | `C`: 0.1, 1, 10 · `gamma`: scale, 0.01, 0.1                                  |
| Bagging SVC         | 10 SVMs trained on random data samples, averaged.            | inner `C`: 0.1, 1, 10 · inner `gamma`: scale, 0.1                            |

Each model's best score and winning settings are logged to **MLflow** (view them
with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

### Step 5 — Pick the winner and refit

The model with the highest cross-validation accuracy wins. It's refit on the
full training set. (For the SVM-based models, real probability output is only
switched on here for the final winner — it's skipped during the grid search
because it's slow and doesn't affect which settings win.)

### Step 6 — Final evaluation

The winner predicts on the **held-out 20% test set** it never saw. That
`test_accuracy` is the honest estimate of real-world performance.

### Step 7 — Save & register

- Save `NBAdata/best_model.pkl` and `NBAdata/scaler.pkl` (the serving fallback).
- Package scaler + model as one **Pipeline** (so scaling can't drift from the
  model) and register it in the **MLflow model registry** as `nba-win-predictor`,
  with the alias `@production` — which is exactly what the live API loads.

---

## 5. How to read the accuracy

Accuracy only means something compared to a baseline. The script logs two:

- **Majority-class baseline (~0.53):** always guess the more common outcome.
- **Home-team-always-wins baseline (~0.55):** always pick the home team.

Historically (month-to-date averages) the trained models landed around
**0.60–0.61** — beating "just pick the home team" by roughly 5 points.
**As of the 2026-08-28 rolling-window rebuild (§2), that changed for real:**

> **2026-08-28 — rolling-window rebuild.** Replacing month-to-date averages
> with trailing 10-game rolling averages moved the winning model from
> **Random Forest (0.615 cross-val / 0.598 test)** to **Logistic Regression
> (0.625 cross-val / 0.631 test)** — a genuine, above-noise-band improvement
> (+1.0pt cross-val, +3.3pt test), not the wash the `NetRtg`/`Pace` addition
> was two commits earlier. The gap over the home-court baseline (0.554) is now
> roughly **7.7 points**, up from ~4.6. This confirms the diagnosis in §6: the
> ceiling was about **time resolution**, not stat variety or model choice —
> the same 4 stat-variety experiment (NetRtg/OffRtg/DefRtg/Pace) that did
> nothing at month granularity is now part of a feature set that measurably
> works once it reflects recent form instead of a month blend.
>
> Logistic Regression winning this run — after tree-based models (Random
> Forest, Bagging SVC, XGBoost) had traded the top spot across recent
> retrains — is worth watching on future retrains rather than reading too
> much into from one run; per §6, the models have always clustered tightly
> enough that the specific winner isn't very meaningful on its own.
>
> **2026-08-28 — FTR fix (isolated retrain, right after the above).** Fixing
> `FTR` to be true `FTA/FGA` instead of the `FT_PCT` proxy (§3) moved the
> winner to **SVC-RBF (0.628 cross-val / 0.626 test)** — cross-val ticked up
> from 0.625 to 0.628, test ticked *down* from 0.631 to 0.626, both inside
> the noise band. Net effect: another wash, like the `NetRtg`/`Pace`
> addition, not like the rolling-window rebuild. Worth doing anyway — it's a
> real correctness fix (the old proxy measured something different from what
> the feature name claimed), and correctness fixes don't need to pay for
> themselves in accuracy — but don't expect it to move the number.
>
> Before this: all six models clustered near 0.60 — a sign of an information
> ceiling in the *features*, not a weakness of any one algorithm. That
> reasoning is what motivated the rebuild; see §6.

---

## 6. How to improve accuracy

**Start from the diagnosis.** All six models — from a simple linear one to
gradient-boosted trees — cluster around **0.60**, within about a point of each
other. When wildly different algorithms all land in the same place, the limit
isn't the algorithm or the amount of data: it's that **the features don't
contain much more signal**. So the biggest wins come from giving the model
_better information_, not a fancier model. The levers below are ordered roughly
by expected payoff.

### Lever 1 — Richer features (by far the biggest opportunity)

> ✅ **Done (2026-08-28): recent form via rolling windows.** Each team's 10
> stats are now trailing 10-game averages instead of month-to-date (§2) — see
> §5 for the accuracy result. The items below are what's still open.

Real games turn on other things the current 10 stats still miss:

- **Rest & schedule fatigue:** days of rest, back-to-back games, games in the
  last 7 days, travel distance / time-zone changes. Tired teams underperform —
  this is well-known predictive signal we currently ignore entirely.
- **Player availability / injuries:** is a star playing tonight? Even a simple
  "is the top-minutes player active" flag can matter a lot.
- **Opponent-adjusted stats & strength of schedule:** a 55% win rate against
  weak teams isn't the same as against strong ones. Adjusting each team's stats
  for who they played removes a big confounder.
- **Home/away splits:** replace the single `Team1Home` flag with each team's
  _actual_ home-vs-road performance — some teams are far better at home than
  others.
- **Head-to-head & style matchups:** recent results between these two teams, and
  pace/style mismatches (e.g. a fast team vs a slow one).

### Lever 2 — Better use of the features we already have

- **Differences instead of pairs:** feed `Team1_stat − Team2_stat` rather than
  both separately. It's the _gap_ between teams that predicts the winner, and
  making that explicit especially helps the simpler models.
- **League-relative stats:** express each stat relative to the league average
  that season, so a number means the same thing across eras/pace environments.

### Lever 3 — Make the evaluation honest (may _lower_ the number, but you can trust it)

- **Chronological split, not random.** We currently hold out a _random_ 20% of
  games. For something that plays out over time, the honest test is **train on
  earlier games, test on later ones** — that's how the model is actually used
  (predicting future games). A random split can leak subtle future information
  and flatter the score. **Diagnosed, not yet the default:** a one-off
  chronological-split test (train on the earlier ~80% of dates, test on the
  rest) was run against the pre-rebuild model and came back *slightly higher*
  than the random split (61.4% vs. 60.0%) — so the random split wasn't
  flattering the score at the time. Worth re-running against the current
  rolling-window model to confirm that still holds, but it's not urgent.
- **Watch early-season noise.** Partially addressed by the rolling-window
  rebuild's `MIN_GAMES_IN_WINDOW=3` threshold, which drops a team's first
  couple of games each season rather than averaging over too little history —
  but this is a hard cutoff, not the down-weighting this lever originally
  suggested.
- ✅ **Done (2026-08-28): fix `FTR` to be true free-throw rate.** Was sourced
  from `FT_PCT` (see §3); now `FTA/FGA`. Done as its own isolated retrain —
  see §5. Result: a wash (inside the noise band), same as `NetRtg`/`Pace`
  earlier. Worth having anyway as a correctness fix, just not an accuracy one.

### Lever 4 — Model-side tweaks (smaller payoff, since the models already tie)

- **Probability calibration** (`CalibratedClassifierCV`): makes the _probabilities_
  more trustworthy (a "70%" really wins ~70% of the time). Improves probability
  quality more than raw accuracy.
- **Stacking/ensembling** the top few models, or a **wider hyperparameter search**
  (`RandomizedSearchCV` over larger ranges) — worth a try, but expect small gains
  given how tightly the models already cluster.

### A realistic ceiling

Single NBA games are genuinely close to coin flips. Even professional models that
use betting-market data top out around **65–70%**. Landing at 0.63 on rolling
team box-score averages (up from 0.61 pre-rebuild) is a solid result — and the
honest way to judge any future change is against the **home-court baseline
(0.554)**, using the same cross-validation, remembering the score naturally
wobbles by **±1–1.5 points** run to run. Chase changes that clear that noise
band, not ones inside it.

> One deliberate exclusion: **betting odds / point spreads** would boost accuracy
> a lot, but they're essentially the market's own prediction — using them is a
> different goal (tracking the market) than predicting games from team
> performance. Decide which problem you're solving before adding them.

---

## 7. Where everything lives

| Thing                                       | Path                                                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Training script                              | `scripts/modeling/decision_tree_training.py`                                                         |
| Feature contract (shared with serving)       | `scripts/modeling/features.py`                                                                       |
| MLflow / registry settings                   | `scripts/modeling/mlflow_config.py`                                                                  |
| Per-game team box score pull                 | `scripts/data_pull/team_game_logs_pull.py`                                                           |
| Rolling-window feature builder               | `scripts/data_prep/build_rolling_team_stats.py`                                                      |
| Current-form snapshot builder (for serving)  | `scripts/data_prep/build_current_rolling_snapshot.py`                                                |
| Raw data (DVC-tracked)                       | `NBAdata/matchups/`, `NBAdata/team_game_logs/`, `NBAdata/rolling_stats/nba_team_rolling_stats_*.csv` |
| Current-form snapshot (git-tracked)          | `NBAdata/rolling_stats/nba_team_current_rolling_stats_*.csv`                                         |
| Combined training set (generated)            | `NBAdata/NBA_Training_Matchups_2019_2025.csv`                                                        |
| Saved model + scaler (serving fallback)      | `NBAdata/best_model.pkl`, `NBAdata/scaler.pkl`                                                       |
| Experiment tracking UI                       | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → [http://localhost:5000](http://localhost:5000) |

Note: `NBAdata/monthly_stats/` and `NBAdata/archive/historical/monthly_stats/`
(the old month-granularity data) still exist on disk but are no longer read by
the pipeline — left in place rather than deleted, same reasoning as leaving
`nba_api_pull.py`/`NBA_Team_Boxscores_2024_25.csv` alone when they were
superseded.
