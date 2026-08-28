import numpy as np
import pandas as pd

from build_rolling_team_stats import (
    MIN_GAMES_IN_WINDOW,
    ROLLING_WINDOW,
    compute_rolling_stats,
)


def _make_log(n_games, team="denver nuggets", season_type="Regular Season", start_id=1):
    """Build a synthetic per-team game log with recognizable, distinct stat
    values: game i (0-indexed) gets PIE = 100 + i so each game's own value
    is unmistakable in an assertion."""
    dates = pd.date_range("2024-10-01", periods=n_games, freq="2D")
    return pd.DataFrame({
        "TEAM_NAME": [team] * n_games,
        "GAME_ID": list(range(start_id, start_id + n_games)),
        "GAME_DATE": dates.strftime("%Y-%m-%d"),
        "WL": ["W" if i % 2 == 0 else "L" for i in range(n_games)],
        "SeasonType": [season_type] * n_games,
        "Season": ["2024-25"] * n_games,
        "PIE": [100.0 + i for i in range(n_games)],
        "EFG_PCT": [200.0 + i for i in range(n_games)],
        "TM_TOV_PCT": [300.0 + i for i in range(n_games)],
        "OREB_PCT": [400.0 + i for i in range(n_games)],
        "FT_PCT": [500.0 + i for i in range(n_games)],
        "NET_RATING": [600.0 + i for i in range(n_games)],
        "OFF_RATING": [700.0 + i for i in range(n_games)],
        "DEF_RATING": [800.0 + i for i in range(n_games)],
        "PACE": [900.0 + i for i in range(n_games)],
    })


def test_rolling_average_never_includes_current_game_own_value():
    # Regression guard: this is the whole point of the rebuild. A plain
    # .rolling(N).mean() (no shift) would include the current row.
    df = _make_log(15)
    result = compute_rolling_stats(df)
    result = result.sort_values("GAME_ID").reset_index(drop=True)

    for i, row in result.iterrows():
        # game i's own PIE value is 100 + i; it must never appear as part of
        # what should be a mean of *earlier* games only.
        own_value = 100.0 + i
        if row["games_in_window"] == 0:
            assert pd.isna(row["PIE"])
            continue
        window_start = max(0, i - ROLLING_WINDOW)
        expected_values = [100.0 + j for j in range(window_start, i)]
        expected_mean = np.mean(expected_values) if expected_values else np.nan
        if pd.isna(row["PIE"]):
            # below MIN_GAMES_IN_WINDOW
            assert len(expected_values) < MIN_GAMES_IN_WINDOW
        else:
            assert row["PIE"] == expected_mean
            assert own_value not in expected_values


def test_games_in_window_climbs_and_caps_at_window_size():
    df = _make_log(15)
    result = compute_rolling_stats(df).sort_values("GAME_ID").reset_index(drop=True)

    expected = [min(i, ROLLING_WINDOW) for i in range(15)]
    assert result["games_in_window"].tolist() == expected


def test_min_games_in_window_produces_nan_when_history_too_short():
    df = _make_log(MIN_GAMES_IN_WINDOW)  # exactly at the boundary: last row has
    # MIN_GAMES_IN_WINDOW - 1 prior games, which is one short
    result = compute_rolling_stats(df).sort_values("GAME_ID").reset_index(drop=True)

    # first MIN_GAMES_IN_WINDOW rows all have fewer than MIN_GAMES_IN_WINDOW
    # prior games (0, 1, ..., MIN_GAMES_IN_WINDOW - 1), so all should be NaN.
    assert result["PIE"].isna().all()

    # one more game gives it exactly MIN_GAMES_IN_WINDOW prior games -> valid
    df_extra = _make_log(MIN_GAMES_IN_WINDOW + 1)
    result_extra = compute_rolling_stats(df_extra).sort_values("GAME_ID").reset_index(drop=True)
    assert not pd.isna(result_extra["PIE"].iloc[-1])
    assert result_extra["games_in_window"].iloc[-1] == MIN_GAMES_IN_WINDOW


def test_preseason_games_excluded_from_window_and_output():
    preseason = _make_log(5, season_type="Pre Season", start_id=1)
    regular = _make_log(5, season_type="Regular Season", start_id=101)
    # Make regular season games occur strictly after preseason games so a
    # naive implementation would fold preseason games into the window.
    regular["GAME_DATE"] = pd.date_range("2024-11-01", periods=5, freq="2D").strftime("%Y-%m-%d")
    # give preseason games an easily-recognizable, wildly-different value
    preseason["PIE"] = [-999.0] * 5

    df = pd.concat([preseason, regular], ignore_index=True)
    result = compute_rolling_stats(df).sort_values("GAME_ID").reset_index(drop=True)

    # Pre Season rows must not appear in the output at all.
    assert set(result["GAME_ID"]) == set(regular["GAME_ID"])

    # Pre Season values must never leak into the rolling averages: the first
    # regular season game has zero valid prior (non-preseason) games.
    first_row = result.iloc[0]
    assert pd.isna(first_row["PIE"])
    assert first_row["games_in_window"] == 0

    # None of the rolling PIE values should ever equal a value derived from
    # the -999 preseason PIE.
    assert (result["PIE"].dropna() > 0).all()


def test_rolling_stats_separate_per_team():
    team_a = _make_log(12, team="denver nuggets", start_id=1)
    team_b = _make_log(12, team="boston celtics", start_id=101)
    # give team_b distinct values so cross-team leakage would be detectable
    team_b["PIE"] = [1000.0 + i for i in range(12)]

    df = pd.concat([team_a, team_b], ignore_index=True)
    result = compute_rolling_stats(df)

    a_last = result[result["TEAM_NAME"] == "denver nuggets"].sort_values("GAME_ID").iloc[-1]
    b_last = result[result["TEAM_NAME"] == "boston celtics"].sort_values("GAME_ID").iloc[-1]

    assert a_last["PIE"] < 200
    assert b_last["PIE"] > 1000


def test_output_columns_match_features_stat_map_expectations():
    df = _make_log(15)
    result = compute_rolling_stats(df)

    expected_cols = {
        "TEAM_NAME", "GAME_ID", "GAME_DATE", "Season", "games_in_window",
        "W_PCT", "PIE", "EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "FT_PCT",
        "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE",
    }
    assert set(result.columns) == expected_cols
