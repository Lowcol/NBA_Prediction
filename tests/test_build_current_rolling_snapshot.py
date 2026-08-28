import pandas as pd

from build_current_rolling_snapshot import build_snapshot

STAT_COLS = [
    "W_PCT", "PIE", "EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "FTR",
    "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE",
]


def make_stat_row(**overrides):
    row = {col: 0.5 for col in STAT_COLS}
    row.update(overrides)
    return row


def write_rolling_file(tmp_path, monkeypatch, season_key, rows):
    import build_current_rolling_snapshot as mod

    monkeypatch.setattr(mod, "ROLLING_DIR", tmp_path)
    path = tmp_path / f"nba_team_rolling_stats_{season_key}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_snapshot_picks_last_valid_row_for_normal_case(tmp_path, monkeypatch):
    rows = [
        {"TEAM_NAME": "denver nuggets", "GAME_DATE": "2024-11-01", "Season": "2024-25",
         **make_stat_row(PIE=0.55)},
        {"TEAM_NAME": "denver nuggets", "GAME_DATE": "2024-11-03", "Season": "2024-25",
         **make_stat_row(PIE=0.60)},
    ]
    write_rolling_file(tmp_path, monkeypatch, "2024_25", rows)

    result = build_snapshot("2024_25")

    row = result[result["TEAM_NAME"] == "denver nuggets"].iloc[0]
    assert row["PIE"] == 0.60
    assert row["Season"] == "2024-25"
    # GAME_DATE of the snapshotted row itself must be carried through -- it's
    # not a rolling stat, it's what serving/inference/predictor.py needs to
    # compute RestDays/B2B dynamically at prediction time.
    assert row["GAME_DATE"] == "2024-11-03"


def test_build_snapshot_carries_over_from_previous_season_when_no_valid_rows(tmp_path, monkeypatch):
    # Target season: only early games, none with a full-enough window yet (NaN stats).
    target_rows = [
        {"TEAM_NAME": "denver nuggets", "GAME_DATE": "2025-10-23", "Season": "2025-26",
         **{col: None for col in STAT_COLS}},
    ]
    write_rolling_file(tmp_path, monkeypatch, "2025_26", target_rows)

    prev_rows = [
        {"TEAM_NAME": "denver nuggets", "GAME_DATE": "2025-04-10", "Season": "2024-25",
         **make_stat_row(PIE=0.58)},
    ]
    prev_path = tmp_path / "nba_team_rolling_stats_2024_25.csv"
    pd.DataFrame(prev_rows).to_csv(prev_path, index=False)

    result = build_snapshot("2025_26")

    row = result[result["TEAM_NAME"] == "denver nuggets"].iloc[0]
    assert row["PIE"] == 0.58
    assert row["Season"] == "2024-25"


def test_build_snapshot_skips_earlier_nan_rows_and_picks_last_non_nan(tmp_path, monkeypatch):
    rows = [
        {"TEAM_NAME": "boston celtics", "GAME_DATE": "2024-10-23", "Season": "2024-25",
         **{col: None for col in STAT_COLS}},
        {"TEAM_NAME": "boston celtics", "GAME_DATE": "2024-10-25", "Season": "2024-25",
         **{col: None for col in STAT_COLS}},
        {"TEAM_NAME": "boston celtics", "GAME_DATE": "2024-10-28", "Season": "2024-25",
         **make_stat_row(PIE=0.51)},
        {"TEAM_NAME": "boston celtics", "GAME_DATE": "2024-10-30", "Season": "2024-25",
         **make_stat_row(PIE=0.53)},
    ]
    write_rolling_file(tmp_path, monkeypatch, "2024_25", rows)

    result = build_snapshot("2024_25")

    row = result[result["TEAM_NAME"] == "boston celtics"].iloc[0]
    assert row["PIE"] == 0.53
