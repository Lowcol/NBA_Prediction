import pandas as pd

from build_injury_features import compute_injury_counts


def _row(target_game_date, game_date, team, status):
    return {
        "target_game_date": target_game_date,
        "report_timestamp": "2024-11-11 17:00:00",
        "Game Date": game_date,
        "Game Time": "07:00 (ET)",
        "Matchup": "BOS@ATL",
        "Team": team,
        "Player Name": "Some Player",
        "Current Status": status,
        "Reason": "Injury",
    }


def test_filters_rows_to_matching_game_date_only():
    # Regression guard: a single report is a rolling 1-2 day window covering
    # multiple game dates. A row whose "Game Date" differs from its own
    # "target_game_date" belongs to a different date's snapshot and must not
    # be counted toward this target date.
    df = pd.DataFrame([
        _row("2024-11-12", "11/12/2024", "atlanta hawks", "Out"),
        _row("2024-11-12", "11/13/2024", "atlanta hawks", "Out"),
    ])

    result = compute_injury_counts(df)

    assert len(result) == 1
    assert result.iloc[0]["GAME_DATE"] == "2024-11-12"
    assert result.iloc[0]["PlayersOut"] == 1


def test_counts_only_out_and_doubtful_statuses():
    statuses = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
    df = pd.DataFrame([_row("2024-11-12", "11/12/2024", "atlanta hawks", s) for s in statuses])

    result = compute_injury_counts(df)

    assert len(result) == 1
    assert result.iloc[0]["PlayersOut"] == 2  # Out + Doubtful only


def test_counts_grouped_per_team_and_date():
    df = pd.DataFrame([
        _row("2024-11-12", "11/12/2024", "atlanta hawks", "Out"),
        _row("2024-11-12", "11/12/2024", "atlanta hawks", "Doubtful"),
        _row("2024-11-12", "11/12/2024", "boston celtics", "Out"),
    ])

    result = compute_injury_counts(df).set_index("TEAM_NAME")

    assert result.loc["atlanta hawks", "PlayersOut"] == 2
    assert result.loc["boston celtics", "PlayersOut"] == 1


def test_output_columns():
    df = pd.DataFrame([_row("2024-11-12", "11/12/2024", "atlanta hawks", "Out")])
    result = compute_injury_counts(df)
    assert list(result.columns) == ["GAME_DATE", "TEAM_NAME", "PlayersOut"]
