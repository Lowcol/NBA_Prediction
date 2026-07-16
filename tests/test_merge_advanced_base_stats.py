import pandas as pd

from merge_advanced_base_stats import merge_season


def test_merge_season_uses_season_from_input_data_not_hardcoded(tmp_path):
    # Regression guard: merge_advanced_base_stats.py used to hardcode
    # season = "2019-20" for every output file regardless of the actual
    # input season, silently corrupting every other season's stats.
    base_path = tmp_path / "base.csv"
    adv_path = tmp_path / "adv.csv"
    output_path = tmp_path / "combined.csv"

    pd.DataFrame({
        "TEAM_NAME": ["Denver Nuggets"],
        "Season": ["2024-25"],
        "Month": [11],
        "W_PCT_base": [0.7],
    }).to_csv(base_path, index=False)

    pd.DataFrame({
        "TEAM_NAME": ["Denver Nuggets"],
        "Season": ["2024-25"],
        "Month": [11],
        "PIE": [0.55],
    }).to_csv(adv_path, index=False)

    merge_season(base_path, adv_path, output_path)

    result = pd.read_csv(output_path)
    assert set(result["Season"].unique()) == {"2024-25"}

    denver_row = result[(result["TEAM_NAME"] == "denver nuggets") & (result["Month"] == 11)]
    assert not denver_row.empty
    assert denver_row.iloc[0]["W_PCT_base"] == 0.7
    assert denver_row.iloc[0]["PIE"] == 0.55


def test_merge_season_output_differs_by_input_season(tmp_path):
    def make_inputs(season: str, suffix: str):
        base_path = tmp_path / f"base_{suffix}.csv"
        adv_path = tmp_path / f"adv_{suffix}.csv"
        pd.DataFrame({
            "TEAM_NAME": ["Boston Celtics"],
            "Season": [season],
            "Month": [1],
            "W_PCT_base": [0.6],
        }).to_csv(base_path, index=False)
        pd.DataFrame({
            "TEAM_NAME": ["Boston Celtics"],
            "Season": [season],
            "Month": [1],
            "PIE": [0.5],
        }).to_csv(adv_path, index=False)
        return base_path, adv_path

    base_a, adv_a = make_inputs("2019-20", "a")
    base_b, adv_b = make_inputs("2023-24", "b")
    output_a = tmp_path / "combined_a.csv"
    output_b = tmp_path / "combined_b.csv"

    merge_season(base_a, adv_a, output_a)
    merge_season(base_b, adv_b, output_b)

    assert set(pd.read_csv(output_a)["Season"].unique()) == {"2019-20"}
    assert set(pd.read_csv(output_b)["Season"].unique()) == {"2023-24"}
