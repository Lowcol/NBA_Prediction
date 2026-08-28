from features import STAT_MAP, SELECTED_FEATURES, resolve_stat_columns


def test_resolve_stat_columns_prefers_first_available_option():
    columns = ["TEAM_NAME", "Season", "Month", "W_PCT_base", "PIE", "EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "FTR"]
    resolved_map, selected_cols = resolve_stat_columns(columns)

    assert resolved_map["W_PCT"] == "W_PCT_base"
    assert resolved_map["PIE"] == "PIE"
    assert {"TEAM_NAME", "Season", "Month"}.issubset(selected_cols)


def test_resolve_stat_columns_falls_back_to_second_option():
    columns = ["TEAM_NAME", "Season", "Month", "W_PCT", "PIE"]
    resolved_map, _ = resolve_stat_columns(columns)

    assert resolved_map["W_PCT"] == "W_PCT"


def test_resolve_stat_columns_skips_stats_with_no_matching_column():
    columns = ["TEAM_NAME", "Season", "Month", "PIE"]
    resolved_map, selected_cols = resolve_stat_columns(columns)

    assert "PIE" in resolved_map
    assert "W_PCT" not in resolved_map
    assert "W_PCT_base" not in selected_cols


def test_selected_features_excludes_postgame_leakage():
    # Regression guard: the model was once trained directly on the completed
    # game's own score. These substrings must never reappear in the feature list.
    leaked_substrings = ["PTS", "PLUS_MINUS"]
    for feature in SELECTED_FEATURES:
        for leaked in leaked_substrings:
            assert leaked not in feature, (
                f"{feature!r} looks like a post-game outcome column and must not be a training feature"
            )


def test_selected_features_map_to_known_stats_or_home():
    # PlayersOut is a separately-computed feature (like injury availability
    # counts joined in decision_tree_training.py, not a rolling-stats/snapshot
    # column resolved via STAT_MAP) -- it's expected to be exempt here.
    for feature in SELECTED_FEATURES:
        if feature == "Team1Home":
            continue
        if feature.endswith("_PlayersOut"):
            continue
        assert feature.startswith("Team1_") or feature.startswith("Team2_")
        stat_name = feature.split("_", 1)[1]
        assert stat_name in STAT_MAP, f"{feature!r} doesn't map to a known stat in STAT_MAP"
