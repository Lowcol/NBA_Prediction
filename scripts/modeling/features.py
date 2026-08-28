"""Feature contract shared between training and serving.

Keeping this in one place means the batch/real-time serving code can't
silently drift from what the model was actually trained on.
"""

STAT_MAP: dict[str, list[str]] = {
    "W_PCT": ["W_PCT_base", "W_PCT"],
    "PIE": ["PIE"],
    "eFG%": ["EFG_PCT"],
    "TOV%": ["TM_TOV_PCT"],
    "ORB%": ["OREB_PCT"],
    "FTR": ["FT_PCT"],
    "NetRtg": ["NET_RATING"],
    "OffRtg": ["OFF_RATING"],
    "DefRtg": ["DEF_RATING"],
    "Pace": ["PACE"],
}

SELECTED_FEATURES = [
    "Team1_W_PCT",
    "Team2_W_PCT",
    "Team1Home",
    "Team1_PIE",
    "Team1_eFG%",
    "Team1_TOV%",
    "Team1_ORB%",
    "Team1_FTR",
    "Team1_NetRtg",
    "Team1_OffRtg",
    "Team1_DefRtg",
    "Team1_Pace",
    "Team2_PIE",
    "Team2_eFG%",
    "Team2_TOV%",
    "Team2_ORB%",
    "Team2_FTR",
    "Team2_NetRtg",
    "Team2_OffRtg",
    "Team2_DefRtg",
    "Team2_Pace",
]


def resolve_stat_columns(stats_df_columns) -> tuple[dict[str, str], list[str]]:
    """Map model-facing stat names (e.g. "PIE") to whatever column name is
    actually present in a given monthly-stats file (e.g. "W_PCT_base" vs "W_PCT")."""
    selected_cols = ["TEAM_NAME", "Season", "Month"]
    resolved_map: dict[str, str] = {}
    for model_col, options in STAT_MAP.items():
        source = next((col for col in options if col in stats_df_columns), None)
        if source is not None:
            resolved_map[model_col] = source
            selected_cols.append(source)
    selected_cols = list(dict.fromkeys(selected_cols))
    return resolved_map, selected_cols
