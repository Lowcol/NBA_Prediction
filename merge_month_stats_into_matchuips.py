import pandas as pd

def merge_matchup_with_stats(season: str, matchup_path: str, stats_path: str, output_path: str):
    # Load data
    matchups_df = pd.read_csv(matchup_path)
    stats_df = pd.read_csv(stats_path)

    # Normalize team names
    matchups_df['Team1'] = matchups_df['Team1'].str.strip().str.lower()
    matchups_df['Team2'] = matchups_df['Team2'].str.strip().str.lower()
    stats_df['TEAM_NAME'] = stats_df['TEAM_NAME'].str.strip().str.lower()

    # Add season and month columns
    matchups_df['Season'] = season
    matchups_df['DATE'] = pd.to_datetime(matchups_df['DATE'], errors='coerce')
    matchups_df['Month'] = matchups_df['DATE'].dt.month

    # Filter only NBA teams
    nba_teams = [team.lower() for team in [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
        "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
        "Houston Rockets", "Indiana Pacers", "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
        "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
        "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
        "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
    ]]
    matchups_df = matchups_df[
        matchups_df['Team1'].isin(nba_teams) & matchups_df['Team2'].isin(nba_teams)
    ]

    # Mapping of relevant stats
    stat_map = {
        'W_PCT_base': 'W_PCT',
        'PTS': 'PTS',
        'PLUS_MINUS': 'PLUS_MINUS',
        'PIE': 'PIE',
        'EFG_PCT': 'eFG%',
        'TM_TOV_PCT': 'TOV%',
        'OREB_PCT': 'ORB%',
        'FT_PCT': 'FTR',
        'HOME_RECORD': 'Home_Win_PCT',
        'ROAD_RECORD': 'Away_Win_PCT'
    }

    # Select available stats
    selected_cols = ['TEAM_NAME', 'Season', 'Month']
    selected_cols += [col for col in stat_map.keys() if col in stats_df.columns]

    # Prepare team stats
    team1_stats = stats_df[selected_cols].rename(columns={**{k: f"Team1_{v}" for k, v in stat_map.items()}, 'TEAM_NAME': 'Team1'})
    team2_stats = stats_df[selected_cols].rename(columns={**{k: f"Team2_{v}" for k, v in stat_map.items()}, 'TEAM_NAME': 'Team2'})

    # Merge with LEFT join to keep all NBA matchups
    merged_df = matchups_df.merge(team1_stats, on=['Team1', 'Season', 'Month'], how='left')
    merged_df = merged_df.merge(team2_stats, on=['Team2', 'Season', 'Month'], how='left')

    # Convert W-L record to win percentage
    for team in ['Team1', 'Team2']:
        for loc in ['Home', 'Away']:
            win_pct_col = f"{team}_{loc}_Win_PCT"
            record_col = f"{team}_{loc}_RECORD"
            if record_col in merged_df.columns:
                merged_df[win_pct_col] = (
                    merged_df[record_col].str.extract(r'(\d+)-(\d+)').astype(float)
                    .apply(lambda x: x[0] / (x[0] + x[1]) if x[0] + x[1] > 0 else None, axis=1)
                )
                merged_df.drop(columns=[record_col], inplace=True)

    # Save
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Merged file saved to: {output_path}")

# Example usage
merge_matchup_with_stats(
    season="2019-20",
    matchup_path="NBAdata/matchups/NBA_2019_20_Matchups_withMonth.csv",
    stats_path="NBAdata/monthly_stats/nba_team_combined_stats_2019_20.csv",
    output_path="NBAdata/merged_gameStats_monthStats/merged_2019_20_full.csv"
)
