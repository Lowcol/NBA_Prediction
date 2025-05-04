import pandas as pd

# Load the matchup dataset
df = pd.read_csv("NBAdata/NBA_2019_20_Matchups_WithAdvancedStats.csv")

# List of team stat columns to average (remove non-numeric or non-team-specific columns)
stat_columns = [
    'W_PCT', 'PLUS_MINUS', 'PTS', 'FG_PCT', 'AST', 'TOV', 'REB', 'STL', 'PF', 'BLK',
    'Home_Win_PCT', 'Away_Win_PCT', 'PIE', 'eFG%', 'TOV%', 'ORB%', 'FTR'
]

# Create a mapping for columns: Team1_* and Team2_*
team1_stats = df[['Team1'] + [f'Team1_{stat}' for stat in stat_columns]]
team2_stats = df[['Team2'] + [f'Team2_{stat}' for stat in stat_columns]]

# Rename columns to unify under same name
team1_stats.columns = ['Team'] + stat_columns
team2_stats.columns = ['Team'] + stat_columns

# Combine and compute mean per team
all_stats = pd.concat([team1_stats, team2_stats])
team_averages = all_stats.groupby('Team').mean().reset_index()

# Save to CSV
output_path = "NBAdata/NBA_2019_20_Season_Averages_Per_Team.csv"
team_averages.to_csv(output_path, index=False)
print(f"✅ Saved team season averages to: {output_path}")
