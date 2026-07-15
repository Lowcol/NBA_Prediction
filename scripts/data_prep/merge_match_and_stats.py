import pandas as pd

# Load
team_stats = pd.read_csv('NBAdata/team_stats/merged_team_stats.csv')
all_games = pd.read_csv('NBAdata/2024-25_matches/all_games_cleaned.csv')

# Clean 'Team' column
team_stats['Team'] = team_stats['Team'].str.replace('*', '', regex=False).str.strip()

# Drop useless columns
columns_to_drop = [col for col in team_stats.columns if "Unnamed" in col or "Arena" in col or "Attend" in col]
columns_to_drop += ['Overall', 'Home', 'Road', 'E', 'W_y', 'A', 'C', 'SE', 'NW', 'P', 'SW', 'Pre', 'Post']
team_stats = team_stats.drop(columns=columns_to_drop)

# Now, select only numeric columns plus 'Team'
numeric_team_stats = team_stats.select_dtypes(include=[float, int]).copy()
numeric_team_stats['Team'] = team_stats['Team']  # Add back 'Team' (manually)

# Merge Home_Team stats
all_games = all_games.merge(numeric_team_stats, left_on='Home_Team', right_on='Team', how='left', suffixes=('', '_Home'))

# Merge Away_Team stats
all_games = all_games.merge(numeric_team_stats, left_on='Away_Team', right_on='Team', how='left', suffixes=('_Home', '_Away'))

# Drop extra Team columns
all_games = all_games.drop(columns=['Team_Home', 'Team_Away'])

# Final check
print(all_games.shape)
print(all_games.columns)
print(all_games.head())

# Save final merged file
all_games.to_csv('NBAdata/data_match_and_stats.csv', index=False)
