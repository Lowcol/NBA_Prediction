import pandas as pd

# Load your merged team stats
merged = pd.read_csv('NBAdata/team_stats/merged_team_stats.csv')

# Drop useless columns
columns_to_drop = [col for col in merged.columns if "Unnamed" in col or "Arena" in col or "Attend" in col]
columns_to_drop += ['Team', 'Overall', 'Home', 'Road', 'E', 'W_y', 'A', 'C', 'SE', 'NW', 'P', 'SW', 'Pre', 'Post']
merged = merged.drop(columns=columns_to_drop)

# Keep only numeric columns
merged = merged.select_dtypes(include=[float, int])

# Now define all remaining columns automatically
all_features = merged.columns.tolist()

select_features = [
    'PTS', 'ORtg', 'FG_per100', 'PTS_y', '2P%_per_game', 'SRS', '2P.1', 'TS%', 'eFG%', 'PW', 'PTS_x', 'FG_totals', '3P%_per100', 
]

# Number of matchups you want
num_matchups = 5000

# Empty list to store matchups
matchup_data = []

for _ in range(num_matchups):
    team_a = merged.sample(1)
    team_b = merged.sample(1)
    
    # (Optional safety check: if you didn't drop Team column earlier)
    # while team_a['Team'].values[0] == team_b['Team'].values[0]:
    #     team_b = merged.sample(1)

    # Get **ALL features** for both teams
    team_a_stats = team_a[select_features].values.flatten()
    team_b_stats = team_b[select_features].values.flatten()
    
    # Determine winner (higher ORtg wins for now, simple rule)
    if team_a['ORtg'].values[0] > team_b['ORtg'].values[0]:
        winner = 0  # Team A wins
    else:
        winner = 1  # Team B wins

    # Combine features
    matchup_row = list(team_a_stats) + list(team_b_stats) + [winner]
    matchup_data.append(matchup_row)

# Create column names
columns = [f"TeamA_{feat}" for feat in select_features] + \
          [f"TeamB_{feat}" for feat in select_features] + \
          ["Winner"]

# Create the DataFrame
matchups = pd.DataFrame(matchup_data, columns=columns)

# Save it
matchups.to_csv('NBAdata/team_stats/matchups_dataset.csv', index=False)

print(matchups.head())
