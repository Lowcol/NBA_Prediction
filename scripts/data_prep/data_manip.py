import pandas as pd

# Load the datasets
per_game = pd.read_csv('NBAdata/team_stats/PerGameStats.csv')
shooting = pd.read_csv('NBAdata/team_stats/ShootingStats.csv')
totals = pd.read_csv('NBAdata/team_stats/TotalStats.csv')
advanced = pd.read_csv('NBAdata/team_stats/AdvancedStats.csv')
expanded_standings = pd.read_csv('NBAdata/team_stats/ExpandedStandings.csv')
per100 = pd.read_csv('NBAdata/team_stats/Per100PossStats.csv')

# First, clean the 'Team' names
def clean_team_names(df):
    df['Team'] = df['Team'].str.replace('*', '', regex=False).str.strip()
    return df

# Clean each dataframe
per_game = clean_team_names(per_game)
shooting = clean_team_names(shooting)
totals = clean_team_names(totals)
advanced = clean_team_names(advanced)
expanded_standings = clean_team_names(expanded_standings)
per100 = clean_team_names(per100)

# Drop useless or repeated columns before merging
drop_columns = ['Rk', 'G', 'MP']

for df in [per_game, shooting, totals, advanced, expanded_standings, per100]:
    for col in drop_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            
            # Rename columns before merging to avoid conflicts
per_game = per_game.rename(columns={
    'FG': 'FG_per_game',
    'FGA': 'FGA_per_game',
    'FG%': 'FG%_per_game',
    '3P': '3P_per_game',
    '3PA': '3PA_per_game',
    '3P%': '3P%_per_game',
    '2P': '2P_per_game',
    '2PA': '2PA_per_game',
    '2P%': '2P%_per_game',
})

totals = totals.rename(columns={
    'FG': 'FG_totals',
    'FGA': 'FGA_totals',
    'FG%': 'FG%_totals',
    '3P': '3P_totals',
    '3PA': '3PA_totals',
    '3P%': '3P%_totals',
    '2P': '2P_totals',
    '2PA': '2PA_totals',
    '2P%': '2P%_totals',
})

per100 = per100.rename(columns={
    'FG': 'FG_per100',
    'FGA': 'FGA_per100',
    'FG%': 'FG%_per100',
    '3P': '3P_per100',
    '3PA': '3PA_per100',
    '3P%': '3P%_per100',
    '2P': '2P_per100',
    '2PA': '2PA_per100',
    '2P%': '2P%_per100',
})

# Merge datasets
merged = per_game.merge(shooting, on="Team", how="inner") \
                 .merge(totals, on="Team", how="inner") \
                 .merge(advanced, on="Team", how="inner") \
                 .merge(expanded_standings, on="Team", how="inner") \
                 .merge(per100, on="Team", how="inner")

# Final check
print(merged.shape)
merged.head()
# Save the merged dataframe to a CSV file
merged.to_csv('NBAdata/team_stats/merged_team_stats.csv', index=False)
