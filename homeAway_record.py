import pandas as pd

# Load the original merged team game data (not matchups)
df = pd.read_csv('NBAdata/NBA_Team_Boxscores_2019_20.csv')

# Add HOME flag
df['HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)

# Add binary WIN flag
df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Group home and away separately
home_records = df[df['HOME'] == 1].groupby('TEAM_NAME')['WIN'].agg(['sum', 'count'])
home_records.columns = ['Home_Wins', 'Home_Games']
home_records['Home_Win_PCT'] = home_records['Home_Wins'] / home_records['Home_Games']

away_records = df[df['HOME'] == 0].groupby('TEAM_NAME')['WIN'].agg(['sum', 'count'])
away_records.columns = ['Away_Wins', 'Away_Games']
away_records['Away_Win_PCT'] = away_records['Away_Wins'] / away_records['Away_Games']

# Merge both records into one table
team_records = pd.merge(home_records, away_records, left_index=True, right_index=True).reset_index()

# Preview
print(team_records.head())
