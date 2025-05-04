# import pandas as pd

# # === Load the data ===
# games = pd.read_csv('NBAdata/NBA_Team_Boxscores_2019_20.csv')
# monthly_stats = pd.read_csv('NBAdata/monthly_stats/NBA_Team_Monthly_Stats_2019_20.csv')

# # === Prepare games ===
# games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
# games['MONTH'] = games['GAME_DATE'].dt.month
# games['SEASON'] = '2019-20'

# # === Rename monthly stats to prevent name conflicts ===
# monthly = monthly_stats.rename(columns=lambda col: f'STATS_{col}' if col not in ['TEAM_ID', 'SEASON', 'MONTH'] else col)

# # === Filter only valid months ===
# valid_months = monthly['MONTH'].unique()
# games = games[games['MONTH'].isin(valid_months)]

# # === Merge team stats into game data ===
# merged = pd.merge(
#     games,
#     monthly,
#     on=['TEAM_ID', 'SEASON', 'MONTH'],
#     how='left'
# )

# # === Convert team rows into matchup rows ===
# matchup_rows = []
# for game_id, group in merged.groupby("GAME_ID"):
#     if len(group) != 2:
#         continue  # skip incomplete games

#     team1 = group.iloc[0]
#     team2 = group.iloc[1]

#     # Make sure team1 is the home team
#     if '@' in team1['MATCHUP']:
#         team1, team2 = team2, team1

#     row = {
#         "Team1": team1["TEAM_NAME"],
#         "Team2": team2["TEAM_NAME"],
#         "Team1Score": team1["PTS"],
#         "Team2Score": team2["PTS"],
#         "Team1Home": 1,
#         "Team1Win": 1 if team1["PTS"] > team2["PTS"] else 0,
#     }

#     # Team1 stats
#     for col in team1.index:
#         if col.startswith("STATS_"):
#             stat = col.replace("STATS_", "")
#             row[f"Team1_{stat}"] = team1[col]

#     # Team2 stats
#     for col in team2.index:
#         if col.startswith("STATS_"):
#             stat = col.replace("STATS_", "")
#             row[f"Team2_{stat}"] = team2[col]

#     matchup_rows.append(row)

# # === Create and save the final matchup dataset ===
# matchups = pd.DataFrame(matchup_rows)
# matchups.to_csv('NBAdata/merged_gameStats_monthStats/NBA_2019_20_Matchups_TrainingReady.csv', index=False)
# print(f"✅ Matchup dataset saved with {matchups.shape[0]} games.")


import pandas as pd

# === Step 1: Load original game-by-team file ===
df_games = pd.read_csv('NBAdata/NBA_Team_Boxscores_2019_20.csv')

# Add flag: 1 if home, 0 if away
df_games['HOME'] = df_games['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)

# Add binary WIN column
df_games['WIN'] = df_games['WL'].apply(lambda x: 1 if x == 'W' else 0)

# === Step 2: Compute home win % ===
home_stats = (
    df_games[df_games['HOME'] == 1]
    .groupby('TEAM_NAME')['WIN']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'Home_Wins', 'count': 'Home_Games'})
)
home_stats['Home_Win_PCT'] = home_stats['Home_Wins'] / home_stats['Home_Games']

# === Step 3: Compute away win % ===
away_stats = (
    df_games[df_games['HOME'] == 0]
    .groupby('TEAM_NAME')['WIN']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'Away_Wins', 'count': 'Away_Games'})
)
away_stats['Away_Win_PCT'] = away_stats['Away_Wins'] / away_stats['Away_Games']

# === Step 4: Combine into single DataFrame ===
team_home_away_pct = pd.merge(home_stats, away_stats, left_index=True, right_index=True).reset_index()

# === Step 5: Load your matchup file ===
matchups = pd.read_csv('NBAdata/merged_gameStats_monthStats/NBA_2019_20_Matchups.csv')

# === Step 6: Merge Team1 records ===
matchups = matchups.merge(
    team_home_away_pct,
    left_on='Team1',
    right_on='TEAM_NAME',
    how='left'
).rename(columns={
    'Home_Win_PCT': 'Team1_Home_Win_PCT',
    'Away_Win_PCT': 'Team1_Away_Win_PCT'
}).drop(columns=['TEAM_NAME', 'Home_Wins', 'Home_Games', 'Away_Wins', 'Away_Games'])

# === Step 7: Merge Team2 records ===
matchups = matchups.merge(
    team_home_away_pct,
    left_on='Team2',
    right_on='TEAM_NAME',
    how='left'
).rename(columns={
    'Home_Win_PCT': 'Team2_Home_Win_PCT',
    'Away_Win_PCT': 'Team2_Away_Win_PCT'
}).drop(columns=['TEAM_NAME', 'Home_Wins', 'Home_Games', 'Away_Wins', 'Away_Games'])

# === Step 8: Save the updated matchups file ===
matchups.to_csv('NBAdata/NBA_2019_20_Matchups_WithHomeAwayWinPct.csv', index=False)
print("✅ Saved: NBA_2019_20_Matchups_WithHomeAwayWinPct.csv")
