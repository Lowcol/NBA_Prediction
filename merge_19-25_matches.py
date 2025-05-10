# import pandas as pd
# import glob
# import os

# # === Path where files are located ===
# data_dir = 'NBAdata/'  # adjust if needed
# file_pattern = os.path.join(data_dir, 'NBA_Team_Boxscores_*.csv')

# # === Read and combine all files ===
# all_games = pd.concat([pd.read_csv(file) for file in glob.glob(file_pattern)], ignore_index=True)

# print("Columns in merged data:")
# print(all_games.columns.tolist())
# exit()  # stop here after printing columns


# # === Sort and group by game ===
# # Assumes a column like 'GAME_ID' exists to group two rows per matchup
# if 'GAME_ID' not in all_games.columns:
#     raise ValueError("Missing GAME_ID column to match teams in each game.")

# matchups = []

# # Grouping each game (should be 2 rows per game)
# for game_id, group in all_games.groupby('GAME_ID'):
#     if len(group) != 2:
#         continue  # skip broken matchups

#     team1 = group.iloc[0]
#     team2 = group.iloc[1]

#     row = {
#         'Team1_W_PCT': team1['W_PCT'],
#         'Team1_PLUS_MINUS': team1['PLUS_MINUS'],
#         'Team1_PTS': team1['PTS'],
#         'Team2_W_PCT': team2['W_PCT'],
#         'Team2_PLUS_MINUS': team2['PLUS_MINUS'],
#         'Team2_PTS': team2['PTS'],
#         'Team1Home': 1 if team1['HOME'] == True else 0,
#         'Team1_Home_Win_PCT': team1['HOME_WIN_PCT'],
#         'Team1_Away_Win_PCT': team1['AWAY_WIN_PCT'],
#         'Team2_Home_Win_PCT': team2['HOME_WIN_PCT'],
#         'Team2_Away_Win_PCT': team2['AWAY_WIN_PCT'],
#         'Team1_PIE': team1['PIE'],
#         'Team1_eFG%': team1['EFG_PCT'],
#         'Team1_TOV%': team1['TOV_PCT'],
#         'Team1_ORB%': team1['OREB_PCT'],
#         'Team1_FTR': team1['FTR'],
#         'Team2_PIE': team2['PIE'],
#         'Team2_eFG%': team2['EFG_PCT'],
#         'Team2_TOV%': team2['TOV_PCT'],
#         'Team2_ORB%': team2['OREB_PCT'],
#         'Team2_FTR': team2['FTR'],
#         'Team1Win': 1 if team1['PTS'] > team2['PTS'] else 0
#     }

#     matchups.append(row)

# # === Create final DataFrame and export ===
# df_matchups = pd.DataFrame(matchups)
# df_matchups.to_csv('NBA_Training_Matchups_2019_2025.csv', index=False)
# print("✅ Matchup data saved to 'NBA_Training_Matchups_2019_2025.csv'")


import pandas as pd

# === Load files ===
matchups_df = pd.read_csv("NBAdata/matchups/NBA_2019_20_Matchups.csv")
stats_df = pd.read_csv("NBAdata/monthly_stats/nba_team_advanced_stats_2019_20.csv")

# === Normalize team names ===
matchups_df['Team1'] = matchups_df['Team1'].str.strip().str.lower()
matchups_df['Team2'] = matchups_df['Team2'].str.strip().str.lower()
stats_df['TEAM_NAME'] = stats_df['TEAM_NAME'].str.strip().str.lower()

# === Add date-based columns for matching ===
matchups_df['DATE'] = pd.to_datetime(matchups_df['DATE'], errors='coerce')
matchups_df['Month'] = matchups_df['DATE'].dt.month
matchups_df['Season'] = "2019-20"

# === Filter only valid months ===
valid_months = stats_df['Month'].dropna().unique()
matchups_df = matchups_df[matchups_df['Month'].isin(valid_months)]

# === Select stats to join ===
selected_stats = stats_df[[
    'TEAM_NAME', 'W_PCT', 'PTS', 'PLUS_MINUS', 'FT_PCT', 'FG_PCT', 'TOV', 'OREB', 'Season', 'Month'
]]

team1_stats = selected_stats.rename(columns={
    'TEAM_NAME': 'Team1',
    'W_PCT': 'Team1_W_PCT',
    'PTS': 'Team1_PTS',
    'PLUS_MINUS': 'Team1_PLUS_MINUS',
    'FT_PCT': 'Team1_FTR',
    'FG_PCT': 'Team1_eFG%',
    'TOV': 'Team1_TOV%',
    'OREB': 'Team1_ORB%'
})

team2_stats = selected_stats.rename(columns={
    'TEAM_NAME': 'Team2',
    'W_PCT': 'Team2_W_PCT',
    'PTS': 'Team2_PTS',
    'PLUS_MINUS': 'Team2_PLUS_MINUS',
    'FT_PCT': 'Team2_FTR',
    'FG_PCT': 'Team2_eFG%',
    'TOV': 'Team2_TOV%',
    'OREB': 'Team2_ORB%'
})

# === Merge both teams' stats ===
merged_df = matchups_df.merge(team1_stats, on=['Team1', 'Season', 'Month'], how='left')
merged_df = merged_df.merge(team2_stats, on=['Team2', 'Season', 'Month'], how='left')

# === Save to file ===
merged_df.to_csv("merged_2019_20.csv", index=False)
print("✅ Saved merged_2019_20.csv")
