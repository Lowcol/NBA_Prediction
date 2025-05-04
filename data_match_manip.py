import pandas as pd
import glob

# 1. List all your monthly match files
files = glob.glob('NBAdata/2024-25_matches/*.csv')  # Adjust your path if needed

# 2. Process each file
# for file_path in files:
#     print(f"Processing {file_path}...")

#     # Load the CSV
#     games = pd.read_csv(file_path)

#     # Rename columns
#     games = games.rename(columns={
#         'Visitor/Neutral': 'Away_Team',
#         'PTS': 'Away_PTS',
#         'Home/Neutral': 'Home_Team',
#         'PTS.1': 'Home_PTS'
#     })

#     # Keep only important columns
#     games = games[['Date', 'Home_Team', 'Home_PTS', 'Away_Team', 'Away_PTS']]

#     # Create Winner column
#     games['Winner'] = games.apply(lambda row: 0 if row['Home_PTS'] > row['Away_PTS'] else 1, axis=1)

#     # Save cleaned file
#     cleaned_filename = file_path.replace('.csv', '_cleaned.csv')
#     games.to_csv(cleaned_filename, index=False)

#     print(f"Saved cleaned version to {cleaned_filename}")


file_cleaned = glob.glob('NBAdata/2024-25_matches/*_cleaned.csv')

dfs = []

for file_path in file_cleaned:
    df = pd.read_csv(file_path)
    dfs.append(df)
    
all_games = pd.concat(dfs, ignore_index=True)

print("All cleaned games saved to 'NBAdata/2024-25_matches/all_games_cleaned.csv'")