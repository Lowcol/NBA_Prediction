from nba_api.stats.endpoints import leaguegamefinder
from curl_cffi import requests as cr
from nba_api.stats.library.http import NBAStatsHTTP
import pandas as pd
import time
import os

# --- THE NBA TLS BYPASS ---
# 1. Create a spoofed Chrome 120 session
session = cr.Session(impersonate="chrome120")

# 2. Do a "warmup" ping to grab the Akamai security cookies
print("Warming up Akamai cookies...")
session.get("https://www.nba.com/stats/", timeout=20) 

# 3. Force nba_api to use our spoofed session instead of standard requests
NBAStatsHTTP.get_session = lambda self: session
# --------------------------

# Seasons and output filenames
seasons = {
    "2019-20": "NBA_2019_20_Matchups.csv",
    "2020-21": "NBA_2020_21_Matchups.csv",
    "2021-22": "NBA_2021_22_Matchups.csv",
    "2022-23": "NBA_2022_23_Matchups.csv",
    "2023-24": "NBA_2023_24_Matchups.csv",
    "2024-25": "NBA_2024_25_Matchups.csv"
}

# Season types to include
season_types = ['Regular Season', 'Playoffs', 'PlayIn']

for season, filename in seasons.items():
    print(f"\nFetching {season} game data...")
    all_rows = []

    for season_type in season_types:
        print(f"   - {season_type}...")
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                player_or_team_abbreviation='T'
            )
            time.sleep(1)  # avoid rate limit
            df = finder.get_data_frames()[0]
        except Exception as e:
            print(f"   Error fetching {season_type} data: {e}")
            continue

        df = df[[
            'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL',
            'PTS', 'FG_PCT', 'AST', 'REB', 'TOV', 'PLUS_MINUS'
        ]].sort_values(['GAME_ID', 'TEAM_NAME'])

        # Group into matchups
        for game_id, group in df.groupby('GAME_ID'):
            if len(group) != 2:
                continue
            team1, team2 = group.iloc[0], group.iloc[1]
            row = {
                'GAME_ID': game_id,
                'DATE': team1['GAME_DATE'],
                'Team1': team1['TEAM_NAME'],
                'Team2': team2['TEAM_NAME'],
                'Team1_PTS': team1['PTS'],
                'Team2_PTS': team2['PTS'],
                'Team1_FG_PCT': team1['FG_PCT'],
                'Team2_FG_PCT': team2['FG_PCT'],
                'Team1_AST': team1['AST'],
                'Team2_AST': team2['AST'],
                'Team1_REB': team1['REB'],
                'Team2_REB': team2['REB'],
                'Team1_TOV': team1['TOV'],
                'Team2_TOV': team2['TOV'],
                'Team1_PLUS_MINUS': team1['PLUS_MINUS'],
                'Team2_PLUS_MINUS': team2['PLUS_MINUS'],
                'Team1Home': 1 if 'vs.' in team1['MATCHUP'] else 0,
                'Team1Win': 1 if team1['PTS'] > team2['PTS'] else 0
            }
            all_rows.append(row)

    # Build DataFrame
    matchups_df = pd.DataFrame(all_rows)

    # Normalize team names
    matchups_df['Team1'] = matchups_df['Team1'].str.strip().str.lower()
    matchups_df['Team2'] = matchups_df['Team2'].str.strip().str.lower()

    # Save file
    os.makedirs("NBAdata/matchups", exist_ok=True)
    matchups_df.to_csv(f'NBAdata/matchups/{filename}', index=False)
    print(f"Saved: NBAdata/matchups/{filename} with {len(matchups_df)} games.")
