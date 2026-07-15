from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd
import time

# List of past 5 seasons
seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

for season in seasons:
    print(f"Fetching data for {season}...")

    # Fetch team game logs
    gamelog = LeagueGameLog(
        season=season,
        season_type_all_star='Regular Season',
        player_or_team_abbreviation='T'  # 'T' for Teams
    )

    # Get the dataframe
    df = gamelog.get_data_frames()[0]

    # Select important columns only
    important_columns = [
        'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 
        'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
        'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PLUS_MINUS'
    ]

    df = df[important_columns]

    # Sort by game date
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')

    # Save to CSV
    output_path = f'NBAdata/NBA_Team_Boxscores_{season.replace("-", "_")}.csv'
    df.to_csv(output_path, index=False)

    print(f"Saved {df.shape[0]} rows into {output_path}")

    # Small sleep to avoid being rate-limited by NBA API
    time.sleep(1)

print("All seasons fetched successfully!")
