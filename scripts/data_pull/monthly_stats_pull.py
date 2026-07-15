import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashteamstats

def fetch_full_season_team_stats(season: str):
    season_data = []
    season_types = ['Pre Season', 'Regular Season', 'PlayIn', 'Playoffs']

    for season_type in season_types:
        for month in range(1, 13):  # Months 1-12 (Oct-Sep)
            try:
                print(f"{season_type} - Month {month}")
                stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    month=month,
                    measure_type_detailed_defense='Base',
                    per_mode_detailed='PerGame',
                    season_type_all_star=season_type
                )
                df = stats.get_data_frames()[0]
                if df.empty:
                    print("   Warning: No data")
                    continue
                df['Season'] = season
                df['Month'] = month
                df['SeasonType'] = season_type
                season_data.append(df)
                time.sleep(3)
            except Exception as e:
                print(f"   Error: {e}")
                time.sleep(3)

    if season_data:
        full_df = pd.concat(season_data, ignore_index=True)
        filename = f"nba_team_base_stats_{season.replace('-', '_')}.csv"
        full_df.to_csv(filename, index=False)
        print(f"Saved full season stats to {filename}")
    else:
        print("Warning: No data collected")

if __name__ == "__main__":
    fetch_full_season_team_stats("2019-20")
