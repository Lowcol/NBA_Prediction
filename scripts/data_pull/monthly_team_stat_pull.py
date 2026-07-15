from nba_api.stats.endpoints import LeagueDashTeamStats
import pandas as pd
import time

season = '2024-25'  # Change this to the desired season

all_months = []

for month in range(1, 8):  # October (1) to April (7)
    print(f"Fetching {season}, Month {month}...")
    
    stats = LeagueDashTeamStats(
        season=season,
        season_type_all_star='Regular Season',
        month=month,
        measure_type_detailed_defense='Base'  # 'Base' = traditional stats
    )
    
    df = stats.get_data_frames()[0]
    df['SEASON'] = season
    df['MONTH'] = month
    
    all_months.append(df)
    time.sleep(1)  # Be nice to the API

# Combine all months
df_monthly = pd.concat(all_months, ignore_index=True)

# Save
df_monthly.to_csv(f"NBAdata/monthly_stats/NBA_Team_Monthly_Stats_{season.replace('-', '_')}.csv", index=False)
print(f"Saved stats for {season}")



# seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

# for season in seasons:
#     all_months = []
#     for month in range(1, 8):
#         print(f"Fetching {season}, Month {month}...")
#         stats = LeagueDashTeamStats(
#             season=season,
#             season_type_all_star='Regular Season',
#             month=month,
#             measure_type_detailed_defense='Base'
#         )
#         df = stats.get_data_frames()[0]
#         df['SEASON'] = season
#         df['MONTH'] = month
#         all_months.append(df)
#         time.sleep(1)
    
#     df_season = pd.concat(all_months, ignore_index=True)
#     df_season.to_csv(f"NBA_Team_Monthly_Stats_{season.replace('-', '_')}.csv", index=False)
#     print(f"Done: {season}")
