import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats

# === Load your matchup data ===
matchups = pd.read_csv('NBAdata/NBA_2019_20_Matchups_WithHomeAwayWinPct.csv')

# === Pull 2019-20 advanced team stats ===
print("📡 Fetching 2019-20 advanced team stats...")
advanced_df = LeagueDashTeamStats(
    season='2019-20',
    season_type_all_star='Regular Season',
    measure_type_detailed_defense='Advanced'
).get_data_frames()[0]

print("✅ Columns returned:")
print(advanced_df.columns.tolist())

# === Keep only the relevant columns ===
advanced = advanced_df[[ 'TEAM_NAME', 'PIE', 'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT' ]]

# === Merge for Team1 ===
matchups = matchups.merge(
    advanced,
    left_on='Team1',
    right_on='TEAM_NAME',
    how='left'
).rename(columns={
    'PIE': 'Team1_PIE',
    'EFG_PCT': 'Team1_eFG%',
    'TM_TOV_PCT': 'Team1_TOV%',
    'OREB_PCT': 'Team1_ORB%'
}).drop(columns=['TEAM_NAME'])

# === Merge for Team2 ===
matchups = matchups.merge(
    advanced,
    left_on='Team2',
    right_on='TEAM_NAME',
    how='left'
).rename(columns={
    'PIE': 'Team2_PIE',
    'EFG_PCT': 'Team2_eFG%',
    'TM_TOV_PCT': 'Team2_TOV%',
    'OREB_PCT': 'Team2_ORB%'
}).drop(columns=['TEAM_NAME'])

# === Add FTR manually: FTA / FGA ===
matchups['Team1_FTR'] = matchups['Team1_FTA'] / matchups['Team1_FGA']
matchups['Team2_FTR'] = matchups['Team2_FTA'] / matchups['Team2_FGA']

# === Save final dataset ===
output_path = 'NBAdata/NBA_2019_20_Matchups_WithAdvancedStats.csv'
matchups.to_csv(output_path, index=False)
print(f"✅ Done! Enriched file saved to: {output_path}")
