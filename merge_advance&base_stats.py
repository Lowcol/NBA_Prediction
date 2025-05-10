import pandas as pd
import itertools

# === Load files ===
base_df = pd.read_csv("NBAdata/monthly_stats/nba_team_base_stats_2024_25.csv")
adv_df = pd.read_csv("NBAdata/monthly_stats/nba_team_advanced_stats_2024_25.csv")

# Normalize names
base_df['TEAM_NAME'] = base_df['TEAM_NAME'].str.strip().str.lower()
adv_df['TEAM_NAME'] = adv_df['TEAM_NAME'].str.strip().str.lower()

# Drop duplicates
base_df = base_df.drop_duplicates(subset=['TEAM_NAME', 'Season', 'Month'])
adv_df = adv_df.drop_duplicates(subset=['TEAM_NAME', 'Season', 'Month'])

# Merge base + advanced
merged_df = pd.merge(
    base_df,
    adv_df,
    on=['TEAM_NAME', 'Season', 'Month'],
    how='outer',
    suffixes=('_base', '_adv')
)

# Create full index of team-month-season combos
nba_teams = [
    "atlanta hawks", "boston celtics", "brooklyn nets", "charlotte hornets", "chicago bulls",
    "cleveland cavaliers", "dallas mavericks", "denver nuggets", "detroit pistons", "golden state warriors",
    "houston rockets", "indiana pacers", "la clippers", "los angeles lakers", "memphis grizzlies",
    "miami heat", "milwaukee bucks", "minnesota timberwolves", "new orleans pelicans", "new york knicks",
    "oklahoma city thunder", "orlando magic", "philadelphia 76ers", "phoenix suns", "portland trail blazers",
    "sacramento kings", "san antonio spurs", "toronto raptors", "utah jazz", "washington wizards"
]
season = "2019-20"
months = list(range(1, 7)) + [10, 11, 12]
full_index = pd.DataFrame(itertools.product(nba_teams, [season], months), columns=['TEAM_NAME', 'Season', 'Month'])

# Merge with full index and preserve team metadata
merged_full = pd.merge(full_index, merged_df, on=['TEAM_NAME', 'Season', 'Month'], how='left')

# Sort and fill missing values
merged_full.sort_values(by=['TEAM_NAME', 'Season', 'Month'], inplace=True)
filled_df = (
    merged_full
    .groupby('TEAM_NAME', group_keys=False)
    .apply(lambda group: group.ffill())
    .reset_index(drop=True)  # Keep TEAM_NAME after groupby
)

# Final sort
filled_df = filled_df.sort_values(by=['Season', 'Month', 'TEAM_NAME'])

# Save to file
filled_df.to_csv("NBAdata/monthly_stats/nba_team_combined_stats_2024_25.csv", index=False)
print("✅ Final padded and forward-filled dataset saved.")
