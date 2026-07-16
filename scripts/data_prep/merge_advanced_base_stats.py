import itertools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NBAdata"
CURRENT_DIR = DATA_ROOT / "monthly_stats"
HISTORICAL_DIR = DATA_ROOT / "archive" / "historical" / "monthly_stats"

NBA_TEAMS = [
    "atlanta hawks", "boston celtics", "brooklyn nets", "charlotte hornets", "chicago bulls",
    "cleveland cavaliers", "dallas mavericks", "denver nuggets", "detroit pistons", "golden state warriors",
    "houston rockets", "indiana pacers", "la clippers", "los angeles lakers", "memphis grizzlies",
    "miami heat", "milwaukee bucks", "minnesota timberwolves", "new orleans pelicans", "new york knicks",
    "oklahoma city thunder", "orlando magic", "philadelphia 76ers", "phoenix suns", "portland trail blazers",
    "sacramento kings", "san antonio spurs", "toronto raptors", "utah jazz", "washington wizards"
]
MONTHS = list(range(1, 7)) + [10, 11, 12]


def merge_season(base_path: Path, adv_path: Path, output_path: Path) -> None:
    base_df = pd.read_csv(base_path)
    adv_df = pd.read_csv(adv_path)

    base_df['TEAM_NAME'] = base_df['TEAM_NAME'].str.strip().str.lower()
    adv_df['TEAM_NAME'] = adv_df['TEAM_NAME'].str.strip().str.lower()

    season = base_df['Season'].iloc[0]

    base_df = base_df.drop_duplicates(subset=['TEAM_NAME', 'Season', 'Month'])
    adv_df = adv_df.drop_duplicates(subset=['TEAM_NAME', 'Season', 'Month'])

    merged_df = pd.merge(
        base_df,
        adv_df,
        on=['TEAM_NAME', 'Season', 'Month'],
        how='outer',
        suffixes=('_base', '_adv'),
    )

    full_index = pd.DataFrame(
        itertools.product(NBA_TEAMS, [season], MONTHS),
        columns=['TEAM_NAME', 'Season', 'Month'],
    )
    merged_full = pd.merge(full_index, merged_df, on=['TEAM_NAME', 'Season', 'Month'], how='left')

    merged_full.sort_values(by=['TEAM_NAME', 'Season', 'Month'], inplace=True)
    filled_df = (
        merged_full
        .groupby('TEAM_NAME', group_keys=False)
        .apply(lambda group: group.ffill())
        .reset_index(drop=True)
    )
    filled_df = filled_df.sort_values(by=['Season', 'Month', 'TEAM_NAME'])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filled_df.to_csv(output_path, index=False)
    print(f"Saved {output_path} (season={season})")


def main() -> None:
    merge_season(
        CURRENT_DIR / "nba_team_base_stats_2024_25.csv",
        CURRENT_DIR / "nba_team_advanced_stats_2024_25.csv",
        CURRENT_DIR / "nba_team_combined_stats_2024_25.csv",
    )

    for base_path in sorted(HISTORICAL_DIR.glob("nba_team_base_stats_*.csv")):
        suffix = base_path.name[len("nba_team_base_stats_"):]
        adv_path = HISTORICAL_DIR / f"nba_team_advanced_stats_{suffix}"
        if not adv_path.exists():
            print(f"Skipping {suffix}: no matching advanced stats file")
            continue
        output_path = HISTORICAL_DIR / f"nba_team_combined_stats_{suffix}"
        merge_season(base_path, adv_path, output_path)


if __name__ == "__main__":
    main()
