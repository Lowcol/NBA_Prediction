import pandas as pd

def add_month_to_matchups(input_path: str, output_path: str):
    # Load matchup file
    df = pd.read_csv(input_path)

    # Ensure DATE column is parsed properly
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # Add Month column
    df['Month'] = df['DATE'].dt.month

    # Save updated file
    df.to_csv(output_path, index=False)
    print(f"✅ Updated matchup file saved with 'Month' column: {output_path}")

# Example usage
add_month_to_matchups(
    input_path="NBAdata/matchups/NBA_2019_20_Matchups.csv",
    output_path="NBAdata/matchups/NBA_2019_20_Matchups_withMonth.csv"
)
