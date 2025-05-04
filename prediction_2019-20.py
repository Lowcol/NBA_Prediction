import pandas as pd
import joblib
from collections import Counter

# === Load model + scaler ===
model = joblib.load("NBAdata/final_model.pkl")
scaler = joblib.load("NBAdata/final_scaler.pkl")

# === Load playoff matchup stats ===
matchups_df = pd.read_csv("NBAdata/NBA_2019_20_Playoff_Matchups_ToPredict.csv")

# === Features used for prediction (should match training) ===
selected_features = [
    'Team1_W_PCT', 'Team2_W_PCT', 'Team1Home',
    'Team1_Home_Win_PCT', 'Team1_Away_Win_PCT', 'Team2_Home_Win_PCT', 'Team2_Away_Win_PCT',
    'Team1_eFG%', 'Team1_TOV%', 'Team1_ORB%', 'Team1_FTR', 'Team1_PIE',
    'Team2_eFG%', 'Team2_TOV%', 'Team2_ORB%', 'Team2_FTR', 'Team2_PIE'
]

# === Bracket structure ===
initial_pairs = list(zip(matchups_df['Team1'], matchups_df['Team2']))

def simulate_series(team1, team2, df):
    """Simulate a best-of-7 series using model predictions."""
    wins = {team1: 0, team2: 0}
    row = df[(df['Team1'] == team1) & (df['Team2'] == team2)]

    if row.empty:
        row = df[(df['Team1'] == team2) & (df['Team2'] == team1)]
        if row.empty:
            print(f"⚠️ Matchup data not found for: {team1} vs {team2}")
            return team1  # fallback
        flipped = True
    else:
        flipped = False

    row = row[selected_features]
    row_scaled = scaler.transform(row)

    # Simulate up to 7 games
    for _ in range(7):
        pred = model.predict(row_scaled)[0]
        winner = team2 if flipped and pred == 0 else team1 if not flipped and pred == 0 else team2 if not flipped else team1
        wins[winner] += 1
        if wins[winner] == 4:
            return winner
    return team1 if wins[team1] > wins[team2] else team2  # fallback

# === Simulate full playoffs ===
rounds = [initial_pairs]
champion = None

print("🏀 Starting 2019–20 Playoff Simulation (Best of 7):")
round_num = 1
while rounds[-1]:
    current_round = rounds[-1]
    next_round = []
    print(f"\n🔁 Round {round_num} Results:")
    for team1, team2 in current_round:
        winner = simulate_series(team1, team2, matchups_df)
        print(f"  ✅ {winner} advances over {team2 if winner == team1 else team1}")
        next_round.append(winner)
    
    # Pair next round teams
    next_pairs = list(zip(next_round[::2], next_round[1::2]))
    if not next_pairs:
        champion = next_round[0]
        break
    rounds.append(next_pairs)
    round_num += 1

print(f"\n🏆 Predicted NBA Champion: {champion}")
