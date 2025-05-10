import pandas as pd
import joblib

# === Load model + scaler ===
model = joblib.load("NBAdata/best_model.pkl")
scaler = joblib.load("NBAdata/scaler.pkl")

# === Load season averages for each team ===
season_avg = pd.read_csv("NBAdata/NBA_2019_20_Season_Averages_Per_Team.csv")

# === Features used for prediction (must match training set) ===
selected_features = [
    'Team1_W_PCT', 'Team1_PLUS_MINUS', 'Team1_PTS',
    'Team2_W_PCT', 'Team2_PLUS_MINUS', 'Team2_PTS',
    'Team1Home', 'Team1_Home_Win_PCT', 'Team1_Away_Win_PCT', 'Team2_Home_Win_PCT', 'Team2_Away_Win_PCT',
    'Team1_PIE', 'Team1_eFG%', 'Team1_TOV%', 'Team1_ORB%', 'Team1_FTR',
    'Team2_PIE', 'Team2_eFG%', 'Team2_TOV%', 'Team2_ORB%', 'Team2_FTR'
]
# === Define playoff matchups (Round 1) ===
manual_matchups = [
    ("Boston Celtics", "Indiana Pacers"),
    ("Milwaukee Bucks", "Detroit Pistons"),
    ("Philadelphia 76ers", "Brooklyn Nets"),
    ("Toronto Raptors", "Orlando Magic"),
    ("Denver Nuggets", "San Antonio Spurs"),
    ("Golden State Warriors", "Los Angeles Lakers"),
    ("Houston Rockets", "Utah Jazz"),
    ("Portland Trail Blazers", "Oklahoma City Thunder")
]

print("📊 Simulating Best-of-7 Series:")
for team1, team2 in manual_matchups:
    wins = {team1: 0, team2: 0}
    game = 1

    print(f"\n🏀 {team1} vs {team2} (Best-of-7)")

    while max(wins.values()) < 4:
        # Get stats
        team1_stats = season_avg[season_avg["Team"] == team1].add_prefix("Team1_")
        team2_stats = season_avg[season_avg["Team"] == team2].add_prefix("Team2_")

        if team1_stats.empty or team2_stats.empty:
            print(f"⚠️ Missing stats for one of the teams: {team1}, {team2}")
            break

        row = pd.concat([team1_stats.reset_index(drop=True), team2_stats.reset_index(drop=True)], axis=1)
        team1_home_schedule = [1, 1, 0, 0, 1, 0, 1]
        row["Team1Home"] = team1_home_schedule[game - 1]

        row = row[selected_features]

        # Predict
        print("\n🔍 Raw Features Before Scaling:")
        print(row)

        row_scaled = scaler.transform(row)
        
        print("📏 Scaled Features:")
        print(row_scaled)

        pred = model.predict(row_scaled)[0]
        prob = model.predict_proba(row_scaled)[0] if hasattr(model, "predict_proba") else None
        print("📈 Prediction Probabilities:", prob)


        winner = team1 if pred == 0 else team2
        wins[winner] += 1

        print(f"Game {game}: {team1} vs {team2} → Winner: {winner} ({wins[team1]}-{wins[team2]})")
        if prob is not None:
            print(f"   → Confidence: {prob[pred]*100:.1f}%")

        game += 1

    series_winner = team1 if wins[team1] == 4 else team2
    print(f"🏆 Series Winner: {series_winner}")
