import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# === Load dataset ===
df = pd.read_csv('NBAdata/NBA_2019_20_Matchups_WithAdvancedStats.csv')
target = 'Team1Win'

# === Define feature groups ===
feature_sets = {
    "All Features": [
        'Team1_W_PCT', 'Team1_PLUS_MINUS', 'Team1_PTS', 'Team1_FG_PCT', 'Team1_AST', 'Team1_TOV',
        'Team1_REB', 'Team1_STL', 'Team1_PF', 'Team1_BLK',
        'Team2_W_PCT', 'Team2_PLUS_MINUS', 'Team2_PTS', 'Team2_FG_PCT', 'Team2_AST', 'Team2_TOV',
        'Team2_REB', 'Team2_STL', 'Team2_PF', 'Team2_BLK',
        'Team1Home', 'Team1_Home_Win_PCT', 'Team1_Away_Win_PCT', 'Team2_Home_Win_PCT', 'Team2_Away_Win_PCT',
        'Team1_PIE', 'Team1_eFG%', 'Team1_TOV%', 'Team1_ORB%', 'Team1_FTR',
        'Team2_PIE', 'Team2_eFG%', 'Team2_TOV%', 'Team2_ORB%', 'Team2_FTR'
    ],
    "Core Stats Only": [
        'Team1_PTS', 'Team1_AST', 'Team1_REB', 'Team1_TOV',
        'Team2_PTS', 'Team2_AST', 'Team2_REB', 'Team2_TOV'
    ],
    "Efficiency Only": [
        'Team1_eFG%', 'Team1_TOV%', 'Team1_ORB%', 'Team1_FTR',
        'Team2_eFG%', 'Team2_TOV%', 'Team2_ORB%', 'Team2_FTR',
    ],
    "Contextual Only": [
        'Team1_W_PCT', 'Team2_W_PCT', 'Team1Home',
        'Team1_Home_Win_PCT', 'Team1_Away_Win_PCT', 'Team2_Home_Win_PCT', 'Team2_Away_Win_PCT'
    ],
    "Efficiency and Contextual": [
        'Team1_W_PCT', 'Team2_W_PCT', 'Team1Home',
        'Team1_Home_Win_PCT', 'Team1_Away_Win_PCT', 'Team2_Home_Win_PCT', 'Team2_Away_Win_PCT',
        'Team1_eFG%', 'Team1_TOV%', 'Team1_ORB%', 'Team1_FTR', 'Team1_PIE',
        'Team2_eFG%', 'Team2_TOV%', 'Team2_ORB%', 'Team2_FTR', 'Team2_PIE'
    ]
}

all_results = []

# === Sully's Four Factors weights ===
four_factor_weights = {
    'Team1_eFG%': 0.50, 'Team1_TOV%': 0.30, 'Team1_ORB%': 0.15, 'Team1_FTR': 0.05,
    'Team2_eFG%': 0.50, 'Team2_TOV%': 0.30, 'Team2_ORB%': 0.15, 'Team2_FTR': 0.05
}

# === Main loop: test each feature set ===
for name, selected_features in feature_sets.items():
    print(f"\nTraining on: {name}")

    # Drop missing values
    temp_df = df.dropna(subset=selected_features + [target]).copy()
    X = temp_df[selected_features].copy()
    y = temp_df[target]

    # Apply weights if any of Sully's features are in this set
    for feat, weight in four_factor_weights.items():
        if feat in X.columns:
            X[feat] = X[feat] * weight

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'),
        "SVC (RBF)": SVC(gamma='scale', probability=True),
        "Voting Ensemble": VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
                ('svc', SVC(gamma='scale', probability=True))
            ],
            voting='soft'
        )
    }

    best_accuracy = 0
    best_model = None
    best_model_name = ''
    best_feature_set = ''
    best_scaler = None

    # Train + Evaluate
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{model_name} Accuracy: {acc:.4f}")
        all_results.append({
            'Feature Set': name,
            'Model': model_name,
            'Accuracy': acc
        })

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = model_name
            best_feature_set = name
            best_scaler = scaler


print(f"\nBest model: {best_model_name} ({best_feature_set}) with accuracy: {best_accuracy:.4f}")

# Save model and scaler
joblib.dump(best_model, f'NBAdata/best_model_{best_model_name.replace(" ", "_")}.pkl')
joblib.dump(best_scaler, 'NBAdata/scaler.pkl')
print("Model and scaler saved.")


# # === Convert to DataFrame after loop ===
# results_df = pd.DataFrame(all_results)

# # === Plot grouped bar chart ===
# plt.figure(figsize=(12, 6))
# sns.barplot(data=results_df, x='Feature Set', y='Accuracy', hue='Model')
# plt.title('Model Accuracy by Feature Set')
# plt.xticks(rotation=20)
# plt.ylim(0.5, 1.0)
# plt.tight_layout()
# plt.legend(title='Model')
# plt.show()