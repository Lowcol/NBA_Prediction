from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === Load your dataset ===
df = pd.read_csv('NBAdata/NBA_2019_20_Matchups_WithAdvancedStats.csv')

# === Define target + features to use ===
target = 'Team1Win'

# selected_features = [
#     'Team1_W_PCT', 'Team1_PLUS_MINUS', 'Team1_PTS', 'Team1_FG_PCT', 'Team1_AST', 'Team1_TOV',
#     'Team1_REB', 'Team1_STL', 'Team1_PF', 'Team1_BLK',
#     'Team2_W_PCT', 'Team2_PLUS_MINUS', 'Team2_PTS', 'Team2_FG_PCT', 'Team2_AST', 'Team2_TOV',
#     'Team2_REB', 'Team2_STL', 'Team2_PF', 'Team2_BLK',
#     'Team1Home', 'Team1_Home_Win_PCT', 'Team1_Away_Win_PCT', 'Team2_Home_Win_PCT', 'Team2_Away_Win_PCT',
#     'Team1_PIE', 'Team1_eFG%', 'Team1_TOV%', 'Team1_ORB%', 'Team1_FTR',
#     'Team2_PIE', 'Team2_eFG%', 'Team2_TOV%', 'Team2_ORB%', 'Team2_FTR'
# ]

selected_features = [
    'Team1_W_PCT', 'Team1_PLUS_MINUS', 'Team1_PTS',
    'Team2_W_PCT', 'Team2_PLUS_MINUS', 'Team2_PTS',
    'Team1Home', 'Team1_Home_Win_PCT', 'Team1_Away_Win_PCT', 'Team2_Home_Win_PCT', 'Team2_Away_Win_PCT',
    'Team1_PIE', 'Team1_eFG%', 'Team1_TOV%', 'Team1_ORB%', 'Team1_FTR',
    'Team2_PIE', 'Team2_eFG%', 'Team2_TOV%', 'Team2_ORB%', 'Team2_FTR'
]





# Drop rows with any missing selected features
df = df.dropna(subset=selected_features + [target])

X = df[selected_features]
y = df[target]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check balance
print("Class distribution (Team1Win):")
print(y.value_counts(normalize=True))


# === Train multiple models ===

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "SVC (RBF Kernel)": SVC(gamma='scale', probability=True),
    "Bagging SVC": BaggingClassifier(estimator=SVC(gamma='scale'), n_estimators=10, random_state=0)
}


results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# === Plot accuracy comparison ===
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="Blues_d")
plt.ylabel("Accuracy")
plt.title("Model Comparison on Selected Features")
plt.ylim(0.5, 1)
plt.tight_layout()
plt.show()