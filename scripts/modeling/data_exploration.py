import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your dataset
data = pd.read_csv('NBAdata/NBA_Team_Boxscores_2024_25.csv')  # or any of your season files

# 2. Encode W/L column to 1/0 (if not done yet)
data['WIN'] = data['WL'].apply(lambda x: 1 if x == 'W' else 0)

# 3. Drop useless columns for correlation
columns_to_ignore = ['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL']
data_corr = data.drop(columns=columns_to_ignore)

# 4. Calculate correlation matrix
corr_matrix = data_corr.corr()

# 5. Plot heatmap
plt.figure(figsize=(16,12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# 6. (Optional) Focus only on correlation with WIN
print("\nCorrelation with WIN (descending):")
print(corr_matrix['WIN'].sort_values(ascending=False))
