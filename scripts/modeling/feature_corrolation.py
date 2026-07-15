import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load the matchup-format dataset ===
# df = pd.read_csv('NBAdata/merged_gameStats_monthStats/NBA_2019_20_Matchups.csv')
df = pd.read_csv('NBAdata/NBA_2019_20_Matchups_WithAdvancedStats.csv')

# === Rename WIN column (Team1Win is already binary) ===
df['WIN'] = df['Team1Win']  # for consistency with rest of code

# === Keep only numeric features ===
df_clean = df.select_dtypes(include='number')

# === Drop columns that are all NaN or constant ===
df_clean = df_clean.dropna(axis=1, how='all')
df_clean = df_clean.loc[:, df_clean.nunique() > 1]

# === Compute correlation matrix ===
corr_matrix = df_clean.corr()

# === Focus on correlation with WIN ===
win_corr = corr_matrix['WIN'].sort_values(ascending=False)

# === Plot 1: Top 20 features most positively correlated with winning ===
top_features = win_corr.drop('WIN').head(30)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_features.values, y=top_features.index, palette='coolwarm')
plt.title('Top 20 Features Most Positively Correlated with Team1Win')
plt.xlabel('Correlation with Team1Win')
plt.tight_layout()
plt.show()

# === Plot 2: Full heatmap of numeric features ===
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of All Numeric Features')
plt.tight_layout()
plt.show()
