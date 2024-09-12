import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load the datasets
per_game_file_path = r"C:\Users\faiza\Downloads\Player Per Game.csv"
award_shares_file_path = r"C:\Users\faiza\Downloads\Player Award Shares.csv"
advanced_file_path = r"C:\Users\faiza\Downloads\Advanced.csv"
play_by_play_file_path = r"C:\Users\faiza\Downloads\Player Play By Play.csv"
shooting_file_path = r"C:\Users\faiza\Downloads\Player Shooting.csv"
totals_file_path = r"C:\Users\faiza\Downloads\Player Totals.csv"
per_100_file_path = r"C:\Users\faiza\Downloads\Per 100 Poss.csv"
per_36_file_path = r"C:\Users\faiza\Downloads\Per 36 Minutes.csv"
career_info_file_path = r"C:\Users\faiza\Downloads\Player Career Info.csv"

# Load all datasets
df = pd.read_csv(per_game_file_path)
award_df = pd.read_csv(award_shares_file_path)
advanced_df = pd.read_csv(advanced_file_path)
play_by_play_df = pd.read_csv(play_by_play_file_path)
shooting_df = pd.read_csv(shooting_file_path)
totals_df = pd.read_csv(totals_file_path)
per_100_df = pd.read_csv(per_100_file_path)
per_36_df = pd.read_csv(per_36_file_path)
career_info_df = pd.read_csv(career_info_file_path)

# 2. Fill missing values in the main player stats dataset
df.fillna(method='ffill', inplace=True)

# 3. Merge selective columns from the datasets based on 'player' and 'season'
merged_df = pd.merge(df[['player', 'season', 'pts_per_game', 'ast_per_game', 'trb_per_game', 'g']], 
                     award_df[['player', 'season', 'share']], on=['player', 'season'], how='left')

merged_df = pd.merge(merged_df, advanced_df[['player', 'season', 'per', 'ws', 'ts_percent']], 
                     on=['player', 'season'], how='left')

merged_df = pd.merge(merged_df, per_100_df[['player', 'season', 'pts_per_100_poss']], 
                     on=['player', 'season'], how='left')

merged_df = pd.merge(merged_df, per_36_df[['player', 'season', 'pts_per_36_min']], 
                     on=['player', 'season'], how='left')

# Merge player career info based on 'player' only (since there is no 'season' in career_info_df)
merged_df = pd.merge(merged_df, career_info_df[['player', 'num_seasons', 'hof']], 
                     on='player', how='left')

# 4. Fill any remaining missing values
merged_df.fillna(0, inplace=True)

# 5. Filter for players with at least 200 games played
player_stats = merged_df.groupby('player').agg({
    'pts_per_game': 'mean',
    'ast_per_game': 'mean',
    'trb_per_game': 'mean',
    'share': 'mean',
    'per': 'mean',  # Player Efficiency Rating
    'ws': 'mean',  # Win Shares
    'ts_percent': 'mean',  # True Shooting Percentage
    'pts_per_100_poss': 'mean',  # Points per 100 possessions
    'pts_per_36_min': 'mean',  # Points per 36 minutes
    'num_seasons': 'mean',  # Career length
    'hof': 'max',  # Hall of Fame status (1 if Hall of Famer, 0 otherwise)
    'g': 'sum'  # Total games played
}).reset_index()

# Apply the filter for players with at least 200 games
player_stats = player_stats[player_stats['g'] >= 200]

# 6. Adjust the GOAT score with increased weight for awards and Hall of Fame status
player_stats['goat_score'] = (
    player_stats['pts_per_game'] * 0.4 +  # Points per game gets a strong weight
    player_stats['ast_per_game'] * 0.2 +  # Assists medium weight
    player_stats['trb_per_game'] * 0.1 +  # Rebounds lowest weight
    player_stats['share'] * 0.6 +  # High weight for award shares
    player_stats['per'] * 0.3 +  # Player Efficiency Rating
    player_stats['ws'] * 0.4 +  # Win Shares
    player_stats['ts_percent'] * 0.2 +  # True shooting percentage
    player_stats['pts_per_100_poss'] * 0.3 +  # Points per 100 possessions
    player_stats['pts_per_36_min'] * 0.3 +  # Points per 36 minutes
    player_stats['num_seasons'] * 0.2 +  # Career length
    player_stats['hof'] * 1.0  # Give maximum weight to Hall of Fame status
)

# 7. Sort players by GOAT score and display the top 10
player_stats = player_stats.sort_values(by='goat_score', ascending=False)

# Print the top 10 players by GOAT score
print("Top 10 players by GOAT score after filtering players with at least 200 games:")
print(player_stats.head(10))

# 8. Visualize the top 10 players using a bar plot
top_players = player_stats.head(10)
sns.barplot(x='goat_score', y='player', data=top_players)
plt.title('Top 10 GOAT Players by GOAT Score')
plt.show()

# 9. Prepare the data for machine learning model (features and target)
X = player_stats[['pts_per_game', 'ast_per_game', 'trb_per_game', 'share', 'per', 'ws', 'ts_percent', 'pts_per_100_poss', 'pts_per_36_min', 'num_seasons', 'g', 'hof']]  # Features
y = player_stats['goat_score']  # Target (GOAT score)

# 10. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Train a Random Forest model to predict the GOAT score
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 12. Predict the GOAT score for the test set
y_pred = model.predict(X_test)

# 13. Print the predicted GOAT scores for the test set
print("Predicted GOAT scores for the test set:")
print(y_pred)
