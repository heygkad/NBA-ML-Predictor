import os
import pandas as pd

file_path = os.path.join(os.getcwd(), 'nba_team_game_logs_rolling_2008_2024.csv')
df = pd.read_csv(file_path)

# Identify home/away based on MATCHUP string
df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ")

home_df = df[df["IS_HOME"] == True]
away_df = df[df["IS_HOME"] == False]

# Merge on GAME_ID
games = home_df.merge(
    away_df,
    on="GAME_ID",
    suffixes=("_home", "_away")
)

# Create target label
games["home_win"] = (games["PTS_home"] > games["PTS_away"]).astype(int)


games.to_csv("training_data.csv", index=False)
