from nba_api.stats.endpoints import teamgamelogs
import pandas as pd
import time

def fetch_team_game_logs(start_year=2008, end_year=2024):
    all_logs = []

    # Columns to remove
    drop_cols = [
        "GP_RANK","W_RANK","L_RANK","W_PCT_RANK","MIN_RANK",
        "FGM_RANK","FGA_RANK","FG_PCT_RANK","FG3M_RANK","FG3A_RANK","FG3_PCT_RANK",
        "FTM_RANK","FTA_RANK","FT_PCT_RANK",
        "OREB_RANK","DREB_RANK","REB_RANK","AST_RANK","TOV_RANK","STL_RANK",
        "BLK_RANK","BLKA_RANK",
        "PF_RANK","PFD_RANK","PTS_RANK","PLUS_MINUS_RANK",
        "AVAILABLE_FLAG"
    ]

    for year in range(start_year, end_year + 1):
        season_str = f"{year}-{str(year+1)[-2:]}"
        print(f"Fetching season {season_str}...")

        try:
            logs = teamgamelogs.TeamGameLogs(
                season_nullable=season_str,
                season_type_nullable="Regular Season"
            ).get_data_frames()[0]

            logs = logs.drop(columns=[c for c in drop_cols if c in logs.columns])
            logs["SEASON"] = season_str
            all_logs.append(logs)

        except Exception as e:
            print(f"Failed for {season_str}: {e}")

        time.sleep(1)

    # Combine all seasons
    df = pd.concat(all_logs, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])

    return df



def add_rolling_and_season_averages(df):

    # Stats we want to compute rolling & season averages on
    stat_cols = [
        "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
        "REB", "AST", "TOV", "STL", "BLK",
        "OREB", "DREB", "PF", "PLUS_MINUS"
    ]

    # Compute rolling averages
    for window in [5, 10]:
        for col in stat_cols:
            df[f"{col}_last{window}"] = (
                df.groupby("TEAM_ID")[col]
                  .rolling(window, min_periods=1)
                  .mean()
                  .reset_index(drop=True)
            )

    # Compute season-to-date averages
    for col in stat_cols:
        df[f"{col}_season_avg"] = (
            df.groupby(["TEAM_ID", "SEASON"])[col]
              .expanding()
              .mean()
              .reset_index(level=[0,1], drop=True)
        )

    return df



# -----------------------------
# Run everything
# -----------------------------
df = fetch_team_game_logs(2008, 2024)
df = add_rolling_and_season_averages(df)

df.to_csv("nba_team_game_logs_rolling_2008_2024.csv", index=False)

print("\nSaved CSV as: nba_team_game_logs_rolling_2008_2024.csv")
print(df.head())
