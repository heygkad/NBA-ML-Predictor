# inference_data.py

import time
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import teamgamelogs

# Columns identical to training
numeric_cols = [
    "PTS", "REB", "AST", "TOV", "STL", "BLK",
    "OREB", "DREB", "FG_PCT", "FG3_PCT", "FT_PCT",
    "PF", "PLUS_MINUS"
]

# Cache: {(team_id, season_str, today_str): logs_df}
_logs_cache = {}


# ---------------------------------------------------------
# SAFE API FUNCTION with caching + rate-limit protection
# ---------------------------------------------------------
def fetch_team_logs_cached(team_id: int, season_str: str, today_date: str):
    key = (team_id, season_str, today_date)

    if key in _logs_cache:
        return _logs_cache[key]

    # API RATE LIMIT FRIENDLY DELAY
    time.sleep(0.7)

    print(f"📥 Fetching logs for team {team_id} season={season_str}")

    yesterday = (datetime.strptime(today_date, "%Y-%m-%d") -
                 timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch from NBA API
    logs = teamgamelogs.TeamGameLogs(
        season_nullable=season_str,
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    # Filter team + up to yesterday
    logs = logs[
        (logs["TEAM_ID"] == team_id) &
        (logs["GAME_DATE"] <= yesterday)
    ].sort_values("GAME_DATE").reset_index(drop=True)

    # Store in cache
    _logs_cache[key] = logs
    return logs


# ---------------------------------------------------------
# Compute rolling + season averages (matches training)
# ---------------------------------------------------------
def compute_features(team_df: pd.DataFrame, prefix: str):
    if len(team_df) == 0:
        raise ValueError(f"No logs available for prefix={prefix}")

    df = team_df.copy()

    new_cols = {}

    # Rolling features
    for col in numeric_cols:
        new_cols[f"{col}_last5_{prefix}"] = df[col].rolling(5).mean()
        new_cols[f"{col}_last10_{prefix}"] = df[col].rolling(10).mean()

    # Season averages
    for col in numeric_cols:
        new_cols[f"{col}_season_avg_{prefix}"] = df[col].expanding().mean()

    feat_df = pd.DataFrame(new_cols)

    # return only latest row
    return feat_df.tail(1).reset_index(drop=True)


# ---------------------------------------------------------
# Build features for ONE matchup (home vs away)
# ---------------------------------------------------------
def build_features_for_matchup(home_id, away_id, today_date, season_year):
    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    print(f"\n🧱 Building features: {home_id} vs {away_id} (season {season_str})")

    # Fetch logs (cached)
    home_logs = fetch_team_logs_cached(home_id, season_str, today_date)
    away_logs = fetch_team_logs_cached(away_id, season_str, today_date)

    # Compute rolling + season averages
    home_feats = compute_features(home_logs, "home")
    away_feats = compute_features(away_logs, "away")

    # Merge horizontally into 1 row
    row = pd.concat([home_feats, away_feats], axis=1)

    return row
