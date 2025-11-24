# main.py

from inference_data import build_features_for_matchup
from team_map import TEAM_ID_TO_ABR
import joblib
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2

# -------------------------
# Load model
# -------------------------
model = joblib.load("rf_model.pkl")
model_features = model.feature_names_in_

# -------------------------
# Today's date + season
# -------------------------
today = datetime.now().strftime("%Y-%m-%d")
season_year = 2024   # update every season

print(f"\n📅 Fetching games for {today}...\n")

# -------------------------
# Pull today's games
# -------------------------
scoreboard = scoreboardv2.ScoreboardV2(
    game_date=today,
    league_id="00",
    day_offset=0
)

games = scoreboard.game_header.get_data_frame()

if games.empty:
    print("❌ No games found for today")
    exit()

pred_rows = []

# --------------------------------------
# Loop through each game on today’s slate
# --------------------------------------
for _, row in games.iterrows():

    home_id = int(row["HOME_TEAM_ID"])
    away_id = int(row["VISITOR_TEAM_ID"])

    print(f"🏀 Processing: {TEAM_ID_TO_ABR.get(home_id)} vs {TEAM_ID_TO_ABR.get(away_id)}")

    try:
        # build feature row
        feats = build_features_for_matchup(home_id, away_id, today, season_year)

        # match model feature order
        feats = feats.reindex(columns=model_features)

        # predictions
        probs = model.predict_proba(feats)[0]
        pred_class = model.predict(feats)[0]

        pred_rows.append({
            "GAME_ID": row["GAME_ID"],
            "HOME_TEAM_ID": home_id,
            "HOME_TEAM_ABR": TEAM_ID_TO_ABR.get(home_id, "UNK"),
            "AWAY_TEAM_ID": away_id,
            "AWAY_TEAM_ABR": TEAM_ID_TO_ABR.get(away_id, "UNK"),
            "HOME_WIN_PROB": float(probs[1]),
            "AWAY_WIN_PROB": float(probs[0]),
            "PREDICTED_HOME_WIN": int(pred_class)
        })

    except Exception as e:
        print(f"❌ Error processing {TEAM_ID_TO_ABR.get(home_id)} vs {TEAM_ID_TO_ABR.get(away_id)}: {e}")


# -------------------------
# Output results
# -------------------------
pred_df = pd.DataFrame(pred_rows)
print("\n🎯 FINAL PREDICTIONS:")
print(pred_df)

# Optional: Save predictions
pred_df.to_csv("predictions_today.csv", index=False)
print("\n💾 Saved predictions to predictions_today.csv")
