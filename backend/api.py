from flask import Flask, jsonify
from flask_cors import CORS
from inference_data import build_features_for_matchup
from team_map import TEAM_ID_TO_ABR
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the model ONCE when the server starts
model = joblib.load("rf_model.pkl")


# -----------------------
# Helper: fetch today's games
# -----------------------
def get_todays_games():
    today_str = datetime.today().strftime("%Y-%m-%d")

    sb = scoreboardv2.ScoreboardV2(
        game_date=today_str,
        day_offset=0,
        league_id="00"
    )

    games = sb.game_header.get_data_frame()

    matchups = []
    for _, row in games.iterrows():
        matchups.append({
            "game_id": row["GAME_ID"],
            "home_id": int(row["HOME_TEAM_ID"]),
            "away_id": int(row["VISITOR_TEAM_ID"]),
            "home_abbr": TEAM_ID_TO_ABR.get(int(row["HOME_TEAM_ID"]), "UNK"),
            "away_abbr": TEAM_ID_TO_ABR.get(int(row["VISITOR_TEAM_ID"]), "UNK"),
        })

    return today_str, matchups


# -----------------------
# API: Today’s predictions
# -----------------------
@app.route("/predictions/today", methods=["GET"])
def predict_today():
    today, games = get_todays_games()
    results = []

    for g in games:
        # Build rolling stats for this matchup
        feats = build_features_for_matchup(
            g["home_id"], g["away_id"], today, season_year=datetime.today().year - 1
        )

        # Align to model feature columns
        feats = feats.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict
        prob = model.predict_proba(feats)[0]
        home_prob = float(prob[1])
        away_prob = float(prob[0])

        results.append({
            "game_id": g["game_id"],
            "home_team_id": g["home_id"],
            "away_team_id": g["away_id"],
            "home_abbr": g["home_abbr"],
            "away_abbr": g["away_abbr"],
            "home_win_prob": home_prob,
            "away_win_prob": away_prob,
        })

    return jsonify({"date": today, "games": results})


if __name__ == "__main__":
    app.run(debug=True)
