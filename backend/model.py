import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_PATH = "rf_model.pkl"

# --------------------------------
# LOAD NEW DATASET
# --------------------------------
df = pd.read_csv("training_data.csv")

# Sort by chronological order USING home date
df["GAME_DATE_home"] = pd.to_datetime(df["GAME_DATE_home"])
df = df.sort_values(["GAME_DATE_home", "GAME_ID"]).reset_index(drop=True)

target = "home_win"

# --------------------------------
# AUTO-DETECT ALL FEATURES
# --------------------------------
feature_cols = [
    col for col in df.columns
    if (
        "_last5_home" in col or "_last5_away" in col or
        "_last10_home" in col or "_last10_away" in col or
        "season_avg_home" in col or "season_avg_away" in col
    )
]

print(f"Using {len(feature_cols)} features.")

X = df[feature_cols]
y = df[target]

# --------------------------------
# TIME-BASED SPLIT
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False
)

# --------------------------------
# LOAD OR TRAIN MODEL
# --------------------------------
if os.path.exists(MODEL_PATH):
    print("📦 Loading existing model...")
    model = joblib.load(MODEL_PATH)

else:
    print("🧠 Training new RandomForest model…")
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Saved model to {MODEL_PATH}")

# --------------------------------
# EVALUATE MODEL
# --------------------------------
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))

# --------------------------------
# BUILD RESULTS TABLE
# --------------------------------
results = df.loc[X_test.index, [
    "GAME_ID",
    "TEAM_ABBREVIATION_home",
    "TEAM_ABBREVIATION_away",
    "PTS_home",
    "PTS_away"
]].copy()

results["HOME_WIN_PROB"] = probs
results["AWAY_WIN_PROB"] = 1 - probs
results["PREDICTED_HOME_WIN"] = preds

print("\n--- SAMPLE OUTPUT ---")
print(results.head(20))
