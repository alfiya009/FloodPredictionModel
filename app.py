from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process as fuzzy_process
import requests
import os

app = Flask(__name__)

# ---------- Configuration ----------
API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'f215342ef6fb31829da6b26256b5d768')
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# ML artifacts
# Update the model loading code in app.py
try:
    # ML artifacts
    MODEL = joblib.load("models/ensemble_model.joblib")
    SCALER = joblib.load("models/scaler.joblib")
    TARGET_ENCODER = joblib.load("models/target_encoder.joblib")
    
    TRAIN_CSV = "data/final_flood_classification data.csv"
    FORECAST_CSV = "data/mumbai_regions_7day_forecast.csv"
except Exception as e:
    print(f"Error loading models: {e}")
    # Provide fallback behavior

df_train = pd.read_csv(TRAIN_CSV)
if " Population" in df_train.columns:
    df_train.rename(columns={" Population": "Population"}, inplace=True)

# label encoders
possible_cat_cols = ["Ward Code", "Land Use Classes", "Soil Type", "Areas"]
REQUIRED_FEATURES = list(SCALER.feature_names_in_)
cat_cols = [c for c in possible_cat_cols if c in df_train.columns and c in REQUIRED_FEATURES]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    series = df_train[col].astype(str).fillna("Unknown")
    if "Unknown" not in series.values:
        series = pd.concat([series, pd.Series(["Unknown"])], ignore_index=True)
    le.fit(series)
    le_dict[col] = le

medians = df_train[REQUIRED_FEATURES].select_dtypes(include=np.number).median().to_dict()

df_forecast = pd.read_csv(FORECAST_CSV)
if 'Area' in df_forecast.columns and 'Areas' not in df_forecast.columns:
    df_forecast.rename(columns={'Area': 'Areas'}, inplace=True)

forecast_names = df_forecast['Areas'].astype(str).unique().tolist()
normalized_forecast = {name.strip().lower(): name for name in forecast_names}

# ---------- Helpers ----------
def fuzzy_match_area(user_input, limit=3):
    """Return top matches instead of only one"""
    if not user_input:
        return []
    choices = list(normalized_forecast.keys())
    results = fuzzy_process.extract(user_input.strip().lower(), choices, limit=limit)
    return [(normalized_forecast[r[0]], r[1]) for r in results]

def _encode_categoricals(row: pd.Series) -> pd.Series:
    for col, le in le_dict.items():
        if col in row.index:
            val = "Unknown" if pd.isna(row[col]) else str(row[col])
            try:
                row[col] = le.transform([val])[0]
            except Exception:
                row[col] = le.transform(["Unknown"])[0]
    return row

def prepare_features_from_forecast(area_name, forecast_row):
    row = pd.Series({c: np.nan for c in REQUIRED_FEATURES})
    row["Areas"] = area_name
    if "Latitude" in forecast_row:
        row["Latitude"] = forecast_row.get("Latitude")
    if "Longitude" in forecast_row:
        row["Longitude"] = forecast_row.get("Longitude")
    if "Ward Code" in forecast_row and "Ward Code" in REQUIRED_FEATURES:
        row["Ward Code"] = forecast_row.get("Ward Code")
    rain_fields = ["Rainfall_mm", "Rainfall (mm)", "Rainfall", "rainfall"]
    for rf in rain_fields:
        if rf in forecast_row:
            if "Rainfall_mm" in REQUIRED_FEATURES:
                row["Rainfall_mm"] = forecast_row.get(rf)
            break
    row = _encode_categoricals(row)
    for col in REQUIRED_FEATURES:
        if pd.isna(row.get(col)):
            row[col] = medians.get(col, 0)
    return pd.DataFrame([[row.get(col, 0) for col in REQUIRED_FEATURES]], columns=REQUIRED_FEATURES)

def predict_risk_from_features(df_features):
    Xs = SCALER.transform(df_features)
    pred = MODEL.predict(Xs)
    return TARGET_ENCODER.inverse_transform(pred)[0]

def get_weather(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    r = requests.get(WEATHER_URL, params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    weather = {
        "temperature": data["main"]["temp"],
        "condition": data["weather"][0]["description"].title(),
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "rainfall": data.get("rain", {}).get("1h", 0),
        "icon_url": f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png"
    }
    return weather


# ---------- Flask routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    weather_data = None
    error = None
    risk = None
    matched_area = None
    suggestions = []

    if request.method == "POST":
        user_text = request.form.get("city_or_area", "").strip()
        if not user_text:
            error = "Please enter an Area or Ward name."
        else:
            matches = fuzzy_match_area(user_text, limit=3)  # top 3 suggestions
            if not matches:
                error = f"No match found for '{user_text}'."
            else:
                matched_area, score = matches[0]
                suggestions = [m[0] for m in matches]  # show top matches

                rows = df_forecast[df_forecast['Areas'].astype(str).str.strip().str.lower() == matched_area.strip().lower()]
                if rows.empty:
                    error = "No forecast row found for matched area."
                else:
                    forecast_row = rows.iloc[0].to_dict()
                    features = prepare_features_from_forecast(matched_area, forecast_row)
                    try:
                        risk = predict_risk_from_features(features)

                        # ✅ FIX: use lat/lon for weather
                        lat = forecast_row.get("Latitude")
                        lon = forecast_row.get("Longitude")
                        if lat and lon:
                            weather_data = get_weather(lat, lon)
                        else:
                            weather_data = get_weather(19.076, 72.8777)  # fallback: Mumbai center
                    except Exception as e:
                        error = f"Prediction failed: {e}"

    return render_template("index.html",
                           error=error,
                           risk=risk,
                           matched_area=matched_area,
                           suggestions=suggestions,
                           **(weather_data or {}))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

