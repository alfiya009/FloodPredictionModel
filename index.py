# # app.py
# from flask import Flask, render_template, request
# import requests
# import os

# app = Flask(__name__)

# # Get API key from environment variable or use demo key
# API_KEY = 'f215342ef6fb31829da6b26256b5d768'

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     weather_data = {}
#     error = None
    
#     if request.method == 'POST':
#         city = request.form['city']
#         url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
        
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 data = response.json()
#                 weather_data = {
#                     'city': data['name'],
#                     'temperature': data['main']['temp'],
#                     'description': data['weather'][0]['description'],
#                     'icon': data['weather'][0]['icon'],
#                     'humidity': data['main']['humidity'],
#                     'wind_speed': data['wind']['speed']
#                 }
#             else:
#                 error = "City not found. Please try again."
#         except Exception as e:
#             error = "Error connecting to weather service. Please try again later."
    
#     return render_template('index.html', weather=weather_data, error=error)

# if __name__ == '__main__':
#     app.run(debug=True)




########################################################################################

# from flask import Flask, render_template, request
# import requests
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)

# API_KEY = 'f215342ef6fb31829da6b26256b5d768'

# # --- Load ML model & encoders ---
# MODEL = joblib.load("ensemble_model.joblib")
# SCALER = joblib.load("scaler.joblib")
# TARGET_ENCODER = joblib.load("target_encoder.joblib")

# df_train = pd.read_csv("final_flood_classification data.csv")
# if " Population" in df_train.columns:
#     df_train.rename(columns={" Population": "Population"}, inplace=True)

# REQUIRED_FEATURES = list(SCALER.feature_names_in_)

# # LabelEncoder for categoricals
# cat_cols = [c for c in ["Ward Code", "Land Use Classes", "Soil Type", "Areas"] if c in df_train.columns and c in REQUIRED_FEATURES]
# le_dict = {}
# for col in cat_cols:
#     le = LabelEncoder()
#     series = df_train[col].astype(str).fillna("Unknown")
#     if "Unknown" not in series.values:
#         series = pd.concat([series, pd.Series(["Unknown"])], ignore_index=True)
#     le.fit(series)
#     le_dict[col] = le

# # Numeric fallback
# medians = df_train[REQUIRED_FEATURES].select_dtypes(include=np.number).median().to_dict()

# def get_weather(city: str):
#     url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
#     r = requests.get(url, timeout=10)
#     if r.status_code != 200:
#         return None
#     data = r.json()
#     return {
#         "city": data["name"],
#         "temperature": data["main"]["temp"],
#         "description": data["weather"][0]["description"],
#         "icon": data["weather"][0]["icon"],
#         "humidity": data["main"]["humidity"],
#         "wind_speed": data["wind"]["speed"],
#         "lat": data["coord"]["lat"],
#         "lon": data["coord"]["lon"],
#         "rainfall": (data.get("rain", {}) or {}).get("1h", 0.0)
#     }

# def _encode_categoricals(row: pd.Series) -> pd.Series:
#     for col, le in le_dict.items():
#         if col in row.index:
#             val = "Unknown" if pd.isna(row[col]) else str(row[col])
#             try:
#                 row[col] = le.transform([val])[0]
#             except Exception:
#                 row[col] = le.transform(["Unknown"])[0]
#     return row

# def prepare_features(city: str, lat: float, lon: float, weather: dict):
#     row = pd.Series({c: np.nan for c in REQUIRED_FEATURES})
#     row["Areas"] = city
#     row["Latitude"] = lat
#     row["Longitude"] = lon
#     if "Rainfall_mm" in REQUIRED_FEATURES:
#         row["Rainfall_mm"] = weather.get("rainfall", 0)

#     # Encode categoricals
#     row = _encode_categoricals(row)

#     # Fill missing with medians
#     for col in REQUIRED_FEATURES:
#         if pd.isna(row.get(col)):
#             row[col] = medians.get(col, 0)

#     return pd.DataFrame([[row.get(col, 0) for col in REQUIRED_FEATURES]], columns=REQUIRED_FEATURES)

# def predict(features: pd.DataFrame) -> str:
#     Xs = SCALER.transform(features)
#     pred = MODEL.predict(Xs)
#     return TARGET_ENCODER.inverse_transform(pred)[0]

# @app.route("/", methods=["GET", "POST"])
# def index():
#     weather_data, error, risk = None, None, None

#     if request.method == "POST":
#         city = request.form["city"]
#         weather_data = get_weather(city)
#         if not weather_data:
#             error = "City not found. Please try again."
#         else:
#             try:
#                 features = prepare_features(weather_data["city"], weather_data["lat"], weather_data["lon"], weather_data)
#                 risk = predict(features)
#             except Exception as e:
#                 risk = "Prediction failed"

#     return render_template("index.html", weather=weather_data, error=error, risk=risk)

# if __name__ == "__main__":
#     app.run(debug=True)

















