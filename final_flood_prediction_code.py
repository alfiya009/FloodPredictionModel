import pandas as pd
import joblib
import requests
import os
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rapidfuzz import process as fuzzy_process

# Suppress warnings that are not critical for a simple prediction
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- STEP 1: Load the trained model and other files ---
try:
    ensemble_model = joblib.load('ensemble_model.joblib')
    scaler = joblib.load('scaler.joblib')
    target_encoder = joblib.load('target_encoder.joblib')
    print("✅ ML model, scaler, and encoder loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Please ensure model files are in the same directory.")
    exit()

# --- STEP 2: Configure API and Model Features ---
API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'f215342ef6fb31829da6b26256b5d768')
MODEL_FEATURES = [
    'Ward Code', 'Latitude', 'Longitude', 'Population', 'Road Density_m',
    'Land Use Classes', 'Soil Type', 'Average_temp', 'Average_humidity',
    'Rainfall (mm)', 'Evapotranspiration', 'Groundwater_recharge', 'Slope',
    'Aspect', 'Curvature'
]

# Load the entire original dataset to get city-specific features
try:
    df_full = pd.read_csv("final_flood_classification data.csv")
    
    if ' Population' in df_full.columns:
        df_full.rename(columns={' Population': 'Population'}, inplace=True)
    if 'Discharge (m³/s)' in df_full.columns:
        df_full.rename(columns={'Discharge (m³/s)': 'Discharge_m3s'}, inplace=True)

    df_full = df_full.replace("--", np.nan)
    df_full = df_full.fillna(df_full.median(numeric_only=True))
    
    cat_cols_full = ["Ward Code", "Land Use Classes", "Soil Type"]
    le_dict = {col: LabelEncoder().fit(df_full[col].astype(str).fillna("Unknown")) for col in cat_cols_full}

    df_full_clean = df_full.copy()
    for col, le in le_dict.items():
        df_full_clean[col] = le.transform(df_full_clean[col].astype(str).fillna("Unknown"))

    features_to_drop = [
        "Flood-risk_level", "DATE", "true_conditions_count", 
        "Flood_occured", "Monitoring_required", "Drainage_properties", 
        "Drainage_line_id", "Nearest Station"
    ]
    
    df_full_clean = df_full_clean.drop(columns=features_to_drop, errors='ignore')
    MODEL_FEATURES = df_full_clean.columns.tolist()

    PLACEHOLDER_DATA = {col: df_full_clean[col].median() for col in df_full_clean.select_dtypes(include=np.number).columns}
    CITIES = df_full['Areas'].unique().tolist()

except FileNotFoundError as e:
    print(f"❌ Error: {e}. Please ensure 'final_flood_classification data.csv' is present.")
    exit()

# --- STEP 3: Define Helper Functions ---
def get_weather_data(city):
    """Fetches live weather data for a given city."""
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': data.get('rain', {}).get('1h', 0)
        }
    else:
        raise Exception(f"Failed to fetch weather data for {city}: {response.status_code}")

def predict_risk(city_name):
    """Prepares data and makes a prediction using the loaded model."""
    try:
        weather_data = get_weather_data(city_name)
        print(f"\n✅ Weather Data for {city_name}: {weather_data}")

        # Get city-specific data as base for prediction input
        city_row = df_full[df_full['Areas'] == city_name].iloc[0]
        input_data = city_row.drop(features_to_drop + ['Areas', 'Nearest Station'], errors='ignore')
        
        # Encode categorical features for the input
        for col, le in le_dict.items():
            input_data[col] = le.transform([city_row[col].astype(str).fillna("Unknown")])[0]

        # Update with live weather data
        input_data['Rainfall (mm)'] = weather_data['rainfall']
        
        # Ensure all columns are in the correct order
        input_data_df = pd.DataFrame([input_data.values], columns=MODEL_FEATURES)

        # Scale the data using the saved scaler
        scaled_input = scaler.transform(input_data_df)
        
        # Make the final prediction
        prediction_numeric = ensemble_model.predict(scaled_input)
        prediction_label = target_encoder.inverse_transform(prediction_numeric)
        
        return prediction_label[0]
        
    except Exception as e:
        print(f"❌ Could not get prediction for {city_name}: {e}")
        return "Prediction failed"

# --- Main execution block ---
# --- Main execution block ---
if __name__ == "__main__":
    # Normalize city list once (lowercase & strip spaces)
    normalized_cities = {c.strip().lower(): c for c in CITIES}  # maps lowercase -> original

    while True:
        city_input = input("Enter a city name to predict flood risk: ").strip().lower()
        if not city_input:
            print("City name cannot be empty. Please try again.")
            continue

        # Fuzzy match against normalized city names
        match_result = fuzzy_process.extractOne(city_input, list(normalized_cities.keys()))
        if match_result:
            best_match_key, score = match_result[0], match_result[1]
            best_match = normalized_cities[best_match_key]

            if score > 60:  # allow partial matches
                print(f"\n🔎 Matched your input '{city_input}' with '{best_match}' (Confidence: {score}%)")
                predicted_risk = predict_risk(best_match)
                if predicted_risk:
                    print(f"\n--- Flood Risk Report for {best_match} ---")
                    print(f"Predicted Flood Risk: {predicted_risk}")
                    print("----------------------------------------")
                break
            else:
                # show top 5 suggestions if score too low
                top_matches = fuzzy_process.extract(city_input, list(normalized_cities.keys()), limit=5)
                suggestions = [normalized_cities[m[0]] for m in top_matches]
                print(f"❌ Could not find a good matching city for '{city_input}'. Did you mean one of these? {suggestions}")
        else:
            print(f"❌ Could not find a matching city for '{city_input}'. Please check the spelling.")
