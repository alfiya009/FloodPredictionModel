import pandas as pd
import requests
import osmnx as ox
import time
import joblib

# Load trained models
ensemble_model = joblib.load('working_final_ensemble.joblib')
scaler = joblib.load('working_final_scaler.joblib')
target_encoder = joblib.load('working_final_label_encoder.joblib')

# APIs
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Model features (must match training)
MODEL_FEATURES = [
    'Latitude', 'Longitude', 'Population', 'Road Density_m', 'Rainfall_mm',
    'Rainfall_Intensity_mm_hr', 'Discharge_m3s', 'Elevation', 'Soil Wetness Index',
    'Rainfall Days Count', 'Longest rainfall _days', 'Distance_to_water_m', 'Built_up%',
    'True_nearest_distance_m', 'true_conditions_count'
]

def get_weather(lat, lon):
    """Fetch 7-day forecast"""
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ["precipitation_sum", "precipitation_hours"],
        "forecast_days": 7, "timezone": "auto"
    }
    r = requests.get(OPEN_METEO_URL, params=params)
    if r.status_code != 200: return None
    data = r.json()
    try:
        rainfall_mm = sum(data["daily"]["precipitation_sum"])
        rainfall_days = sum(1 for val in data["daily"]["precipitation_sum"] if val > 0)
        longest_rainfall_days = max(data["daily"]["precipitation_hours"])
        rainfall_intensity = max(data["daily"]["precipitation_sum"])
        return rainfall_mm, rainfall_intensity, rainfall_days, longest_rainfall_days
    except Exception:
        return None

def get_elevation(lat, lon):
    r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"})
    if r.status_code != 200: return None
    return r.json()["results"][0]["elevation"]

def get_soil(lat, lon):
    params = {"lon": lon, "lat": lat, "property": ["clay","silt","sand","soc"], "depth":"15-30cm"}
    r = requests.get(SOILGRIDS_URL, params=params, timeout=15)
    if r.status_code != 200: return None
    data = r.json()
    clay = data["properties"]["layers"][0]["values"]["mean"]
    soil_wetness = data["properties"]["layers"][3]["values"]["mean"]
    return clay, soil_wetness

def get_osm_features(lat, lon):
    G = ox.graph_from_point((lat, lon), dist=500, network_type="drive")
    road_density = len(G.edges) / (ox.utils_graph.graph_area(G)/1e6)
    water = ox.geometries_from_point((lat, lon), tags={"waterway": True}, dist=1000)
    distance_to_water = 0 if not water.empty else None
    built_up = ox.geometries_from_point((lat, lon), tags={"building": True}, dist=1000)
    built_up_percent = (built_up.area.sum()/(1000*1000))*100 if not built_up.empty else 0
    return road_density, distance_to_water, built_up_percent

def main():
    # Ask user for location
    lat = float(input("Enter Latitude: "))
    lon = float(input("Enter Longitude: "))

    weather = get_weather(lat, lon)
    elevation = get_elevation(lat, lon)
    soil = get_soil(lat, lon)
    osm = get_osm_features(lat, lon)

    if not weather or not elevation or not soil or not osm:
        print("❌ Error fetching data. Try again.")
        return

    # Prepare DataFrame for prediction
    input_data = pd.DataFrame({
        'Latitude': [lat],
        'Longitude': [lon],
        'Population': [100000],
        'Road Density_m': [osm[0]],
        'Rainfall_mm': [weather[0]],
        'Rainfall_Intensity_mm_hr': [weather[1]],
        'Discharge_m3s': [0],
        'Elevation': [elevation],
        'Soil Wetness Index': [soil[1]],
        'Rainfall Days Count': [weather[2]],
        'Longest rainfall _days': [weather[3]],
        'Distance_to_water_m': [osm[1]],
        'Built_up%': [osm[2]],
        'True_nearest_distance_m': [0],
        'true_conditions_count': [0]
    }, columns=MODEL_FEATURES)

    # Scale numeric features
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction_numeric = ensemble_model.predict(input_data_scaled)
    risk_level = target_encoder.inverse_transform(prediction_numeric)[0]

    print(f"✅ Predicted Flood Risk Level for this location: {risk_level}")

if __name__ == "__main__":
    main()
