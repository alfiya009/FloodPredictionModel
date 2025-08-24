import pandas as pd
import requests
import osmnx as ox
import time

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_CSV = "mumbai_ward_area_floodrisk.csv"
OUTPUT_CSV = "mumbai_regions_attributes.csv"

# APIs
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_FLOOD_URL = "https://api.open-meteo.com/v1/flood"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"   # ✅ Fixed endpoint

# Load input CSV
df = pd.read_csv(INPUT_CSV)

# Prepare output
results = []

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------

def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["precipitation_sum", "precipitation_hours"],
        "hourly": "precipitation",
        "forecast_days": 2,
        "timezone": "auto"
    }
    r = requests.get(OPEN_METEO_URL, params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    try:
        rainfall_mm = sum(data["daily"]["precipitation_sum"])
        rainfall_intensity = max(data["hourly"]["precipitation"])  # mm/hr
        rainfall_days = sum(1 for val in data["daily"]["precipitation_sum"] if val > 0)
        longest_rainfall_days = max(data["daily"]["precipitation_hours"])
        return rainfall_mm, rainfall_intensity, rainfall_days, longest_rainfall_days
    except Exception:
        return None


def get_flood(lat, lon):
    params = {"latitude": lat, "longitude": lon}
    r = requests.get(OPEN_METEO_FLOOD_URL, params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    try:
        discharge = data["discharge"][0]
        station = data["river_name"]
        return discharge, station
    except Exception:
        return None


def get_elevation(lat, lon):
    r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"})
    if r.status_code != 200:
        return None
    try:
        return r.json()["results"][0]["elevation"]
    except Exception:
        return None


def get_soil(lat, lon):
    """Fetch soil type & soil wetness from SoilGrids v2 API."""
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["clay", "silt", "sand", "soc", "cfvo"],  # sample properties
        "depth": "15-30cm"
    }
    try:
        r = requests.get(SOILGRIDS_URL, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        # Example extraction
        clay = data["properties"]["layers"][0]["values"]["mean"]
        soil_type = "Clay" if clay > 30 else "Sandy"  # simplistic
        soil_wetness = data["properties"]["layers"][3]["values"]["mean"]  # soc ~ soil organic carbon
        return soil_type, soil_wetness
    except Exception:
        return None


def get_osm_features(lat, lon):
    try:
        G = ox.graph_from_point((lat, lon), dist=500, network_type="drive")
        road_density = len(G.edges) / (ox.utils_graph.graph_area(G) / 1e6)  # km/km²
        water = ox.geometries_from_point((lat, lon), tags={"waterway": True}, dist=1000)
        distance_to_water = None if water.empty else 0  # simplify: assume water nearby if found
        built_up = ox.geometries_from_point((lat, lon), tags={"building": True}, dist=1000)
        built_up_percent = (built_up.area.sum() / (1000 * 1000)) * 100 if not built_up.empty else 0
        return road_density, distance_to_water, built_up_percent
    except Exception:
        return None


# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
for idx, row in df.iterrows():
    lat, lon = row["Latitude"], row["Longitude"]

    weather = get_weather(lat, lon)
    flood = get_flood(lat, lon)
    elevation = get_elevation(lat, lon)
    soil = get_soil(lat, lon)
    osm = get_osm_features(lat, lon)

    results.append({
        "Ward Code": row["Ward Code"],
        "Area": row["Areas"],
        "Latitude": lat,
        "Longitude": lon,
        "Rainfall_mm": weather[0] if weather else None,
        "Rainfall_Intensity_mm_hr": weather[1] if weather else None,
        "Rainfall_Days_Count": weather[2] if weather else None,
        "Longest_Rainfall_Days": weather[3] if weather else None,
        "Discharge_m3s": flood[0] if flood else None,
        "Nearest_Station": flood[1] if flood else None,
        "Elevation": elevation,
        "Soil_Type": soil[0] if soil else None,
        "Soil_Wetness_Index": soil[1] if soil else None,
        "Road_Density_m": osm[0] if osm else None,
        "Distance_to_Water_m": osm[1] if osm else None,
        "Built_up_%": osm[2] if osm else None,
        "Population": None,  # needs census merge
        "Flood_Risk_Level": None  # needs NDMA/Google API
    })

    print(f"Processed {row['Areas']} ✅")
    time.sleep(1)

# Save to CSV
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ All attributes saved to {OUTPUT_CSV}")
