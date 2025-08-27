import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_CSV = "mumbai_static_areas_unique.csv"
OUTPUT_CSV = "mumbai_regions_7day_forecast.csv"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Load static input CSV
df = pd.read_csv(INPUT_CSV)

results = []

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------
def get_weather_7days(lat, lon):
    """Fetch 7-day weather forecast with rainfall + precipitation hours."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["precipitation_sum", "precipitation_hours"],
        "hourly": "precipitation",
        "forecast_days": 7,
        "timezone": "auto"
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    try:
        daily_precip = data["daily"]["precipitation_sum"]
        daily_hours = data["daily"]["precipitation_hours"]
        daily_dates = data["daily"]["time"]

        # compute daily max rainfall intensity (mm/hr)
        intensity_per_day = []
        for d in range(7):
            day_start = datetime.fromisoformat(daily_dates[d])
            day_end = day_start + timedelta(days=1)
            # collect hourly precip for that day
            daily_vals = [
                val for t, val in zip(data["hourly"]["time"], data["hourly"]["precipitation"])
                if day_start <= datetime.fromisoformat(t) < day_end
            ]
            intensity_per_day.append(max(daily_vals) if daily_vals else 0)

        # rainfall days count (binary 1 if >0 rain else 0)
        rainfall_day_flags = [1 if v > 0 else 0 for v in daily_precip]

        return list(zip(daily_dates, daily_precip, intensity_per_day, rainfall_day_flags, daily_hours))
    except Exception:
        return None

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
for idx, row in df.iterrows():
    lat, lon = row["Latitude"], row["Longitude"]

    weather_7days = get_weather_7days(lat, lon)

    if weather_7days:
        for day, rain_mm, intensity, rain_flag, rain_hours in weather_7days:
            results.append({
                "Date": day,  # ✅ Date at left
                "Ward Code": row["Ward Code"],
                "Area": row["Areas"],
                "Latitude": lat,
                "Longitude": lon,
                "Nearest Station": row.get("Nearest Station", None),
                "Elevation": row.get("Elevation", None),
                "Land Use Classes": row.get("Land Use Classes", None),
                "Population": row.get("Population", None),
                "Road Density_m": row.get("Road Density_m", None),
                "Distance_to_water_m": row.get("Distance_to_water_m", None),
                "Soil Type": row.get("Soil Type", None),
                "Built_up%": row.get("Built_up%", None),
                "True_nearest_distance_m": row.get("True_nearest_distance_m", None),

                # dynamic attributes
                "Rainfall_mm": rain_mm,
                "Rainfall_Intensity_mm_hr": intensity,
                "Rainfall_Days_Count": rain_flag,
                "Rainfall_Hours": rain_hours
            })

    print(f"Processed {row['Areas']} ✅")
    time.sleep(1)  # avoid overloading API

# Save final CSV
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ All 7-day forecast + static attributes saved to {OUTPUT_CSV}")
