import pandas as pd
import joblib
import os
import numpy as np
import warnings
from datetime import datetime
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process as fuzzy_process

# Suppress warnings that are not critical for a simple prediction
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- STEP 1: Load the trained model and other files ---
try:
    ensemble_model = joblib.load('models/ensemble_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    target_encoder = joblib.load('models/target_encoder.joblib')
    print("✅ ML model, scaler, and encoder loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Please ensure model files are in the same directory.")
    exit()

# --- STEP 2: Load data files ---
try:
    # Load the training dataset for features info
    df_full = pd.read_csv("data/final_flood_classification data.csv")
    
    # Load the forecast data (from 7days.py output)
    forecast_file = "data/mumbai_regions_7day_forecast.csv"
    if os.path.exists(forecast_file):
        df_forecast = pd.read_csv(forecast_file)
        forecast_last_modified = datetime.fromtimestamp(os.path.getmtime(forecast_file))
        print(f"✅ Forecast data loaded successfully (last updated: {forecast_last_modified.strftime('%Y-%m-%d %H:%M')})")
    else:
        print(f"⚠️ Warning: Forecast data file not found. Please run 7days.py to generate forecast data.")
        df_forecast = None
    
    # Clean up column names in training data
    if ' Population' in df_full.columns:
        df_full.rename(columns={' Population': 'Population'}, inplace=True)
    if 'Discharge (m³/s)' in df_full.columns:
        df_full.rename(columns={'Discharge (m³/s)': 'Discharge_m3s'}, inplace=True)

    df_full = df_full.replace("--", np.nan)
    df_full = df_full.fillna(df_full.median(numeric_only=True))
    
    # Prepare encoders for categorical columns
    cat_cols_full = ["Ward Code", "Land Use Classes", "Soil Type"]
    le_dict = {}
    for col in cat_cols_full:
        if col in df_full.columns:
            le_dict[col] = LabelEncoder().fit(df_full[col].astype(str).fillna("Unknown"))

    # Create a clean copy of the training data
    df_full_clean = df_full.copy()
    for col, le in le_dict.items():
        if col in df_full_clean.columns:
            df_full_clean[col] = le.transform(df_full_clean[col].astype(str).fillna("Unknown"))

    # Define features to drop
    features_to_drop = [
        "Flood-risk_level", "DATE", "true_conditions_count", 
        "Flood_occured", "Monitoring_required", "Drainage_properties", 
        "Drainage_line_id", "Nearest Station", "Areas"
    ]
    
    # Drop columns that exist
    drop_cols = [col for col in features_to_drop if col in df_full_clean.columns]
    df_full_clean = df_full_clean.drop(columns=drop_cols, errors='ignore')
    MODEL_FEATURES = getattr(scaler, 'feature_names_in_', None)
    if MODEL_FEATURES is None:
            MODEL_FEATURES = df_full_clean.columns.tolist()  # Fallback
    else:
         MODEL_FEATURES = MODEL_FEATURES.tolist()


    # Get list of areas for which we can make predictions
    if df_forecast is not None:
        AREAS = df_forecast['Area'].unique().tolist()
    else:
        AREAS = df_full['Areas'].unique().tolist() if 'Areas' in df_full.columns else []
    
    print(f"✅ Found {len(AREAS)} areas for prediction")

except FileNotFoundError as e:
    print(f"❌ Error: {e}. Please ensure data files are present.")
    exit()

# --- STEP 3: Define prediction function ---
def predict_risk(area_name, forecast_date=None):
    """Predicts flood risk for a given area using forecast data."""
    try:
        if df_forecast is None:
            raise Exception("No forecast data available. Please run 7days.py first.")
        
        # Get today's forecast or specified date's forecast
        if forecast_date:
            area_forecast = df_forecast[(df_forecast['Area'] == area_name) & 
                                        (df_forecast['Date'] == forecast_date)]
        else:
            # Get the most recent date from the forecast data
            today = datetime.now().strftime('%Y-%m-%d')
            area_forecast = df_forecast[(df_forecast['Area'] == area_name) & 
                                        (df_forecast['Date'] >= today)]
            
            if area_forecast.empty:
                # If no forecast for today or future, get the most recent one
                area_forecast = df_forecast[df_forecast['Area'] == area_name].sort_values('Date', ascending=False)
        
        if area_forecast.empty:
            raise Exception(f"No forecast data found for area '{area_name}'")
        
        # Take the first matching forecast
        forecast_row = area_forecast.iloc[0]
        forecast_date = forecast_row['Date']
        
        print(f"\n✅ Using forecast data for {area_name} on {forecast_date}")
        print(f"   Rainfall: {forecast_row['Rainfall_mm']:.1f} mm, Intensity: {forecast_row['Rainfall_Intensity_mm_hr']:.1f} mm/hr")
        
        # Create a simple input data dictionary with default values
        input_data = {feature: 0.0 for feature in MODEL_FEATURES}
        
        # Handle categorical columns
        for col in ["Ward Code", "Land Use Classes", "Soil Type"]:
            if col in MODEL_FEATURES and col in forecast_row:
                val = str(forecast_row[col]).strip()
                if col in le_dict and val in le_dict[col].classes_:
                    input_data[col] = le_dict[col].transform([val])[0]
                else:
                    input_data[col] = 0
        
        # Handle numeric features present in forecast
        for feature in MODEL_FEATURES:
            if feature in ["Ward Code", "Land Use Classes", "Soil Type"]:
                continue
            if feature in forecast_row:
                input_data[feature] = float(forecast_row[feature])
            elif feature == 'Rainfall (mm)' and 'Rainfall_mm' in forecast_row:
                input_data[feature] = float(forecast_row['Rainfall_mm'])
        
        # Create DataFrame with exact training features only
        input_df = pd.DataFrame([[input_data[feature] for feature in MODEL_FEATURES]], columns=MODEL_FEATURES)
        
        # Print input data for debugging
        print("\nUsing these features for prediction:")
        for feature in MODEL_FEATURES:
            print(f"  {feature}: {input_df[feature].iloc[0]}")
        
        # Scale the data
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction_numeric = ensemble_model.predict(scaled_input)
        prediction_label = target_encoder.inverse_transform(prediction_numeric)
        
        return prediction_label[0], forecast_date
        
    except Exception as e:
        import traceback
        print(f"❌ Could not get prediction for {area_name}: {str(e)}")
        traceback.print_exc()
        return "Prediction failed", None

# --- Main execution block ---
if __name__ == "__main__":
    if df_forecast is None:
        print("\n⚠️ Warning: No forecast data available. Run 7days.py first to generate forecasts.")
        exit()
        
    normalized_areas = {a.strip().lower(): a for a in AREAS}
    
    while True:
        area_input = input("\nEnter an area name to predict flood risk (or 'exit' to quit): ").strip()
        
        if area_input.lower() == 'exit':
            break
            
        if not area_input:
            print("Area name cannot be empty. Please try again.")
            continue

        match_result = fuzzy_process.extractOne(area_input.lower(), list(normalized_areas.keys()))
        
        if match_result and match_result[1] > 60:
            best_match_key = match_result[0]
            best_match = normalized_areas[best_match_key]
            
            print(f"\n🔎 Matched your input '{area_input}' with '{best_match}' (Confidence: {match_result[1]:.1f}%)")
            
            if df_forecast is not None:
                available_dates = df_forecast[df_forecast['Area'] == best_match]['Date'].unique().tolist()
                if available_dates:
                    print(f"Available forecast dates: {', '.join(available_dates)}")
                    date_input = input("Enter a forecast date or press Enter for latest: ").strip()
                    forecast_date = date_input if date_input in available_dates else None
                else:
                    forecast_date = None
            else:
                forecast_date = None
            
            predicted_risk, used_date = predict_risk(best_match, forecast_date)
            
            print(f"\n--- Flood Risk Report for {best_match} ---")
            if predicted_risk != "Prediction failed" and used_date:
                print(f"Date: {used_date}")
                print(f"Predicted Flood Risk: {predicted_risk}")
                
                if predicted_risk in ["Very High", "High"]:
                    print("\nRecommendations:")
                    print("- Avoid travel to this area if possible")
                    print("- Move vehicles to higher ground")
                    print("- Prepare for possible evacuation")
                elif predicted_risk == "Medium":
                    print("\nRecommendations:")
                    print("- Stay alert and monitor local news")
                    print("- Avoid basement areas")
                    print("- Have emergency supplies ready")
                else:
                    print("\nRecommendations:")
                    print("- Normal precautions for monsoon season")
            else:
                print(f"Predicted Flood Risk: Prediction failed")
            
            print("----------------------------------------")
        else:
            top_matches = fuzzy_process.extract(area_input.lower(), list(normalized_areas.keys()), limit=5)
            suggestions = [normalized_areas[m[0]] for m in top_matches]
            print(f"❌ Could not find a good matching area for '{area_input}'.")
            print(f"Did you mean one of these? {', '.join(suggestions)}")
