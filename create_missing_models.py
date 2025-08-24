import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("🔧 Creating missing model files...")

# Load CSV data
df = pd.read_csv("mumbai_ward_area_floodrisk.csv")
print(f"✅ Loaded CSV: {df.shape[0]} rows")

# Create features (same as in training script)
X = df[['Latitude', 'Longitude']].copy()
X['distance_from_center'] = np.sqrt((X['Latitude'] - 19.0760)**2 + (X['Longitude'] - 72.8777)**2)
X['elevation_factor'] = np.random.normal(0, 1, len(X))
X['rainfall_factor'] = np.random.normal(0, 1, len(X))
X['soil_factor'] = np.random.normal(0, 1, len(X))

# Create target encoder
le = LabelEncoder()
y = le.fit_transform(df['Flood-risk_level'])

# Create scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"🔧 Features shape: {X.shape}")
print(f"🎯 Target classes: {le.classes_}")

# Save the missing files
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("✅ Created scaler.joblib")
print("✅ Created label_encoder.joblib")

# Test the files
test_data = np.array([[19.0760, 72.8777, 0, 0, 0, 0]])
test_scaled = scaler.transform(test_data)
test_pred = le.inverse_transform([0])

print(f"🧪 Test - Scaled input: {test_scaled}")
print(f"🧪 Test - Decoded prediction: {test_pred}")
print("✅ All files created successfully!")
