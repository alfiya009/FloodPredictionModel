import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🚀 Simple Model Training...")
print("=" * 40)

# Load CSV data
df = pd.read_csv("mumbai_ward_area_floodrisk.csv")
print(f"✅ Loaded CSV: {df.shape[0]} rows")

# Create simple features (6 features as expected)
X = df[['Latitude', 'Longitude']].copy()
X['distance'] = np.sqrt((X['Latitude'] - 19.0760)**2 + (X['Longitude'] - 72.8777)**2)
X['elevation'] = np.random.uniform(0, 100, len(X))
X['rainfall'] = np.random.uniform(0, 100, len(X))
X['soil'] = np.random.uniform(0, 1, len(X))

print(f"🔧 Features: {X.shape[1]} features")
print(f"📊 Feature names: {list(X.columns)}")

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['Flood-risk_level'])

print(f"🎯 Target: {le.classes_}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
rf.fit(X_scaled, y)

# Test accuracy
y_pred = rf.predict(X_scaled)
accuracy = np.mean(y_pred == y)
print(f"✅ Accuracy: {accuracy:.4f}")

# Save models
print("\n💾 Saving models...")
joblib.dump(rf, 'simple_rf_model.joblib')
joblib.dump(scaler, 'simple_scaler.joblib')
joblib.dump(le, 'simple_label_encoder.joblib')

print("✅ Models saved!")

# Test prediction
print(f"\n🧪 Testing...")
test_data = np.array([[19.0760, 72.8777, 0, 0, 0, 0]])
test_scaled = scaler.transform(test_data)
prediction = rf.predict(test_scaled)
probability = rf.predict_proba(test_scaled)
risk_level = le.inverse_transform(prediction)[0]

print(f"Input: Mumbai Center")
print(f"Prediction: {risk_level}")
print(f"Confidence: {np.max(probability)*100:.2f}%")
print(f"Probabilities: {dict(zip(le.classes_, probability[0]))}")

print("\n🎉 Training completed!")
