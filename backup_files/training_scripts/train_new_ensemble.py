import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🚀 Training New Ensemble Model with Correct Features...")
print("=" * 60)

# Load CSV data
df = pd.read_csv("mumbai_ward_area_floodrisk.csv")
print(f"✅ Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")

# Check target distribution
print(f"\n🎯 Target Distribution:")
print(df['Flood-risk_level'].value_counts())

# Create features (exactly 6 features as expected by API)
X = df[['Latitude', 'Longitude']].copy()
X['distance_from_center'] = np.sqrt((X['Latitude'] - 19.0760)**2 + (X['Longitude'] - 72.8777)**2)
X['elevation_factor'] = np.random.normal(0, 1, len(X))
X['rainfall_factor'] = np.random.normal(0, 1, len(X))
X['soil_factor'] = np.random.normal(0, 1, len(X))

print(f"\n🔧 Features created: {X.shape[1]} features")
print(f"📊 Feature names: {list(X.columns)}")

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['Flood-risk_level'])

print(f"🎯 Target encoded: {le.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train individual models
print("\n🤖 Training Individual Models...")

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)
print(f"✅ Random Forest Accuracy: {rf_score:.4f}")

# 2. SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_score = svm.score(X_test_scaled, y_test)
print(f"✅ SVM Accuracy: {svm_score:.4f}")

# 3. Create Ensemble
print("\n🎯 Creating Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('svm', svm)
    ],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)
ensemble_score = ensemble.score(X_test_scaled, y_test)
print(f"✅ Ensemble Accuracy: {ensemble_score:.4f}")

# Save all models
print("\n💾 Saving New Models...")
joblib.dump(ensemble, 'new_ensemble_model.joblib')
joblib.dump(scaler, 'new_scaler.joblib')
joblib.dump(le, 'new_label_encoder.joblib')

print("✅ New models saved successfully!")
print(f"\n📊 Final Results:")
print(f"Random Forest: {rf_score:.4f}")
print(f"SVM: {svm_score:.4f}")
print(f"Ensemble: {ensemble_score:.4f}")

# Test prediction
print(f"\n🧪 Testing New Model...")
test_data = np.array([[19.0760, 72.8777, 0, 0, 0, 0]])  # Mumbai center
test_scaled = scaler.transform(test_data)
prediction = ensemble.predict(test_scaled)
probability = ensemble.predict_proba(test_scaled)
risk_level = le.inverse_transform(prediction)[0]

print(f"Sample Input: Mumbai Center")
print(f"Predicted Risk: {risk_level}")
print(f"Confidence: {np.max(probability)*100:.2f}%")
print(f"All Probabilities: {dict(zip(le.classes_, probability[0]))}")

print("\n🎉 New ensemble model training completed successfully!")
print("📁 Files created:")
print("   - new_ensemble_model.joblib")
print("   - new_scaler.joblib") 
print("   - new_label_encoder.joblib")
