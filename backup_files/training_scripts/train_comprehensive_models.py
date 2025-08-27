import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

print("🚀 Starting Comprehensive Model Training...")
print("=" * 50)

# Load the CSV data
CSV_PATH = "mumbai_ward_area_floodrisk.csv"
df = pd.read_csv(CSV_PATH)

print(f"✅ Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 Columns: {list(df.columns)}")

# Check target distribution
print(f"\n🎯 Target Distribution:")
print(df['Flood-risk_level'].value_counts())

# Prepare features
# We'll use coordinates and create synthetic features for demonstration
X = df[['Latitude', 'Longitude']].copy()

# Create synthetic features based on coordinates (simulating real-world data)
X['distance_from_center'] = np.sqrt((X['Latitude'] - 19.0760)**2 + (X['Longitude'] - 72.8777)**2)
X['elevation_factor'] = np.random.normal(0, 1, len(X))  # Simulated elevation
X['rainfall_factor'] = np.random.normal(0, 1, len(X))   # Simulated rainfall
X['soil_factor'] = np.random.normal(0, 1, len(X))       # Simulated soil type

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['Flood-risk_level'])

print(f"\n🔧 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# 1. Train SVM
print("\n🤖 Training SVM...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_score = svm.score(X_test_scaled, y_test)
print(f"✅ SVM Accuracy: {svm_score:.4f}")

# 2. Train Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)
print(f"✅ Random Forest Accuracy: {rf_score:.4f}")

# 3. Train XGBoost
print("\n🚀 Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_score = xgb.score(X_test_scaled, y_test)
print(f"✅ XGBoost Accuracy: {xgb_score:.4f}")

# 4. Create Ensemble
print("\n🎯 Creating Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)
ensemble_score = ensemble.score(X_test_scaled, y_test)
print(f"✅ Ensemble Accuracy: {ensemble_score:.4f}")

# Save all models
print("\n💾 Saving Models...")
joblib.dump(svm, 'svm_model.joblib')
joblib.dump(rf, 'random_forest_model.joblib')
joblib.dump(xgb, 'xgboost_model.joblib')
joblib.dump(ensemble, 'ensemble_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("✅ All models saved successfully!")
print(f"\n📊 Final Results:")
print(f"SVM: {svm_score:.4f}")
print(f"Random Forest: {rf_score:.4f}")
print(f"XGBoost: {xgb_score:.4f}")
print(f"Ensemble: {ensemble_score:.4f}")

# Test prediction on sample data
print(f"\n🧪 Sample Prediction Test:")
sample_data = np.array([[19.0760, 72.8777, 0, 0, 0, 0]])  # Mumbai center
sample_scaled = scaler.transform(sample_data)
prediction = ensemble.predict(sample_scaled)
probability = ensemble.predict_proba(sample_scaled)
risk_level = le.inverse_transform(prediction)[0]

print(f"Sample Input: Mumbai Center")
print(f"Predicted Risk: {risk_level}")
print(f"Confidence: {np.max(probability)*100:.2f}%")
print(f"All Probabilities: {dict(zip(le.classes_, probability[0]))}")
