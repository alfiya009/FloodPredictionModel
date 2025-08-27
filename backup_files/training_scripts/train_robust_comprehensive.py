import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("🚀 Training with Robust Real Dataset Processing...")
print("=" * 60)

# Load the comprehensive CSV data
df = pd.read_csv("final_flood_classification data.csv")
print(f"✅ Loaded comprehensive CSV: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 All columns: {list(df.columns)}")

# Check target distribution
print(f"\n🎯 Target Distribution:")
print(df['Flood-risk_level'].value_counts())

# Data preprocessing
print("\n🔧 Robust Data Preprocessing...")

# Drop problematic columns
columns_to_drop = ['DATE', 'Drainage_properties', 'Drainage_line_id']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)
        print(f"✅ Dropped {col}")

# Select only the most important numeric features for reliable training
important_features = [
    'Latitude', 'Longitude', 'Rainfall_mm', 'Discharge_m3s', 'Elevation', 
    'Population', 'Road Density_m', 'Runoff equivalent', 'Soil Wetness Index',
    'Rainfall_Intensity_mm_hr', 'Rainfall Days Count', 'Longest rainfall _days',
    'Distance_to_water_m', 'Built_up%', 'True_nearest_distance_m', 'true_conditions_count'
]

# Keep only important features and target
keep_columns = important_features + ['Flood-risk_level']
df_clean = df[keep_columns].copy()

print(f"🔧 Selected {len(important_features)} important numeric features")

# Handle missing values
df_clean = df_clean.fillna(df_clean.median())

# Prepare features and target
X = df_clean[important_features].copy()
y = df_clean['Flood-risk_level']

print(f"🔧 Final features: {X.shape[1]} features")
print(f"📊 Feature names: {list(X.columns)}")

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"🎯 Target encoded: {le.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train individual models
print("\n🤖 Training Individual Models...")

# 1. Random Forest with optimized parameters
rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=20, 
    min_samples_split=5, 
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)
print(f"✅ Random Forest Accuracy: {rf_score:.4f}")

# 2. SVM with optimized parameters
svm = SVC(
    kernel='rbf', 
    probability=True, 
    C=15, 
    gamma='scale', 
    random_state=42
)
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
print("\n💾 Saving Models...")
joblib.dump(ensemble, 'robust_comprehensive_ensemble.joblib')
joblib.dump(rf, 'robust_comprehensive_rf.joblib')
joblib.dump(svm, 'robust_comprehensive_svm.joblib')
joblib.dump(scaler, 'robust_comprehensive_scaler.joblib')
joblib.dump(le, 'robust_comprehensive_label_encoder.joblib')

print("✅ All robust comprehensive models saved successfully!")
print(f"\n📊 Final Results:")
print(f"Random Forest: {rf_score:.4f}")
print(f"SVM: {svm_score:.4f}")
print(f"Ensemble: {ensemble_score:.4f}")

# Test prediction on sample data
print(f"\n🧪 Sample Prediction Test...")
# Use first row of test data for prediction
sample_data = X_test_scaled[0:1]
prediction = ensemble.predict(sample_data)
probability = ensemble.predict_proba(sample_data)
risk_level = le.inverse_transform(prediction)[0]

print(f"Sample Input: First test sample")
print(f"Predicted Risk: {risk_level}")
print(f"Confidence: {np.max(probability)*100:.2f}%")
print(f"All Probabilities: {dict(zip(le.classes_, probability[0]))}")

# Feature importance for Random Forest
feature_importance = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

print(f"\n🔍 Top 10 Most Important Features:")
print(importance_df.head(10))

print("\n🎉 Robust comprehensive training completed successfully!")
print("📁 Files created:")
print("   - robust_comprehensive_ensemble.joblib")
print("   - robust_comprehensive_rf.joblib")
print("   - robust_comprehensive_svm.joblib")
print("   - robust_comprehensive_scaler.joblib")
print("   - robust_comprehensive_label_encoder.joblib")
