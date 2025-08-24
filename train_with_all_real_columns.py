import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("🚀 Training with ALL Real Dataset Columns...")
print("=" * 60)

# Load the comprehensive CSV data
df = pd.read_csv("final_flood_classification data.csv")
print(f"✅ Loaded comprehensive CSV: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 All columns: {list(df.columns)}")

# Check target distribution
print(f"\n🎯 Target Distribution:")
print(df['Flood-risk_level'].value_counts())

# Data preprocessing
print("\n🔧 Data Preprocessing...")

# Drop DATE column as it's not useful for prediction
if 'DATE' in df.columns:
    df = df.drop('DATE', axis=1)
    print("✅ Dropped DATE column")

# Handle missing values - only for numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Handle all categorical columns (including Yes/No columns)
categorical_columns = ['Ward Code', 'Areas', 'Nearest Station', 'Land Use Classes', 'Soil Type', 
                      'Flood_occured', 'Monitoring_required', 'Drainage_properties']

for col in categorical_columns:
    if col in df.columns:
        # Fill missing values with mode
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        # Convert to numeric codes
        df[col] = df[col].astype('category').cat.codes

# Select features (exclude target)
feature_columns = [col for col in df.columns if col != 'Flood-risk_level']
X = df[feature_columns].copy()

print(f"🔧 Features selected: {X.shape[1]} features")
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

# 1. Random Forest with more trees for better accuracy
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)
print(f"✅ Random Forest Accuracy: {rf_score:.4f}")

# 2. SVM with better parameters
svm = SVC(kernel='rbf', probability=True, C=10, gamma='scale', random_state=42)
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
joblib.dump(ensemble, 'comprehensive_ensemble.joblib')
joblib.dump(rf, 'comprehensive_rf.joblib')
joblib.dump(svm, 'comprehensive_svm.joblib')
joblib.dump(scaler, 'comprehensive_scaler.joblib')
joblib.dump(le, 'comprehensive_label_encoder.joblib')

print("✅ All comprehensive models saved successfully!")
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

print("\n🎉 Comprehensive training completed successfully!")
print("📁 Files created:")
print("   - comprehensive_ensemble.joblib")
print("   - comprehensive_rf.joblib")
print("   - comprehensive_svm.joblib")
print("   - comprehensive_scaler.joblib")
print("   - comprehensive_label_encoder.joblib")
