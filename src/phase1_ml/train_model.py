# ============================================================
# EXERCISE 5.2 — MACHINE LEARNING MODEL TRAINING
# ============================================================

import pandas as pd
import numpy as np

print("Libraries loaded successfully!")


# ------------------------------------------------------------
# LOAD LABELLED DATASET
# ------------------------------------------------------------

data_file = "data/processed/labelled_sites.csv"

df = pd.read_csv(data_file)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

# ============================================================
# STEP 2 — SELECT FEATURES AND TARGET
# ============================================================

# Features used for machine learning
features = [
    'dist_road_m',
    'dist_hospital_m',
    'flood_risk'
]

# Target variable
target = 'label'

# Create X and y
X = df[features]
y = df[target]

print("\nFeatures selected:")
print(features)

print("\nTarget variable:")
print(target)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nClass distribution:")
print(y.value_counts())

# ============================================================
# STEP 3 — TRAIN / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split completed!")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ============================================================
# STEP 4 — RANDOM FOREST MODEL
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Predict on test data
rf_predictions = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:", rf_accuracy)

# Classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# ============================================================
# STEP 5 — XGBOOST MODEL
# ============================================================

from xgboost import XGBClassifier

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)

# Train the model
xgb_model.fit(X_train, y_train)

print("\nXGBoost training completed!")

# Predict on test data
xgb_predictions = xgb_model.predict(X_test)

# Calculate accuracy
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

print("\nXGBoost Accuracy:", xgb_accuracy)

# Classification report
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))

# ============================================================
# STEP 6 — COMPARE RANDOM FOREST AND XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Random Forest Accuracy : {rf_accuracy:.4f}")
print(f"XGBoost Accuracy       : {xgb_accuracy:.4f}")

# Select the better model
if rf_accuracy >= xgb_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_accuracy
else:
    best_model = xgb_model
    best_model_name = "XGBoost"
    best_accuracy = xgb_accuracy

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy, 4))

# ============================================================
# STEP 7 — SHAP ANALYSIS
# ============================================================

import shap
import matplotlib.pyplot as plt
import os

print("\nStarting SHAP analysis...")

# Create SHAP explainer for the best model
explainer = shap.TreeExplainer(best_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

print("SHAP values calculated successfully!")

# Create output folder if it does not exist
os.makedirs("outputs/reports", exist_ok=True)

# SHAP summary plot
plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.title("SHAP Feature Importance - Random Forest")
plt.tight_layout()

# Save figure
shap_output = "outputs/reports/shap_importance.png"

plt.savefig(
    shap_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP plot saved successfully!")
print(shap_output)

# ============================================================
# SAVE BEST MODEL
# ============================================================

import joblib
import os

# Create model folder if it does not exist
os.makedirs("models/saved", exist_ok=True)

# Save the selected best model
model_output = "models/saved/site_scorer_model.pkl"

joblib.dump(best_model, model_output)

print("Model saved successfully!")
print(model_output)