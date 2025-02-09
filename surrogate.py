#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # for saving the trained model

# -------------------------
# File Paths & Configuration
# -------------------------
# Simulation output CSV: columns: lid_velocity, viscosity, lid_time_step, velocity_magnitude, pressure
output_data_path = "/home/musacim/simulation/openfoam/simulation_output_data.csv"

# Random seed for reproducibility
RANDOM_SEED = 42

# Model save path (optional)
model_save_path = "surrogate_model_rf.joblib"

# -------------------------
# Load Data
# -------------------------
try:
    df = pd.read_csv(output_data_path)
except Exception as e:
    raise RuntimeError(f"Error reading simulation output data file at {output_data_path}: {e}")

print(f"Loaded {len(df)} rows of simulation output data.")

# -------------------------
# Define Features and Targets
# -------------------------
# Here we use the simulation input parameters as features:
#   - lid_velocity, viscosity
# and we predict the simulation outputs:
#   - velocity_magnitude, pressure
features = ["lid_velocity", "viscosity"]
targets = ["velocity_magnitude", "pressure"]

# Optionally, inspect a few rows
print("Data preview:")
print(df[features + targets].head())

X = df[features].values   # shape: (n_samples, 2)
y = df[targets].values    # shape: (n_samples, 2)

# -------------------------
# Split Data into Train and Test Sets
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# -------------------------
# Train a Multi-output Random Forest Regressor
# -------------------------
model = RandomForestRegressor(
    n_estimators=100,    # you can experiment with more trees or tune this parameter
    random_state=RANDOM_SEED,
    n_jobs=-1            # use all available cores
)

# Fit the model
print("Training the surrogate model...")
model.fit(X_train, y_train)
print("Training complete.")

# -------------------------
# Evaluate the Model
# -------------------------
# Predict on both training and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics for each target separately
def print_metrics(true, pred, target_name):
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    print(f"{target_name} - MSE: {mse:.6f}, R²: {r2:.4f}")

print("\nPerformance on Training Data:")
for i, target_name in enumerate(targets):
    print_metrics(y_train[:, i], y_train_pred[:, i], target_name)

print("\nPerformance on Test Data:")
for i, target_name in enumerate(targets):
    print_metrics(y_test[:, i], y_test_pred[:, i], target_name)

# -------------------------
# Save the Model for Future Use
# -------------------------
joblib.dump(model, model_save_path)
print(f"\nSurrogate model saved to {model_save_path}")
