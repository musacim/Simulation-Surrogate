#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# -------------------------
# Configuration Parameters
# -------------------------
INITIAL_TRAIN_SIZE = 100         # Number of initial samples used to train the surrogate model
DRIFT_THRESHOLD_VEL = 0.02       # Acceptable error threshold for velocity magnitude (adjust as needed)
DRIFT_THRESHOLD_PRES = 0.02      # Acceptable error threshold for pressure (adjust as needed)

# File path for the simulation output data (ground truth)
DATA_PATH = "/home/musacim/simulation/openfoam/simulation_output_data.csv"

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows of simulation output data.")

# Define feature and target columns
# Features are the inputs: lid_velocity and viscosity.
# Targets are the outputs: velocity_magnitude and pressure.
features = ["lid_velocity", "viscosity"]
targets = ["velocity_magnitude", "pressure"]

# -------------------------
# Train the Initial Surrogate Model
# -------------------------
# Use the first INITIAL_TRAIN_SIZE samples for initial training.
train_df = df.iloc[:INITIAL_TRAIN_SIZE].copy()
X_train = train_df[features].values
y_train = train_df[targets].values

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"Trained initial surrogate model on the first {INITIAL_TRAIN_SIZE} samples.")

# -------------------------
# Online Prediction and Adaptive Retraining Loop
# -------------------------
# We will iterate through the remaining samples one-by-one, simulating an online process.
# For each new sample, we use the current surrogate model to predict the outputs.
# If the prediction error exceeds the defined thresholds, we trigger a retraining event.
predictions = []  # To store predictions and errors for later analysis
drift_events = []  # To log indices where drift was detected

# Process samples from INITIAL_TRAIN_SIZE to end
for idx in range(INITIAL_TRAIN_SIZE, len(df)):
    # Get the current sample (simulate new incoming data)
    sample = df.iloc[idx]
    # Reshape feature vector for prediction (scikit-learn expects a 2D array)
    X_sample = sample[features].values.reshape(1, -1)
    y_actual = sample[targets].values.reshape(1, -1)
    
    # Make prediction with the current surrogate model
    y_pred = model.predict(X_sample)
    
    # Compute absolute errors for both outputs
    error_velocity = abs(y_pred[0, 0] - y_actual[0, 0])
    error_pressure = abs(y_pred[0, 1] - y_actual[0, 1])
    
    # Save the results (using lid_time_step if available; otherwise, use the index)
    time_step = sample.get("lid_time_step", sample.get("time_step", idx))
    predictions.append({
        "time_step": time_step,
        "pred_velocity": y_pred[0, 0],
        "actual_velocity": y_actual[0, 0],
        "pred_pressure": y_pred[0, 1],
        "actual_pressure": y_actual[0, 1],
        "error_velocity": error_velocity,
        "error_pressure": error_pressure
    })
    
    # Check if the error exceeds our thresholds to indicate data drift
    if error_velocity > DRIFT_THRESHOLD_VEL or error_pressure > DRIFT_THRESHOLD_PRES:
        print(f"Drift detected at sample index {idx} (time_step={time_step}): "
              f"Velocity error = {error_velocity:.4f}, Pressure error = {error_pressure:.4f}")
        drift_events.append(idx)
        
        # Retrain the surrogate model using all available data up to (and including) the current sample.
        # (Alternatively, a sliding window strategy can be used.)
        new_train_df = df.iloc[:idx + 1].copy()
        X_new_train = new_train_df[features].values
        y_new_train = new_train_df[targets].values
        
        model.fit(X_new_train, y_new_train)
        print(f"Retrained surrogate model with {idx + 1} samples.")
    else:
        print(f"Sample {idx} (time_step={time_step}): No drift detected. "
              f"Velocity error = {error_velocity:.4f}, Pressure error = {error_pressure:.4f}")

# -------------------------
# Save the Final Surrogate Model and Results
# -------------------------
MODEL_SAVE_PATH = "adaptive_surrogate_model.joblib"
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Final surrogate model saved as '{MODEL_SAVE_PATH}'.")

# Save prediction results to a CSV for further analysis
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("adaptive_predictions.csv", index=False)
print("Prediction results saved to 'adaptive_predictions.csv'.")

# Optionally, save drift event indices
drift_df = pd.DataFrame({"drift_index": drift_events})
drift_df.to_csv("drift_events.csv", index=False)
print("Drift events saved to 'drift_events.csv'.")
