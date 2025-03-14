#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------
# Parse Command‑Line Arguments
# -------------------------
parser = argparse.ArgumentParser(
    description="Train surrogate model using MHD Hartmann simulation output data."
)
parser.add_argument("--initial", type=int, default=100,
                    help="Number of initial samples to use for training (default: 100)")
parser.add_argument("--data_path", type=str, default="/home/musacim/simulation/openfoam/mhd/hartmann_simulation_output_data.csv",
                    help="Path to the simulation output CSV file.")
args = parser.parse_args()

INITIAL_TRAIN_SIZE = args.initial
DATA_PATH = args.data_path

# -------------------------
# Load Simulation Data
# -------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows of simulation output data.")

# -------------------------
# Define Features and Targets
# -------------------------
# For the Hartmann case, we use inlet_velocity and B_magnitude as features.
features = ["inlet_velocity", "B_magnitude"]
targets = ["velocity_magnitude", "pressure"]

# -------------------------
# Train the Surrogate Model on Initial Data
# -------------------------
train_df = df.iloc[:INITIAL_TRAIN_SIZE].copy()
X_train = train_df[features].values
y_train = train_df[targets].values

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"Trained surrogate model on the first {INITIAL_TRAIN_SIZE} samples.")

# Save the surrogate model for later use
model_filename = "surrogate_model_rf.joblib"
joblib.dump(model, model_filename)
print(f"Saved surrogate model as '{model_filename}'.")

# -------------------------
# Evaluate the Model on the Entire Dataset
# -------------------------
X_all = df[features].values
y_all = df[targets].values
y_pred = model.predict(X_all)

mse_velocity = mean_squared_error(y_all[:, 0], y_pred[:, 0])
mse_pressure = mean_squared_error(y_all[:, 1], y_pred[:, 1])
r2_velocity = r2_score(y_all[:, 0], y_pred[:, 0])
r2_pressure = r2_score(y_all[:, 1], y_pred[:, 1])

print("\nPerformance on Entire Dataset:")
print(f"Velocity Magnitude - MSE: {mse_velocity:.6f}, R²: {r2_velocity:.4f}")
print(f"Pressure           - MSE: {mse_pressure:.6f}, R²: {r2_pressure:.4f}")

# -------------------------
# Save Predictions to CSV (Optional)
# -------------------------
results_df = df.copy()
results_df["pred_velocity"] = y_pred[:, 0]
results_df["pred_pressure"] = y_pred[:, 1]
results_df.to_csv("surrogate_once_predictions.csv", index=False)
print("Saved predictions to 'surrogate_once_predictions.csv'.")

# -------------------------
# Plot Simulation Outputs vs. Surrogate Predictions
# -------------------------
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(results_df["time_step"], results_df["velocity_magnitude"],
         label="Simulation Velocity", marker="o", linestyle="-", color="blue")
plt.plot(results_df["time_step"], results_df["pred_velocity"],
         label="Surrogate Predicted Velocity", marker="x", linestyle="--", color="red")
plt.xlabel("Time Step")
plt.ylabel("Velocity Magnitude")
plt.title("Simulation vs. Surrogate: Velocity Magnitude")
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(results_df["time_step"], results_df["pressure"],
         label="Simulation Pressure", marker="o", linestyle="-", color="green")
plt.plot(results_df["time_step"], results_df["pred_pressure"],
         label="Surrogate Predicted Pressure", marker="x", linestyle="--", color="purple")
plt.xlabel("Time Step")
plt.ylabel("Pressure")
plt.title("Simulation vs. Surrogate: Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("surrogate_once_performance.png")
print("Saved plot as 'surrogate_once_performance.png'.")
plt.show()
