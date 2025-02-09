#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# File Paths
# -------------------------
# Use the surrogate predictions file produced by the "train once" evaluation
predictions_path = "surrogate_once_predictions.csv"

# -------------------------
# Load Data
# -------------------------
try:
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df)} rows from surrogate predictions data.")
except Exception as e:
    raise RuntimeError(f"Error reading surrogate predictions data from {predictions_path}: {e}")

# Ensure required columns exist
required_columns = ["lid_time_step", "velocity_magnitude", "pred_velocity", "pressure", "pred_pressure"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in data. Please check your CSV file.")

# -------------------------
# Plotting
# -------------------------
plt.figure(figsize=(14, 10))

# Subplot 1: Velocity Magnitude
plt.subplot(2, 1, 1)
plt.plot(df["lid_time_step"], df["velocity_magnitude"],
         label="Simulation Velocity", marker="o", linestyle="-", color="blue")
plt.plot(df["lid_time_step"], df["pred_velocity"],
         label="Surrogate Predicted Velocity", marker="x", linestyle="--", color="red")
plt.xlabel("Time Step")
plt.ylabel("Velocity Magnitude")
plt.title("Simulation vs. Surrogate (Trained Once): Velocity Magnitude")
plt.legend()
plt.grid(True)

# Subplot 2: Pressure
plt.subplot(2, 1, 2)
plt.plot(df["lid_time_step"], df["pressure"],
         label="Simulation Pressure", marker="o", linestyle="-", color="green")
plt.plot(df["lid_time_step"], df["pred_pressure"],
         label="Surrogate Predicted Pressure", marker="x", linestyle="--", color="purple")
plt.xlabel("Time Step")
plt.ylabel("Pressure")
plt.title("Simulation vs. Surrogate (Trained Once): Pressure")
plt.legend()
plt.grid(True)

plt.tight_layout()
output_plot_file = "surrogate_once_vs_simulation.png"
plt.savefig(output_plot_file)
print(f"Plot saved as {output_plot_file}")
plt.show()
