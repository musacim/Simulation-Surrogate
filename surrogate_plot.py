#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# File Paths
# -------------------------
# Simulation output data (ground truth)
sim_data_path = "/home/musacim/simulation/openfoam/simulation_output_data.csv"
# Surrogate predictions data from adaptive retraining
predictions_path = "adaptive_predictions.csv"  # adjust path if necessary

# -------------------------
# Load Data
# -------------------------
try:
    sim_df = pd.read_csv(sim_data_path)
    print(f"Loaded {len(sim_df)} rows from simulation output data.")
except Exception as e:
    raise RuntimeError(f"Error reading simulation output data from {sim_data_path}: {e}")

try:
    pred_df = pd.read_csv(predictions_path)
    print(f"Loaded {len(pred_df)} rows from surrogate predictions data.")
except Exception as e:
    raise RuntimeError(f"Error reading surrogate predictions data from {predictions_path}: {e}")

# -------------------------
# Merge Data on Time Step
# -------------------------
# The simulation output data has a column "lid_time_step"
# and the surrogate predictions data has a column "time_step".
merged_df = pd.merge(sim_df, pred_df, left_on="lid_time_step", right_on="time_step", how="inner")
print(f"Merged data contains {len(merged_df)} rows.")

# -------------------------
# Plotting
# -------------------------
plt.figure(figsize=(14, 10))

# Subplot 1: Velocity Magnitude
plt.subplot(2, 1, 1)
plt.plot(merged_df["lid_time_step"], merged_df["velocity_magnitude"],
         label="Simulation Velocity", marker="o", linestyle="-", color="blue")
plt.plot(merged_df["lid_time_step"], merged_df["pred_velocity"],
         label="Surrogate Predicted Velocity", marker="x", linestyle="--", color="red")
plt.xlabel("Time Step")
plt.ylabel("Velocity Magnitude")
plt.title("Simulation vs. Surrogate: Velocity Magnitude")
plt.legend()
plt.grid(True)

# Subplot 2: Pressure
plt.subplot(2, 1, 2)
plt.plot(merged_df["lid_time_step"], merged_df["pressure"],
         label="Simulation Pressure", marker="o", linestyle="-", color="green")
plt.plot(merged_df["lid_time_step"], merged_df["pred_pressure"],
         label="Surrogate Predicted Pressure", marker="x", linestyle="--", color="purple")
plt.xlabel("Time Step")
plt.ylabel("Pressure")
plt.title("Simulation vs. Surrogate: Pressure")
plt.legend()
plt.grid(True)

plt.tight_layout()
output_plot_file = "surrogate_vs_simulation.png"
plt.savefig(output_plot_file)
print(f"Plot saved as {output_plot_file}")
plt.show()
