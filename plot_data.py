#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------
# File Paths
# -------------------------
input_data_path = "/home/musacim/simulation/openfoam/cavity_simulations/simulation_data_wdrift.csv"
output_data_path = "/home/musacim/simulation/openfoam/simulation_output_data.csv"

# -------------------------
# Load Data
# -------------------------
# Input CSV columns: time_step, lid_velocity, viscosity, simulation_time_sec
try:
    input_df = pd.read_csv(input_data_path)
except Exception as e:
    raise RuntimeError(f"Error reading input data file at {input_data_path}: {e}")

# Output CSV columns: lid_velocity, viscosity, lid_time_step, velocity_magnitude, pressure
try:
    output_df = pd.read_csv(output_data_path)
except Exception as e:
    raise RuntimeError(f"Error reading simulation output data file at {output_data_path}: {e}")

print(f"Loaded {len(input_df)} rows of input data and {len(output_df)} rows of output data.")

# -------------------------
# Detect Drift in Input Data
# -------------------------
# We assume that under stable conditions the lid_velocity is around 1.5 with small variations.
# When a drift event occurs, the simulation code adds an offset (e.g., +0.5),
# so values above a chosen threshold (e.g., 1.8) indicate drift.
drift_threshold = 1.8
input_df["is_drift"] = input_df["lid_velocity"] > drift_threshold

# -------------------------
# Create Plots
# -------------------------
fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

# --- Plot 1: Input Parameters vs. Time Step ---
axs[0].plot(input_df["time_step"], input_df["lid_velocity"], label="Lid Velocity", 
            color="blue", marker="o", linestyle="-")
axs[0].plot(input_df["time_step"], input_df["viscosity"], label="Viscosity", 
            color="green", marker="x", linestyle="-")
axs[0].set_ylabel("Parameter Value")
axs[0].set_title("Input Parameters vs. Time Step")
axs[0].grid(True)

# Highlight drift regions on the lid_velocity plot.
drift_steps = input_df[input_df["is_drift"]]["time_step"].values
if drift_steps.size > 0:
    # Group consecutive time steps (if drift lasts multiple steps)
    groups = np.split(drift_steps, np.where(np.diff(drift_steps) != 1)[0] + 1)
    for group in groups:
        start = group[0]
        end = group[-1]
        axs[0].axvspan(start, end, color="red", alpha=0.3, label="Drift Region")
    # Remove duplicate legend entries
    handles, labels = axs[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs[0].legend(unique.values(), unique.keys())
else:
    axs[0].legend()

# --- Plot 2: Simulation Output vs. Time Step ---
# Here we use the output file's "lid_time_step" as the x-axis.
axs[1].plot(output_df["lid_time_step"], output_df["velocity_magnitude"], 
            label="Velocity Magnitude", color="purple", marker="s", linestyle="-")
axs[1].plot(output_df["lid_time_step"], output_df["pressure"], 
            label="Pressure", color="orange", marker="^", linestyle="-")
axs[1].set_ylabel("Output Value")
axs[1].set_title("Simulation Output vs. Time Step")
axs[1].legend()
axs[1].grid(True)

# --- Plot 3: Scatter Plot – Input vs. Output ---
# Merge the input and output data on time step (input_df.time_step and output_df.lid_time_step)
merged_df = pd.merge(input_df, output_df, left_on="time_step", right_on="lid_time_step", how="inner")
# Use input lid_velocity (from input data) for the x-axis.
# Note: after the merge, the input lid_velocity is named "lid_velocity_x" (and the output is "lid_velocity_y").
colors = merged_df["is_drift"].map({True: "red", False: "blue"})
axs[2].scatter(merged_df["lid_velocity_x"], merged_df["velocity_magnitude"], 
               c=colors, s=60, alpha=0.7)
axs[2].set_xlabel("Input Lid Velocity")
axs[2].set_ylabel("Output Velocity Magnitude")
axs[2].set_title("Input vs. Output: Lid Velocity vs. Velocity Magnitude")
axs[2].grid(True)
# Create a custom legend for drift indication.
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Stable',
                          markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Drift',
                          markerfacecolor='red', markersize=10)]
axs[2].legend(handles=legend_elements)

# Set a common x-axis label and adjust layout.
plt.xlabel("Time Step")
plt.tight_layout()

# Save and display the plot
output_plot_file = "input_output_drift_plot.png"
plt.savefig(output_plot_file)
print(f"Plot saved as {output_plot_file}")
plt.show()
