#!/usr/bin/env python3
"""
plot_input_output_with_timestep.py

This script reads a ground-truth CSV file (by default:
    /home/musacim/simulation/openfoam/simulation_output_data.csv)
which contains simulation inputs and outputs along with a time step column.
It then creates time-series plots that display:
  - Lid Velocity (input) and Velocity Magnitude (output) versus time step.
  - Viscosity (input) and Pressure (output) versus time step (using twin y-axes).

Usage:
    python3 plot_input_output_with_timestep.py [--csv PATH_TO_CSV]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot simulation inputs and outputs versus time step."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/musacim/simulation/openfoam/simulation_output_data.csv",
        help="Path to the ground truth CSV file."
    )
    args = parser.parse_args()

    # Load the CSV file
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV file '{args.csv}': {e}")
        return

    # Ensure expected columns are present
    expected_columns = [
        "lid_time_step",     # time step index
        "lid_velocity",      # input parameter
        "viscosity",         # input parameter
        "velocity_magnitude",# simulation output
        "pressure"           # simulation output
    ]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"The following expected columns are missing in the CSV file: {missing}")

    # Create a figure with two subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # --- Subplot 1: Lid Velocity (input) vs. Velocity Magnitude (output) ---
    ax1.plot(df["lid_time_step"], df["lid_velocity"],
             label="Lid Velocity (Input)",
             color="blue", marker="o", linestyle="-", markersize=3)
    ax1.plot(df["lid_time_step"], df["velocity_magnitude"],
             label="Velocity Magnitude (Output)",
             color="red", marker="x", linestyle="--", markersize=3)
    ax1.set_ylabel("Velocity")
    ax1.set_title("Lid Velocity & Velocity Magnitude vs. Time Step")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # --- Subplot 2: Viscosity (input) vs. Pressure (output) ---
    # Plot viscosity on the primary y-axis
    ax2.plot(df["lid_time_step"], df["viscosity"],
             label="Viscosity (Input)",
             color="green", marker="o", linestyle="-", markersize=3)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Viscosity", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_title("Viscosity & Pressure vs. Time Step")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Create a twin y-axis to plot pressure
    ax2b = ax2.twinx()
    ax2b.plot(df["lid_time_step"], df["pressure"],
              label="Pressure (Output)",
              color="purple", marker="x", linestyle="--", markersize=3)
    ax2b.set_ylabel("Pressure", color="purple")
    ax2b.tick_params(axis="y", labelcolor="purple")
    
    # Combine legends from both y-axes for the second subplot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    output_filename = "input_output_with_timestep.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    main()
