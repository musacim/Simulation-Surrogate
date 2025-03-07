#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error

##############################
# CONFIGURATION PARAMETERS
##############################

TOTAL_STEPS = 1000                # Total number of simulation time steps
INITIAL_TRAIN_SIZE = 100          # Steps 1 to 100 for initial simulation & surrogate training
DRIFT_THRESHOLD = 3               # Drift threshold (z-score on inlet_velocity)
RETRAIN_BATCH_SIZE = 80           # When drift is detected, simulate these many additional steps

# Updated scripts and file names for backward-facing step simulation
SIMULATION_SCRIPT = "bfs_sim.py"          # OpenFOAM simulation script for BFS
DATASET_SCRIPT = "bfs_dataset.py"         # Converts BFS case directories into CSV data
SURROGATE_SCRIPT = "bfs_surrogate.py"     # Trains surrogate model; saves model to "surrogate_model_rf.joblib"
SIM_OUTPUT_CSV = "/home/musacim/simulation/openfoam/bfs_simulation_output_data.csv"
COMBINED_OUTPUT_CSV = "bfs_combined_workflow_outputs.csv"

##############################
# TIMING ACCUMULATORS
##############################
total_sim_time = 0.0
total_train_time = 0.0
total_pred_time = 0.0
full_sim_time = 0.0  # For the final full simulation

# To store retraining intervals for plotting (each as tuple: (start, end))
retrain_intervals = []

##############################
# HELPER FUNCTIONS
##############################
def run_simulation_range(start, end):
    print(f"\n[Simulation] Running BFS simulation for time steps {start} to {end} ...")
    t0 = time.time()
    cmd = ["python3", SIMULATION_SCRIPT, "--start", str(start), "--end", str(end)]
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[Simulation] Completed BFS simulation for steps {start} to {end} in {elapsed:.2f} s.\n")
    time.sleep(1)
    return elapsed

def run_dataset_conversion():
    print("\n[Dataset] Converting BFS simulation outputs to CSV ...")
    cmd = ["python3", DATASET_SCRIPT]
    subprocess.run(cmd, check=True)
    print("[Dataset] Conversion completed.\n")
    time.sleep(1)

def run_surrogate_training(train_size):
    print(f"\n[Surrogate] Training surrogate model using the first {train_size} samples ...")
    t0 = time.time()
    cmd = ["python3", SURROGATE_SCRIPT, "--initial", str(train_size)]
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[Surrogate] Training completed in {elapsed:.2f} s.\n")
    time.sleep(1)
    return elapsed

def append_record(record, filename=COMBINED_OUTPUT_CSV):
    file_exists = os.path.exists(filename)
    fieldnames = ["time_step", "inlet_velocity", "viscosity", "output_source",
                  "velocity_magnitude", "pressure", "pred_velocity", "pred_pressure", "drift_metric"]
    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def get_parameters_for_time_step(t):
    """
    Generates dynamic inlet velocity and viscosity for the BFS simulation.
    The inlet velocity is varied in a piecewise linear (ramp) fashion with added sinusoidal perturbations.
    """
    # Define ramp intervals (adjust fractions as needed)
    RAMP1_START = int(TOTAL_STEPS * 0.25)
    RAMP1_END   = int(TOTAL_STEPS * 0.30)
    RAMP2_START = int(TOTAL_STEPS * 0.625)
    RAMP2_END   = int(TOTAL_STEPS * 0.675)

    BASE_INLET_VEL_1 = 1.0   # Base inlet velocity in region 1
    BASE_INLET_VEL_2 = 1.5   # Increased inlet velocity in region 2
    BASE_INLET_VEL_3 = 2.0   # Highest inlet velocity in region 3
    BASE_VISCOSITY = 1e-3    # Constant base viscosity

    if t < RAMP1_START:
        base_inlet = BASE_INLET_VEL_1
    elif RAMP1_START <= t < RAMP1_END:
        fraction = (t - RAMP1_START) / (RAMP1_END - RAMP1_START)
        base_inlet = BASE_INLET_VEL_1 + fraction * (BASE_INLET_VEL_2 - BASE_INLET_VEL_1)
    elif RAMP1_END <= t < RAMP2_START:
        base_inlet = BASE_INLET_VEL_2
    elif RAMP2_START <= t < RAMP2_END:
        fraction = (t - RAMP2_START) / (RAMP2_END - RAMP2_START)
        base_inlet = BASE_INLET_VEL_2 + fraction * (BASE_INLET_VEL_3 - BASE_INLET_VEL_2)
    else:
        base_inlet = BASE_INLET_VEL_3

    # Add oscillatory variations and random noise
    period = 50
    angle = 2 * np.pi * (t / period)
    inlet_amplitude = 0.1
    viscosity_amplitude = 5e-5
    inlet_variation = inlet_amplitude * np.sin(angle)
    viscosity_variation = viscosity_amplitude * np.cos(angle)
    inlet_noise = np.random.uniform(-0.01, 0.01)
    viscosity_noise = np.random.uniform(-1e-5, 1e-5)

    inlet_velocity = base_inlet + inlet_variation + inlet_noise
    viscosity = BASE_VISCOSITY + viscosity_variation + viscosity_noise
    return inlet_velocity, viscosity

def transform_time(t):
    return t

##############################
# MAIN WORKFLOW
##############################
def main():
    global total_sim_time, total_train_time, total_pred_time, full_sim_time
    print("=== Starting BFS Simulation Workflow ===\n")

    # Remove previous combined output CSV if it exists
    if os.path.exists(COMBINED_OUTPUT_CSV):
        os.remove(COMBINED_OUTPUT_CSV)

    ##############################
    # STEP 1: INITIAL SIMULATION & TRAINING
    ##############################
    print(f"[Step 1] Running BFS simulation for initial training (steps 1 to {INITIAL_TRAIN_SIZE})...")
    sim_time = run_simulation_range(1, INITIAL_TRAIN_SIZE)
    total_sim_time += sim_time

    print("[Step 1] Converting BFS simulation outputs to CSV ...")
    run_dataset_conversion()

    print("[Step 1] Training initial surrogate model ...")
    train_time = run_surrogate_training(INITIAL_TRAIN_SIZE)
    total_train_time += train_time

    # Load the simulation data CSV (expected to have columns: bfs_time_step, inlet_velocity, viscosity, velocity_magnitude, pressure)
    initial_df = pd.read_csv(SIM_OUTPUT_CSV)
    train_df = initial_df[initial_df["bfs_time_step"] <= INITIAL_TRAIN_SIZE].copy()
    if train_df.empty:
        print("Error: No training data found.")
        sys.exit(1)
    for _, row in train_df.iterrows():
        rec = {
            "time_step": row["bfs_time_step"],
            "inlet_velocity": row["inlet_velocity"],
            "viscosity": row["viscosity"],
            "output_source": "simulation",
            "velocity_magnitude": row["velocity_magnitude"],
            "pressure": row["pressure"],
            "pred_velocity": row["velocity_magnitude"],
            "pred_pressure": row["pressure"],
            "drift_metric": 0.0
        }
        append_record(rec)
    
    baseline_mean = train_df["inlet_velocity"].mean()
    baseline_std  = train_df["inlet_velocity"].std()
    print(f"Baseline (initial training) - mean inlet_velocity: {baseline_mean:.4f}, std: {baseline_std:.4f}\n")
    
    ##############################
    # STEP 2: SEQUENTIAL PREDICTION & DRIFT MONITORING
    ##############################
    current_t = INITIAL_TRAIN_SIZE + 1
    surrogate_model = joblib.load("surrogate_model_rf.joblib")
    
    while current_t <= TOTAL_STEPS:
        inlet_velocity, viscosity = get_parameters_for_time_step(current_t)
        drift_metric = abs(inlet_velocity - baseline_mean) / (baseline_std if baseline_std > 0 else 1)
        print(f"Time step {current_t}: inlet_velocity = {inlet_velocity:.4f}, drift_metric = {drift_metric:.2f}")
        
        if drift_metric > DRIFT_THRESHOLD:
            print(f"\n*** Drift detected at time step {current_t} (drift_metric = {drift_metric:.2f})! ***")
            retrain_start = current_t
            retrain_end = min(current_t + RETRAIN_BATCH_SIZE - 1, TOTAL_STEPS)
            retrain_intervals.append((retrain_start, retrain_end))
            sim_time = run_simulation_range(retrain_start, retrain_end)
            total_sim_time += sim_time
            run_dataset_conversion()
            new_train_df = pd.read_csv(SIM_OUTPUT_CSV)
            train_time = run_surrogate_training(len(new_train_df))
            total_train_time += train_time
            surrogate_model = joblib.load("surrogate_model_rf.joblib")
            baseline_mean = new_train_df["inlet_velocity"].mean()
            baseline_std  = new_train_df["inlet_velocity"].std()
            print(f"Updated baseline: mean = {baseline_mean:.4f}, std = {baseline_std:.4f}\n")
            retrain_df = new_train_df[(new_train_df["bfs_time_step"] >= retrain_start) &
                                      (new_train_df["bfs_time_step"] <= retrain_end)]
            for _, row in retrain_df.iterrows():
                rec = {
                    "time_step": row["bfs_time_step"],
                    "inlet_velocity": row["inlet_velocity"],
                    "viscosity": row["viscosity"],
                    "output_source": "simulation",
                    "velocity_magnitude": row["velocity_magnitude"],
                    "pressure": row["pressure"],
                    "pred_velocity": row["velocity_magnitude"],
                    "pred_pressure": row["pressure"],
                    "drift_metric": drift_metric
                }
                append_record(rec)
            current_t = retrain_end + 1
            continue
        else:
            t0 = time.time()
            X_new = np.array([[inlet_velocity, viscosity]])
            pred = surrogate_model.predict(X_new)[0]
            pred_time = time.time() - t0
            total_pred_time += pred_time
            rec = {
                "time_step": current_t,
                "inlet_velocity": inlet_velocity,
                "viscosity": viscosity,
                "output_source": "surrogate",
                "velocity_magnitude": None,
                "pressure": None,
                "pred_velocity": pred[0],
                "pred_pressure": pred[1],
                "drift_metric": drift_metric
            }
            append_record(rec)
            current_t += 1

    print(f"\n=== Sequential BFS Workflow Completed. Results saved to '{COMBINED_OUTPUT_CSV}' ===\n")
    
    ##############################
    # STEP 3: FULL SIMULATION RUN FOR GROUND TRUTH
    ##############################
    print("[Final Comparison] Running full BFS simulation for all data points (1 to TOTAL_STEPS)...")
    full_sim_time = run_simulation_range(1, TOTAL_STEPS)
    run_dataset_conversion()
    full_sim_df = pd.read_csv(SIM_OUTPUT_CSV)
    # Rename time step column for consistency
    full_sim_df.rename(columns={"bfs_time_step": "time_step"}, inplace=True)
    
    # Recompute simulation inputs (inlet_velocity and viscosity) for each time step
    inputs_list = []
    for t in range(1, TOTAL_STEPS + 1):
        in_vel, in_visc = get_parameters_for_time_step(t)
        inputs_list.append({"time_step": t, "inlet_velocity_input": in_vel, "viscosity_input": in_visc})
    inputs_df = pd.DataFrame(inputs_list)
    
    # Merge simulation inputs with ground truth simulation outputs.
    groundtruth_df = pd.merge(inputs_df, full_sim_df, on="time_step", how="left")
    groundtruth_df.rename(columns={"velocity_magnitude": "velocity_magnitude_ground",
                                   "pressure": "pressure_ground",
                                   "inlet_velocity": "inlet_velocity_sim",
                                   "viscosity": "viscosity_sim"}, inplace=True)
    
    # Compute example MSE calculations
    mse_surrogate_vel = 0.0
    mse_surrogate_pres = 0.0
    velocity_error_pct = 0.0
    
    combined_df = pd.read_csv(COMBINED_OUTPUT_CSV)
    pred_df = combined_df[combined_df["output_source"]=="surrogate"]
    if not pred_df.empty:
        merge_df = pd.merge(pred_df, full_sim_df, on="time_step", suffixes=("_pred", "_sim"))
        mse_surrogate_vel = mean_squared_error(merge_df["pred_velocity"], merge_df["velocity_magnitude_sim"])
        mse_surrogate_pres = mean_squared_error(merge_df["pred_pressure"], merge_df["pressure_sim"])
        velocity_error_pct = np.mean(np.abs(merge_df["pred_velocity"] - merge_df["velocity_magnitude_sim"]) /
                                     merge_df["velocity_magnitude_sim"]) * 100

    print(f"Overall surrogate prediction MSE - Velocity: {mse_surrogate_vel:.6f}, Pressure: {mse_surrogate_pres:.6f}")
    print(f"Average Velocity Prediction Error: {velocity_error_pct:.2f}%\n")
    
    ##############################
    # STEP 4: COMPREHENSIVE PLOTTING
    ##############################
    df_all = pd.read_csv(COMBINED_OUTPUT_CSV)
    df_all["time_trans"] = df_all["time_step"].apply(transform_time)
    
    fig, (ax_main, ax_bar) = plt.subplots(2, 1, figsize=(20, 16))
    
    # Prepare subsets
    sim_pts = df_all[df_all["output_source"]=="simulation"]
    pred_pts = df_all[df_all["output_source"]=="surrogate"]
    
    # Ensure groundtruth_df has time_trans
    groundtruth_df["time_trans"] = groundtruth_df["time_step"].apply(transform_time)
    
    # Plot simulation outputs (red) and surrogate predictions (blue)
    ax_main.scatter(
        sim_pts["time_trans"], 
        sim_pts["velocity_magnitude"], 
        c="red", marker="o", 
        label="BFS Simulation Output", 
        zorder=1
    )
    ax_main.scatter(
        pred_pts["time_trans"], 
        pred_pts["pred_velocity"], 
        c="blue", marker="x", 
        label="Surrogate Prediction", 
        zorder=2
    )
    
    # Plot drift metric (scaled for visualization)
    ax_main.plot(
        df_all["time_trans"], 
        df_all["drift_metric"] * 0.03, 
        c="green", 
        label="Drift Metric"
    )
    
    ax_main.set_xlabel("Time Step", fontsize=14, labelpad=10)
    ax_main.set_ylabel("Velocity Magnitude", fontsize=14, labelpad=10)
    ax_main.set_title(
        "BFS Workflow: Surrogate Predictions and Simulation Data",
        fontsize=16
    )
    ax_main.grid(True, alpha=0.3)
    
    # Highlight initial training region in green
    ax_main.axvspan(
        transform_time(1), 
        transform_time(INITIAL_TRAIN_SIZE), 
        color="green", alpha=0.15, 
        label="Initial Training Region"
    )
    
    # Highlight retraining intervals in orange
    first_collect_label = True
    for (rs, re) in retrain_intervals:
        label = "New Simulation Data Region" if first_collect_label else None
        ax_main.axvspan(
            transform_time(rs), 
            transform_time(re),
            color="orange", alpha=0.15, 
            label=label
        )
        first_collect_label = False
    
    handles_main, labels_main = ax_main.get_legend_handles_labels()
    unique = dict(zip(labels_main, handles_main))
    ax_main.legend(unique.values(), unique.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=14)
    
    # Bottom subplot: Stacked Timing Metrics
    system_total_time = total_sim_time + total_train_time + total_pred_time
    categories = ["Our System", "Full Simulation"]
    
    ax_bar.bar(categories[0], total_sim_time, color="orange", label="Simulation")
    ax_bar.bar(categories[0], total_train_time, bottom=total_sim_time,
               color="purple", label="Surrogate Training")
    ax_bar.bar(categories[0], total_pred_time, 
               bottom=total_sim_time + total_train_time,
               color="cyan", label="Prediction")
    ax_bar.bar(categories[1], full_sim_time, color="magenta", label="Full Simulation")
    
    ax_bar.text(
        0, system_total_time + 0.05 * system_total_time, 
        f"{system_total_time:.2f} s", ha="center", va="bottom", fontsize=14
    )
    ax_bar.text(
        1, full_sim_time + 0.05 * full_sim_time, 
        f"{full_sim_time:.2f} s", ha="center", va="bottom", fontsize=14
    )
    
    # Annotate speedup value: (Full Simulation Time) / (Our System Total Time)
    if system_total_time > 0:
        speedup = full_sim_time / system_total_time
        max_height = max(system_total_time, full_sim_time)
        ax_bar.text(0.5, max_height * 1.05, f"Speed Up: {speedup:.2f}x", ha="center", va="bottom", fontsize=16, color="darkblue")
    
    ax_bar.set_ylabel("Total Time (s)", fontsize=14)
    ax_bar.set_title("Time Comparison: Our BFS System vs. Full Simulation", fontsize=16)
    ax_bar.set_ylim(0, max(system_total_time, full_sim_time) * 1.2 + 3)
    handles_bar, labels_bar = ax_bar.get_legend_handles_labels()
    unique_bar = dict(zip(labels_bar, handles_bar))
    ax_bar.legend(unique_bar.values(), unique_bar.keys(), fontsize=14, loc="center left", bbox_to_anchor=(-0.15, 0.5))
    
    plt.subplots_adjust(bottom=0.25, hspace=0.4)
    info_text = (
        f"Workflow Information:\n"
        f"- Surrogate Model: RandomForestRegressor\n"
        f"- Initial Training: Steps 1 to {INITIAL_TRAIN_SIZE}\n"
        f"- Drift Detection: z-score on inlet_velocity (Threshold = {DRIFT_THRESHOLD})\n"
        f"- Retraining Batch Size: {RETRAIN_BATCH_SIZE} data points upon drift\n"
        f"- New surrogate model retrained using ALL BFS simulation data collected so far\n"
        f"- Average Velocity Prediction Error: {velocity_error_pct:.2f}%\n"
        f"- Full Simulation run on all {TOTAL_STEPS} data points\n"
    )
    plt.figtext(
        0.5, 0.02, info_text, wrap=True, 
        horizontalalignment="center", fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.8')
    )
    
    plot_filename = "bfs_comprehensive_workflow_plot.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Comprehensive plot saved as '{plot_filename}'")
    
    ##############################
    # STEP 5: GROUND TRUTH PLOT
    ##############################
    fig2, ax_ground = plt.subplots(figsize=(12, 8))
    ax_ground.scatter(
        groundtruth_df["time_trans"],
        groundtruth_df["velocity_magnitude_ground"],
        c="black", marker="s", label="Full BFS Simulation Ground Truth"
    )
    ax_ground.set_xlabel("Time Step", fontsize=14)
    ax_ground.set_ylabel("Velocity Magnitude", fontsize=14)
    ax_ground.set_title("Ground Truth BFS Simulation Output", fontsize=16)
    ax_ground.grid(True, alpha=0.3)
    ax_ground.legend(fontsize=14)
    ground_truth_plot_filename = "bfs_ground_truth_plot.png"
    plt.savefig(ground_truth_plot_filename, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Ground truth plot saved as '{ground_truth_plot_filename}'")

if __name__ == "__main__":
    main()
