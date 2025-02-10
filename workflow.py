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

TOTAL_STEPS = 1000
INITIAL_TRAIN_SIZE = 100
DRIFT_THRESHOLD = 2.5
RETRAIN_BATCH_SIZE = 80

SIMULATION_SCRIPT = "new_sim.py"
DATASET_SCRIPT = "dataset.py"
SURROGATE_SCRIPT = "surrogate.py"
SIM_OUTPUT_CSV = "/home/musacim/simulation/openfoam/simulation_output_data.csv"
COMBINED_OUTPUT_CSV = "combined_workflow_outputs.csv"

##############################
# TIMING ACCUMULATORS
##############################
total_sim_time = 0.0
total_train_time = 0.0
total_pred_time = 0.0
full_sim_time = 0.0

# For plotting retraining intervals
retrain_intervals = []

##############################
# HELPER FUNCTIONS
##############################
def run_simulation_range(start, end):
    print(f"\n[Simulation] Running simulation for time steps {start} to {end} ...")
    t0 = time.time()
    cmd = ["python3", SIMULATION_SCRIPT, "--start", str(start), "--end", str(end)]
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[Simulation] Completed simulation for steps {start} to {end} in {elapsed:.2f} s.\n")
    time.sleep(1)
    return elapsed

def run_dataset_conversion():
    print("\n[Dataset] Converting simulation outputs to CSV ...")
    cmd = ["python3", DATASET_SCRIPT]
    subprocess.run(cmd, check=True)
    print("[Dataset] Conversion completed.\n")
    time.sleep(1)

def run_surrogate_training(initial_train_size):
    print(f"\n[Surrogate] Training surrogate model using the first {initial_train_size} samples ...")
    t0 = time.time()
    cmd = ["python3", SURROGATE_SCRIPT, "--initial", str(initial_train_size)]
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[Surrogate] Training completed in {elapsed:.2f} s.\n")
    time.sleep(1)
    return elapsed

def append_record(record, filename=COMBINED_OUTPUT_CSV):
    file_exists = os.path.exists(filename)
    fieldnames = ["time_step", "lid_velocity", "viscosity", "output_source",
                  "velocity_magnitude", "pressure", "pred_velocity", "pred_pressure", "drift_metric"]
    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def get_parameters_for_time_step(t):
    """
    Dynamically determines ramp regions as fractions of TOTAL_STEPS,
    then adds some sinusoidal noise.
    """
    RAMP1_START = int(TOTAL_STEPS * 0.25)
    RAMP1_END   = int(TOTAL_STEPS * 0.375)
    RAMP2_START = int(TOTAL_STEPS * 0.625)
    RAMP2_END   = int(TOTAL_STEPS * 0.6875)
    
    BASE_LID_VEL_1 = 1.5
    BASE_LID_VEL_2 = 2.0
    BASE_LID_VEL_3 = 2.5
    BASE_VISCOSITY = 1e-3

    if t < RAMP1_START:
        base_lid = BASE_LID_VEL_1
    elif RAMP1_START <= t < RAMP1_END:
        fraction = (t - RAMP1_START) / (RAMP1_END - RAMP1_START)
        base_lid = BASE_LID_VEL_1 + fraction * (BASE_LID_VEL_2 - BASE_LID_VEL_1)
    elif RAMP1_END <= t < RAMP2_START:
        base_lid = BASE_LID_VEL_2
    elif RAMP2_START <= t < RAMP2_END:
        fraction = (t - RAMP2_START) / (RAMP2_END - RAMP2_START)
        base_lid = BASE_LID_VEL_2 + fraction * (BASE_LID_VEL_3 - BASE_LID_VEL_2)
    else:
        base_lid = BASE_LID_VEL_3

    period = 50
    angle = 2 * np.pi * (t / period)
    lid_amplitude = 0.1
    viscosity_amplitude = 5e-5
    lid_variation = lid_amplitude * np.sin(angle)
    viscosity_variation = viscosity_amplitude * np.cos(angle)
    lid_noise = np.random.uniform(-0.01, 0.01)
    viscosity_noise = np.random.uniform(-1e-5, 1e-5)
    lid_velocity = base_lid + lid_variation + lid_noise
    viscosity = BASE_VISCOSITY + viscosity_variation + viscosity_noise
    return lid_velocity, viscosity

def transform_time(t):
    return t

##############################
# MAIN WORKFLOW
##############################
def main():
    global total_sim_time, total_train_time, total_pred_time, full_sim_time
    print("=== Starting Comprehensive Sequential Workflow ===\n")

    # Remove existing combined output
    if os.path.exists(COMBINED_OUTPUT_CSV):
        os.remove(COMBINED_OUTPUT_CSV)

    # STEP 1: INITIAL SIMULATION
    print(f"[Step 1] Running simulation for initial training (steps 1 to {INITIAL_TRAIN_SIZE})...")
    sim_time = run_simulation_range(1, INITIAL_TRAIN_SIZE)
    total_sim_time += sim_time

    print("[Step 1] Converting simulation outputs to CSV ...")
    run_dataset_conversion()

    print("[Step 1] Training initial surrogate model ...")
    train_time = run_surrogate_training(INITIAL_TRAIN_SIZE)
    total_train_time += train_time

    # Load simulation output CSV for initial training data
    initial_df = pd.read_csv(SIM_OUTPUT_CSV)
    train_df = initial_df[initial_df["lid_time_step"] <= INITIAL_TRAIN_SIZE].copy()
    if train_df.empty:
        print("Error: No training data found.")
        sys.exit(1)

    # Write initial training data to combined CSV
    for _, row in train_df.iterrows():
        rec = {
            "time_step": row["lid_time_step"],
            "lid_velocity": row["lid_velocity"],
            "viscosity": row["viscosity"],
            "output_source": "simulation",
            "velocity_magnitude": row["velocity_magnitude"],
            "pressure": row["pressure"],
            "pred_velocity": row["velocity_magnitude"],
            "pred_pressure": row["pressure"],
            "drift_metric": 0.0
        }
        append_record(rec)
    
    baseline_mean = train_df["lid_velocity"].mean()
    baseline_std  = train_df["lid_velocity"].std()
    print(f"Baseline mean: {baseline_mean:.4f}, std: {baseline_std:.4f}\n")
    
    # STEP 2: SEQUENTIAL PREDICTION & DRIFT
    current_t = INITIAL_TRAIN_SIZE + 1
    surrogate_model = joblib.load("surrogate_model_rf.joblib")

    while current_t <= TOTAL_STEPS:
        lid_velocity, viscosity = get_parameters_for_time_step(current_t)
        drift_metric = abs(lid_velocity - baseline_mean) / (baseline_std if baseline_std > 0 else 1)
        if drift_metric > DRIFT_THRESHOLD:
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
            baseline_mean = new_train_df["lid_velocity"].mean()
            baseline_std  = new_train_df["lid_velocity"].std()
            retrain_df = new_train_df[(new_train_df["lid_time_step"] >= retrain_start) &
                                      (new_train_df["lid_time_step"] <= retrain_end)]
            for _, row in retrain_df.iterrows():
                rec = {
                    "time_step": row["lid_time_step"],
                    "lid_velocity": row["lid_velocity"],
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
        else:
            t0 = time.time()
            X_new = np.array([[lid_velocity, viscosity]])
            pred = surrogate_model.predict(X_new)[0]
            pred_time = time.time() - t0
            total_pred_time += pred_time
            rec = {
                "time_step": current_t,
                "lid_velocity": lid_velocity,
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

    print("\n=== Sequential Workflow Completed ===\n")
    
    # STEP 3: FULL SIMULATION
    print("[Final Comparison] Running full simulation for all data points...")
    full_sim_time = run_simulation_range(1, TOTAL_STEPS)
    run_dataset_conversion()
    full_sim_df = pd.read_csv(SIM_OUTPUT_CSV)

    combined_df = pd.read_csv(COMBINED_OUTPUT_CSV)
    pred_df = combined_df[combined_df["output_source"]=="surrogate"]
    merge_df = pd.merge(pred_df, full_sim_df, left_on="time_step", right_on="lid_time_step",
                        suffixes=("_pred", "_sim"))
    mse_vel = mean_squared_error(merge_df["pred_velocity"], merge_df["velocity_magnitude_sim"])
    mse_pres = mean_squared_error(merge_df["pred_pressure"], merge_df["pressure_sim"])
    
    velocity_error_pct = np.mean(np.abs(merge_df["pred_velocity"] - merge_df["velocity_magnitude_sim"]) /
                                 merge_df["velocity_magnitude_sim"]) * 100

    # STEP 4: PLOTTING
    df_all = pd.read_csv(COMBINED_OUTPUT_CSV)
    df_all["time_trans"] = df_all["time_step"].apply(transform_time)
    
    # Create the figure with two subplots; increase overall figure size.
    fig, (ax_main, ax_bar) = plt.subplots(2, 1, figsize=(20, 16))

    # --- Top Subplot: Simulation vs. Surrogate Predictions ---
    sim_pts = df_all[df_all["output_source"]=="simulation"]
    pred_pts = df_all[df_all["output_source"]=="surrogate"]

    ax_main.scatter(sim_pts["time_trans"], sim_pts["velocity_magnitude"], 
                    c="red", marker="o", label="Simulation Output")
    ax_main.scatter(pred_pts["time_trans"], pred_pts["pred_velocity"], 
                    c="blue", marker="x", label="Surrogate Prediction")

    # Plot drift metric (scaled by 0.03) as in the old code:
    ax_main.plot(df_all["time_trans"], df_all["drift_metric"] * 0.03, c="green", label="Drift Metric (x0.03)")
    
    ax_main.set_xlabel("Time Step", fontsize=14, labelpad=10)
    ax_main.set_ylabel("Velocity Magnitude", fontsize=14, labelpad=10)
    ax_main.set_title("Workflow Results: Simulation vs. Surrogate Predictions", fontsize=16)
    ax_main.grid(True, alpha=0.3)

    # Highlight initial training region
    ax_main.axvspan(transform_time(1), transform_time(INITIAL_TRAIN_SIZE), 
                    color="green", alpha=0.15, label="Initial Training Region")
    
    # Highlight retraining intervals (only label the first interval)
    first_collect_label = True
    for (rs, re) in retrain_intervals:
        if first_collect_label:
            ax_main.axvspan(transform_time(rs), transform_time(re),
                            color="orange", alpha=0.15,
                            label="Collecting New Data Points Region")
            first_collect_label = False
        else:
            ax_main.axvspan(transform_time(rs), transform_time(re),
                            color="orange", alpha=0.15, label=None)

    # Merge legends for the main plot and place the legend to the right.
    handles_main, labels_main = ax_main.get_legend_handles_labels()
    unique = dict(zip(labels_main, handles_main))
    ax_main.legend(unique.values(), unique.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12)

    # --- Bottom Subplot: Timing Comparison ---
    system_total_time = total_sim_time + total_train_time + total_pred_time
    categories = ["Our System", "Full Simulation"]
    
    ax_bar.bar(categories[0], total_sim_time, color="orange", label="Simulation")
    ax_bar.bar(categories[0], total_train_time, bottom=total_sim_time,
               color="purple", label="Surrogate Training")
    ax_bar.bar(categories[0], total_pred_time, 
               bottom=total_sim_time + total_train_time,
               color="cyan", label="Prediction")
    
    ax_bar.bar(categories[1], full_sim_time, color="magenta", label="Full Simulation")
    
    # Annotate bar heights
    ax_bar.text(0, system_total_time + 0.3, f"{system_total_time:.2f} s",
                ha="center", va="bottom", fontsize=12)
    ax_bar.text(1, full_sim_time + 0.3, f"{full_sim_time:.2f} s",
                ha="center", va="bottom", fontsize=12)
    
    # Put speedup in the top-left corner of the timing plot
    speedup = full_sim_time / system_total_time if system_total_time > 0 else float('inf')
    ax_bar.text(0.03, 0.95, f"Speedup: {speedup:.2f}x",
                transform=ax_bar.transAxes,
                fontsize=12, color="black", ha="left", va="top")

    ax_bar.set_ylabel("Total Time (s)", fontsize=14)
    ax_bar.set_title("Stacked Timing Metrics: Our System vs. Full Simulation", fontsize=16)
    ax_bar.set_ylim(0, max(system_total_time, full_sim_time) * 1.2 + 3)
    ax_bar.grid(True, alpha=0.3)

    # Place legend for the timing subplot to the left of the axis
    handles, labels = ax_bar.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_bar.legend(unique.values(), unique.keys(), fontsize=12, loc="center left", bbox_to_anchor=(-0.15, 0.5))

    # Increase space between subplots & free bottom margin
    plt.subplots_adjust(bottom=0.25, hspace=0.4)

    # Place workflow info text at the bottom
    info_text = (
        f"Workflow Information:\n"
        f"- Surrogate Model: RandomForestRegressor\n"
        f"- Initial Training: Steps 1 to {INITIAL_TRAIN_SIZE}\n"
        f"- Drift Detection: z-score on lid_velocity (Threshold = {DRIFT_THRESHOLD})\n"
        f"- Retraining Steps: {RETRAIN_BATCH_SIZE} upon drift detection\n"
        f"- Overall Velocity Error: {velocity_error_pct:.2f}%\n"
    )
    fig.text(
        0.5, 0.03, info_text,
        ha="center", va="bottom", fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.8')
    )

    plot_filename = "comprehensive_workflow_plot.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Plot saved as '{plot_filename}'")

if __name__ == "__main__":
    main()
