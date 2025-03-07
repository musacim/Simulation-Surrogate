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

TOTAL_STEPS = 500            # The single source of truth for total steps
INITIAL_TRAIN_SIZE = 50      # Steps 1 to 50 for initial training
DRIFT_THRESHOLD = 3          # z-score threshold for drift
RETRAIN_BATCH_SIZE = 30      # Retraining batch size upon drift detection

SIMULATION_SCRIPT = "cylinder_sim.py"         
DATASET_SCRIPT = "cylinder_dataset.py"            
SURROGATE_SCRIPT = "cylinder_surrogate.py"             
SIM_OUTPUT_CSV = "/home/musacim/simulation/openfoam/mhd/hartmann_simulation_output_data.csv"
COMBINED_OUTPUT_CSV = "combined_workflow_outputs.csv"

##############################
# TIMING ACCUMULATORS
##############################
total_sim_time = 0.0
total_train_time = 0.0
total_pred_time = 0.0

# Store intervals where retraining happened (for plotting)
retrain_intervals = []

##############################
# HELPER FUNCTIONS
##############################
def run_simulation_range(start, end):
    print(f"\n[Simulation] Running simulation for time steps {start} to {end} ...")
    t0 = time.time()
    cmd = [
        "python3", SIMULATION_SCRIPT,
        "--start", str(start),
        "--end", str(end),
        "--total_steps", str(TOTAL_STEPS)
    ]
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

def run_surrogate_training(training_data_size):
    """
    Re-trains the surrogate model using the first 'training_data_size' rows in the CSV,
    or all rows if training_data_size equals the length of the CSV.
    """
    print(f"\n[Surrogate] Training surrogate model using the first {training_data_size} samples ...")
    t0 = time.time()
    cmd = ["python3", SURROGATE_SCRIPT, "--initial", str(training_data_size)]
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[Surrogate] Training completed in {elapsed:.2f} s.\n")
    time.sleep(1)
    return elapsed

def append_record(record, filename=COMBINED_OUTPUT_CSV):
    file_exists = os.path.exists(filename)
    fieldnames = [
        "time_step", "inlet_velocity", "B_magnitude", "output_source",
        "velocity_magnitude", "pressure", "pred_velocity", "pred_pressure", "drift_metric"
    ]
    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def get_parameters_for_time_step(t):
    """
    For PLOTTING / PREDICTION ONLY:
    - Inlet velocity = 1 m/s
    - B ramps from 20 T to 1 T (with a bit of random noise)
    """
    RAMP_START = int(TOTAL_STEPS * 0.5)
    RAMP_END   = int(TOTAL_STEPS * 0.6)
    inlet_velocity = 1.0

    if t < RAMP_START:
        B_magnitude = 20.0
    elif t < RAMP_END:
        fraction = (t - RAMP_START) / (RAMP_END - RAMP_START)
        B_magnitude = 20.0 + fraction * (1.0 - 20.0)
    else:
        B_magnitude = 1.0

    noise = np.random.uniform(-0.1, 0.1)
    B_magnitude += noise
    return inlet_velocity, B_magnitude

def transform_time(t):
    return t

##############################
# MAIN WORKFLOW
##############################
def main():
    global total_sim_time, total_train_time, total_pred_time

    print("=== Starting Comprehensive Sequential Workflow (MHD Hartmann Case) ===\n")

    # Remove any old combined CSV so we start fresh
    if os.path.exists(COMBINED_OUTPUT_CSV):
        os.remove(COMBINED_OUTPUT_CSV)

    # STEP 1: INITIAL SIMULATION & TRAINING
    print(f"[Step 1] Running simulation for initial training (steps 1 to {INITIAL_TRAIN_SIZE})...")
    sim_time = run_simulation_range(1, INITIAL_TRAIN_SIZE)
    total_sim_time += sim_time

    print("[Step 1] Converting simulation outputs to CSV ...")
    run_dataset_conversion()

    print("[Step 1] Training initial surrogate model ...")
    train_time = run_surrogate_training(INITIAL_TRAIN_SIZE)
    total_train_time += train_time

    # Load the CSV that was just generated
    initial_df = pd.read_csv(SIM_OUTPUT_CSV)
    train_df = initial_df[initial_df["time_step"] <= INITIAL_TRAIN_SIZE].copy()
    if train_df.empty:
        print("Error: No training data found in the initial range.")
        sys.exit(1)

    # Record those initial points in the combined CSV
    for _, row in train_df.iterrows():
        rec = {
            "time_step": row["time_step"],
            "inlet_velocity": row["inlet_velocity"],
            "B_magnitude": row["B_magnitude"],
            "output_source": "simulation",
            "velocity_magnitude": row["velocity_magnitude"],
            "pressure": row["pressure"],
            "pred_velocity": row["velocity_magnitude"],
            "pred_pressure": row["pressure"],
            "drift_metric": 0.0
        }
        append_record(rec)

    # DRIFT DETECTION ON B_MAGNITUDE
    baseline_mean = train_df["B_magnitude"].mean()
    baseline_std  = train_df["B_magnitude"].std()
    print(f"Baseline (initial training) - B_magnitude mean: {baseline_mean:.4f}, std: {baseline_std:.4f}\n")
    
    # STEP 2: SEQUENTIAL PREDICTION & DRIFT MONITORING
    current_t = INITIAL_TRAIN_SIZE + 1
    surrogate_model = joblib.load("surrogate_model_rf.joblib")

    while current_t <= TOTAL_STEPS:
        # We'll re-compute B_magnitude for the drift metric
        # (We do not actually run the solver, just the param function)
        _, B_magnitude = get_parameters_for_time_step(current_t)
        drift_metric = abs(B_magnitude - baseline_mean) / (baseline_std if baseline_std > 0 else 1)

        print(f"Time step {current_t}: B_magnitude = {B_magnitude:.2f}, drift_metric = {drift_metric:.2f}")

        # Check drift
        if drift_metric > DRIFT_THRESHOLD:
            print(f"\n*** Drift detected at time step {current_t} (drift_metric = {drift_metric:.2f})! ***")
            retrain_start = current_t
            retrain_end = min(current_t + RETRAIN_BATCH_SIZE - 1, TOTAL_STEPS)
            retrain_intervals.append((retrain_start, retrain_end))

            # RUN THE SIMULATOR for the retraining batch
            sim_time = run_simulation_range(retrain_start, retrain_end)
            total_sim_time += sim_time

            # Convert dataset again (so we have the new simulation data in CSV)
            run_dataset_conversion()
            new_train_df = pd.read_csv(SIM_OUTPUT_CSV)

            # Retrain on ALL data so far
            train_time = run_surrogate_training(len(new_train_df))
            total_train_time += train_time
            surrogate_model = joblib.load("surrogate_model_rf.joblib")

            # Update baseline to reflect all data so far
            baseline_mean = new_train_df["B_magnitude"].mean()
            baseline_std  = new_train_df["B_magnitude"].std()
            print(f"Updated baseline for B_magnitude: mean = {baseline_mean:.4f}, std = {baseline_std:.4f}\n")

            # Add these new simulation data points to the combined CSV
            retrain_df = new_train_df[
                (new_train_df["time_step"] >= retrain_start) &
                (new_train_df["time_step"] <= retrain_end)
            ]
            for _, row in retrain_df.iterrows():
                rec = {
                    "time_step": row["time_step"],
                    "inlet_velocity": row["inlet_velocity"],
                    "B_magnitude": row["B_magnitude"],
                    "output_source": "simulation",
                    "velocity_magnitude": row["velocity_magnitude"],
                    "pressure": row["pressure"],
                    "pred_velocity": row["velocity_magnitude"],
                    "pred_pressure": row["pressure"],
                    "drift_metric": drift_metric
                }
                append_record(rec)

            # Move the current time pointer past the retraining batch
            current_t = retrain_end + 1

        else:
            # SURROGATE PREDICTION
            # We'll also need the inlet_velocity to do a surrogate prediction
            inlet_velocity, B_mag = get_parameters_for_time_step(current_t)
            t0 = time.time()
            X_new = np.array([[inlet_velocity, B_mag]])
            pred = surrogate_model.predict(X_new)[0]
            pred_time = time.time() - t0
            total_pred_time += pred_time

            # Record the surrogate-based result
            rec = {
                "time_step": current_t,
                "inlet_velocity": inlet_velocity,
                "B_magnitude": B_mag,
                "output_source": "surrogate",
                "velocity_magnitude": None,
                "pressure": None,
                "pred_velocity": pred[0],
                "pred_pressure": pred[1],
                "drift_metric": drift_metric
            }
            append_record(rec)
            current_t += 1

    print(f"\n=== Sequential Workflow Completed. Results saved to '{COMBINED_OUTPUT_CSV}' ===\n")

    # (OPTIONAL) If you do NOT want a final full simulation, do nothing here.
    # If you do want a final run for ground truth, re-enable the code below:
    #
    # print("[Final Comparison] Running full simulation for all data points (1 to TOTAL_STEPS)...")
    # full_sim_time = run_simulation_range(1, TOTAL_STEPS)
    # run_dataset_conversion()
    # ...

    # STEP 3 (REMOVED) - No final full simulation for ground truth

    # Plot the combined data so far
    df_all = pd.read_csv(COMBINED_OUTPUT_CSV)
    df_all["time_trans"] = df_all["time_step"].apply(transform_time)

    sim_pts = df_all[df_all["output_source"] == "simulation"]
    pred_pts = df_all[df_all["output_source"] == "surrogate"]

    fig, ax_main = plt.subplots(figsize=(12, 6))

    # For simulation points, we have an actual velocity_magnitude
    ax_main.scatter(sim_pts["time_trans"], sim_pts["velocity_magnitude"],
                    c="red", marker="o", label="Simulation Output", zorder=1)
    # For surrogate points, we only have predicted velocity
    ax_main.scatter(pred_pts["time_trans"], pred_pts["pred_velocity"],
                    c="blue", marker="x", label="Surrogate Prediction", zorder=2)

    ax_main.plot(df_all["time_trans"], df_all["drift_metric"] * 0.05,
                 c="green", label="Drift Metric (scaled)")

    ax_main.set_xlabel("Time Step")
    ax_main.set_ylabel("Velocity Magnitude")
    ax_main.set_title("Surrogate vs. Simulation (Drift on B_magnitude)")

    # Highlight initial training region
    ax_main.axvspan(
        transform_time(1),
        transform_time(INITIAL_TRAIN_SIZE),
        color="green", alpha=0.15,
        label="Initial Training Region"
    )

    # Highlight retraining intervals
    first_collect_label = True
    for (rs, re) in retrain_intervals:
        label = "New Simulation Data Region" if first_collect_label else None
        ax_main.axvspan(transform_time(rs), transform_time(re),
                        color="orange", alpha=0.15, label=label)
        first_collect_label = False

    handles_main, labels_main = ax_main.get_legend_handles_labels()
    unique = dict(zip(labels_main, handles_main))
    ax_main.legend(unique.values(), unique.keys(), loc="upper right")

    plt.tight_layout()
    plt.savefig("surrogate_vs_simulation_plot.png")
    plt.show()
    print("[INFO] Plot saved as 'surrogate_vs_simulation_plot.png'")

if __name__ == "__main__":
    main()
