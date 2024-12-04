# compare.py

import os
import subprocess
import time
import pandas as pd
import joblib
import numpy as np

# Paths and parameters
surrogate_model_path = "/home/musacim/simulation/openfoam/surrogate_model.joblib"
scaler_path = "/home/musacim/simulation/openfoam/scaler.joblib"

# Load the surrogate model and scaler
try:
    surrogate_model = joblib.load(surrogate_model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print("Surrogate model or scaler not found. Please run train_incremental.py first.")
    exit(1)

# Function to run simulation and measure time
def run_simulation(case_dir):
    start_time = time.time()
    try:
        with open(os.path.join(case_dir, 'log.blockMesh'), 'w') as bm_log, \
             open(os.path.join(case_dir, 'log.icoFoam'), 'w') as ico_log:
            subprocess.run(['blockMesh'], cwd=case_dir, stdout=bm_log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(['icoFoam'], cwd=case_dir, stdout=ico_log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed in '{case_dir}': {e}")
    end_time = time.time()
    return end_time - start_time

# Function to run surrogate model and measure time
def run_surrogate(input_params):
    X_input = pd.DataFrame([input_params])
    X_input_scaled = scaler.transform(X_input)
    start_time = time.time()
    velocity_pred = surrogate_model["velocity"].predict(X_input_scaled)
    pressure_pred = surrogate_model["pressure"].predict(X_input_scaled)
    end_time = time.time()
    surrogate_time = end_time - start_time
    return surrogate_time, velocity_pred, pressure_pred

# Function to generate parameters for time step, matching sim_time_series.py
def get_parameters_for_time_step(t):
    if 1 <= t <= 10:
        # Training Region
        lid_velocity = 1.0 + 0.2 * (t - 1)  # 1.0 to 3.0 m/s
        viscosity = 1e-3 + (1e-2 - 1e-3) * (t - 1) / 9  # 1e-3 to 1e-2 m²/s
    elif 11 <= t <= 20:
        # First Shifting Region
        lid_velocity = 3.0 + 0.2 * (t - 10)  # 3.0 to 5.0 m/s
        viscosity = 1e-2 - (1e-2 - 1e-3) * (t - 10) / 10  # 1e-2 to 1e-3 m²/s
    elif 21 <= t <= 30:
        # Second Shifting Region
        lid_velocity = 5.0 + 0.2 * (t - 20)  # 5.0 to 7.0 m/s
        viscosity = 1e-3 - (1e-3 - 1e-4) * (t - 20) / 10  # 1e-3 to 1e-4 m²/s
    return lid_velocity, viscosity

# Compare over time steps
time_steps = range(1, 31)
for t in time_steps:
    lid_velocity, viscosity = get_parameters_for_time_step(t)
    input_params = {"lid_velocity": lid_velocity, "viscosity": viscosity, "lid_time_step": t}

    # Define simulation case directory
    case_name = f"cavity_{lid_velocity:.2f}ms_{viscosity:.3e}_t{t}".replace('+', '')
    simulation_case_dir = os.path.join("/home/musacim/simulation/openfoam/cavity_simulations", case_name)

    # Check if the directory exists
    if not os.path.exists(simulation_case_dir):
        print(f"Case directory '{simulation_case_dir}' does not exist. Skipping this case.")
        continue

    # Measure simulation time
    simulation_time = run_simulation(simulation_case_dir)

    # Measure surrogate model time
    surrogate_time, velocity_pred, pressure_pred = run_surrogate(input_params)

    # Print comparison
    print(f"Time Step: {t}")
    print(f"Lid Velocity: {lid_velocity:.2f} m/s, Viscosity: {viscosity:.3e} m²/s")
    print(f"Simulation Time: {simulation_time:.4f} seconds")
    print(f"Surrogate Model Time: {surrogate_time:.4f} seconds")
    speedup_ratio = simulation_time / surrogate_time if surrogate_time > 0 else float('inf')
    print(f"Speedup Ratio (Simulation / Surrogate): {speedup_ratio:.2f}\n")
