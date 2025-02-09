import os
import subprocess
import time
import pandas as pd
import joblib
import numpy as np

# Paths and parameters
base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"
surrogate_model_path = "/home/musacim/simulation/openfoam/surrogate_model.joblib"
scaler_path = "/home/musacim/simulation/openfoam/scaler.joblib"

# Load the surrogate model and scaler
surrogate_model = joblib.load(surrogate_model_path)
scaler = joblib.load(scaler_path)

# Function to dynamically find matching directories
def find_case_directory(lid_velocity, viscosity, time_step):
    # Generate possible directory name pattern
    formatted_velocity = f"{lid_velocity:.3f}".rstrip('0').rstrip('.')  # Match the directory naming
    formatted_viscosity = f"{viscosity:.3e}".replace('+', '')  # Remove "+" for consistency
    expected_name = f"cavity_{formatted_velocity}ms_{formatted_viscosity}_t{time_step}"
    
    # Check if the directory exists
    case_path = os.path.join(base_dir, expected_name)
    if os.path.exists(case_path):
        return case_path
    else:
        # Log that a match was not found
        print(f"Expected directory '{expected_name}' not found.")
        return None

# Function to run simulation and measure time
def run_simulation(case_dir):
    start_time = time.time()
    with open(os.path.join(case_dir, 'log.blockMesh'), 'w') as bm_log, \
         open(os.path.join(case_dir, 'log.icoFoam'), 'w') as ico_log:
        subprocess.run(['blockMesh'], cwd=case_dir, stdout=bm_log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['icoFoam'], cwd=case_dir, stdout=ico_log, stderr=subprocess.STDOUT, check=True)
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

# Function to generate parameters for time steps
def get_parameters_for_time_step(t):
    if t <= 15:
        lid_velocity = 1.0
        viscosity = 1e-3
    elif 16 <= t <= 30:
        lid_velocity = 1.0 + 0.1 * (t - 15)
        viscosity = 1e-3 - 5e-5 * (t - 15)
    else:
        lid_velocity = 2.5 + 0.05 * (t - 30)
        viscosity = 2.5e-4 - 1e-5 * (t - 30)
    return lid_velocity, viscosity

# Compare over time steps
time_steps = range(1, 46)
total_simulation_time = 0.0
total_surrogate_time = 0.0

for t in time_steps:
    lid_velocity, viscosity = get_parameters_for_time_step(t)
    input_params = {"lid_velocity": lid_velocity, "viscosity": viscosity}

    # Dynamically find the case directory
    simulation_case_dir = find_case_directory(lid_velocity, viscosity, t)
    if simulation_case_dir is None:
        continue

    # Measure simulation time
    simulation_time = run_simulation(simulation_case_dir)

    # Measure surrogate model time
    surrogate_time, velocity_pred, pressure_pred = run_surrogate(input_params)

    # Accumulate total times
    total_simulation_time += simulation_time
    total_surrogate_time += surrogate_time

    # Print comparison
    print(f"Time Step: {t}")
    print(f"Lid Velocity: {lid_velocity:.2f} m/s, Viscosity: {viscosity:.3e} m²/s")
    print(f"Simulation Time: {simulation_time:.4f} seconds")
    print(f"Surrogate Model Time: {surrogate_time:.4f} seconds")
    if surrogate_time > 0:
        speedup_ratio = simulation_time / surrogate_time
        print(f"Speedup Ratio (Simulation / Surrogate): {speedup_ratio:.2f}\n")
    else:
        print("Surrogate model time is too small to calculate speedup ratio.\n")

# Print total results
print(f"\nTotal Simulation Time for all {len(time_steps)} time steps: {total_simulation_time:.2f} seconds.")
print(f"Total Surrogate Model Time for all {len(time_steps)} time steps: {total_surrogate_time:.2f} seconds.")
if total_surrogate_time > 0:
    overall_speedup = total_simulation_time / total_surrogate_time
    print(f"Overall Speedup Ratio (Total Simulation / Total Surrogate): {overall_speedup:.2f}")
else:
    print("Total surrogate model time is too small to calculate overall speedup ratio.")
