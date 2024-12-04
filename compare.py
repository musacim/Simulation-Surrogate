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
surrogate_model = joblib.load(surrogate_model_path)
scaler = joblib.load(scaler_path)

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

# Function to generate parameters for time step, matching sim_time_series.py
def get_parameters_for_time_step(t):
    if t <= 5:
        lid_velocities = np.arange(1.0, 3.1, 0.5)  # Lid velocities from 1.0 to 3.0 m/s
        viscosities = np.logspace(-3, -2, num=5)    # Viscosities from 1e-3 to 1e-2 m²/s
    else:
        lid_velocities = np.arange(3.5, 5.1, 0.5)  # Lid velocities from 3.5 to 5.0 m/s
        viscosities = np.logspace(-4, -3, num=5)    # Viscosities from 1e-4 to 1e-3 m²/s
    return lid_velocities, viscosities

# Compare over time steps
time_steps = range(1, 11)
for t in time_steps:
    lid_velocities, viscosities = get_parameters_for_time_step(t)
    for lid_velocity in lid_velocities:
        for viscosity in viscosities:
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
            speedup_ratio = simulation_time / surrogate_time
            print(f"Speedup Ratio (Simulation / Surrogate): {speedup_ratio:.2f}\n")
