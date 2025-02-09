import os
import subprocess
import numpy as np
import time
import csv

# -----------------------
# Global variables for drift
# -----------------------
# These globals will help us trigger and sustain a drift state for a given number of time steps.
drift_state = False
drift_duration_remaining = 0

# Drift parameters (adjust as needed)
DRIFT_PROBABILITY = 0.02      # 2% chance to start a drift event in a given time step
DRIFT_DURATION = 10           # Number of consecutive time steps a drift event lasts
DRIFT_OFFSET_VELOCITY = 0.5   # Additional offset when drift is active (m/s)
DRIFT_OFFSET_VISCOSITY = 5e-4 # Additional offset when drift is active (m²/s)

# -----------------------
# Simulation directories and timing setup
# -----------------------
script_start_time = time.time()

# Base directory for the initial cavity case
base_case_dir = "/home/musacim/simulation/openfoam/tutorials/incompressible/icoFoam/cavity/cavity"
output_base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Total time steps for simulation runs
time_steps = range(1, 600)  # Adjust as needed

# CSV file to log simulation parameters and outcomes
csv_filename = os.path.join(output_base_dir, "simulation_data_wdrift.csv")

# Ensure the output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# -----------------------
# Updated parameter generation function with rare data drift
# -----------------------
def get_parameters_for_time_step(t):
    """
    Generate simulation parameters for a given time step.
    Incorporates sinusoidal variations with mild noise,
    and triggers a drift event rarely to simulate data drift.
    """
    global drift_state, drift_duration_remaining

    # Base parameter values
    base_lid_velocity = 1.5   # mean lid_velocity (m/s)
    base_viscosity = 1e-3     # mean viscosity (m²/s)

    # Sinusoidal variation amplitudes (stable behavior)
    lid_amplitude = 0.1
    viscosity_amplitude = 5e-5

    # Decide if a drift event should occur
    if drift_state:
        # In drift state: use additional offsets
        current_drift_offset_velocity = DRIFT_OFFSET_VELOCITY
        current_drift_offset_viscosity = DRIFT_OFFSET_VISCOSITY
        drift_duration_remaining -= 1
        if drift_duration_remaining <= 0:
            drift_state = False  # Drift event ends
    else:
        # With a low probability, trigger a drift event
        if np.random.rand() < DRIFT_PROBABILITY:
            drift_state = True
            drift_duration_remaining = DRIFT_DURATION
            current_drift_offset_velocity = DRIFT_OFFSET_VELOCITY
            current_drift_offset_viscosity = DRIFT_OFFSET_VISCOSITY
        else:
            current_drift_offset_velocity = 0.0
            current_drift_offset_viscosity = 0.0

    # Calculate sinusoidal variations (stable part)
    period = 50
    angle = 2 * np.pi * (t / period)
    lid_velocity_variation = lid_amplitude * np.sin(angle)
    viscosity_variation = viscosity_amplitude * np.cos(angle)

    # Add small random noise
    lid_noise = np.random.uniform(-0.01, 0.01)
    viscosity_noise = np.random.uniform(-1e-5, 1e-5)

    # Combine base values, stable variations, noise, and any drift offsets
    lid_velocity = base_lid_velocity + lid_velocity_variation + lid_noise + current_drift_offset_velocity
    viscosity = base_viscosity + viscosity_variation + viscosity_noise + current_drift_offset_viscosity

    return lid_velocity, viscosity

# -----------------------
# Helper functions for case creation and modification (unchanged from your version)
# -----------------------
def create_case_directory(lid_velocity, viscosity, time_step):
    case_name = f"cavity_{lid_velocity:.2f}ms_{viscosity:.3e}_t{time_step}"
    case_dir = os.path.join(output_base_dir, case_name)
    subprocess.run(["cp", "-r", base_case_dir, case_dir], check=True)
    return case_dir

def modify_velocity(case_dir, lid_velocity):
    u_file_path = os.path.join(case_dir, "0/U")
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    movingWall
    {{
        type            fixedValue;
        value           uniform ({lid_velocity} 0 0);
    }}
    fixedWalls
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
"""
    with open(u_file_path, "w") as file:
        file.write(content)

def modify_viscosity(case_dir, viscosity):
    transport_file_path = os.path.join(case_dir, "constant/transportProperties")
    with open(transport_file_path, "r") as file:
        lines = file.readlines()
    with open(transport_file_path, "w") as file:
        for line in lines:
            if "nu" in line:
                file.write(f"nu              nu [0 2 -1 0 0 0 0] {viscosity};\n")
            else:
                file.write(line)

def run_simulation(case_dir):
    """
    Run the simulation using OpenFOAM commands.
    Returns the elapsed simulation time.
    """
    start_time = time.time()  # Start timing
    # Generate mesh
    subprocess.run(["blockMesh"], cwd=case_dir, check=True)
    # Run icoFoam solver and save output to log file
    log_file = os.path.join(case_dir, "log")
    with open(log_file, "w") as log:
        subprocess.run(["icoFoam"], cwd=case_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    return elapsed_time

# -----------------------
# Main simulation loop with data logging
# -----------------------
total_simulation_time = 0.0  # Accumulator for simulation runtimes

# Open CSV file for writing simulation details
with open(csv_filename, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header: time step, lid velocity, viscosity, simulation runtime (seconds)
    csv_writer.writerow(["time_step", "lid_velocity", "viscosity", "simulation_time_sec"])

    # Iterate through each time step
    for t in time_steps:
        # Get simulation parameters (with occasional drift)
        lid_velocity, viscosity = get_parameters_for_time_step(t)
        # Create and prepare the case directory
        case_dir = create_case_directory(lid_velocity, viscosity, t)
        modify_velocity(case_dir, lid_velocity)
        modify_viscosity(case_dir, viscosity)
        
        # Run the simulation and record its runtime
        sim_time = run_simulation(case_dir)
        total_simulation_time += sim_time
        
        # Log the simulation details to the CSV
        csv_writer.writerow([t, lid_velocity, viscosity, sim_time])
        
        print(f"Time Step {t}: lid_velocity = {lid_velocity:.2f} m/s, viscosity = {viscosity:.3e} m²/s, simulation_time = {sim_time:.2f} sec.")

print(f"\nTotal Simulation Time for {len(time_steps)} time steps: {total_simulation_time:.2f} seconds.")
script_end_time = time.time()
elapsed_time = script_end_time - script_start_time
print(f"Total Script Execution Time: {elapsed_time:.2f} seconds.")
