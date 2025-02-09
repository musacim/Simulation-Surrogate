import os
import subprocess
import numpy as np
import time
import csv

# -----------------------
# Configuration
# -----------------------
script_start_time = time.time()

# Base directory for the initial cavity case
base_case_dir = "/home/musacim/simulation/openfoam/tutorials/incompressible/icoFoam/cavity/cavity"
output_base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Output CSV file
csv_filename = os.path.join(output_base_dir, "simulation_data_two_ramps.csv")

# Total time steps for simulation runs
time_steps = range(1, 601)  # 1..600

# -----------------------
# Ramp/Drift Intervals
# -----------------------
# We'll define two ramps:
#  1) Ramp 1: from time=200 to 300 (from 1.5 to 2.0)
#  2) Ramp 2: from time=400 to 500 (from 2.0 to 2.5)
# Everything else is stable at either 1.5, 2.0, or 2.5.
RAMP1_START, RAMP1_END = 200, 300
RAMP2_START, RAMP2_END = 400, 500

BASE_LID_VEL_1 = 1.5   # Stable velocity before ramp 1
BASE_LID_VEL_2 = 2.0   # Stable velocity after ramp 1
BASE_LID_VEL_3 = 2.5   # Stable velocity after ramp 2 (final)

# Viscosity remains mostly constant, plus minor sinusoidal + random noise
BASE_VISCOSITY = 1e-3

# -----------------------
# Helper: Piecewise Linear Drift
# -----------------------
def piecewise_lid_velocity(t):
    """
    Piecewise function:
      - [1..200) : stable at 1.5
      - [200..300): ramp from 1.5 -> 2.0
      - [300..400): stable at 2.0
      - [400..500): ramp from 2.0 -> 2.5
      - [500..600]: stable at 2.5
    """
    # First interval (before Ramp 1)
    if t < RAMP1_START:
        return BASE_LID_VEL_1

    # Ramp 1 interval
    if RAMP1_START <= t < RAMP1_END:
        fraction = (t - RAMP1_START) / (RAMP1_END - RAMP1_START)
        return BASE_LID_VEL_1 + fraction * (BASE_LID_VEL_2 - BASE_LID_VEL_1)

    # Stable after Ramp 1, before Ramp 2
    if RAMP1_END <= t < RAMP2_START:
        return BASE_LID_VEL_2

    # Ramp 2 interval
    if RAMP2_START <= t < RAMP2_END:
        fraction = (t - RAMP2_START) / (RAMP2_END - RAMP2_START)
        return BASE_LID_VEL_2 + fraction * (BASE_LID_VEL_3 - BASE_LID_VEL_2)

    # Final stable region (after Ramp 2)
    return BASE_LID_VEL_3

# -----------------------
# Parameter Generation Function
# -----------------------
def get_parameters_for_time_step(t):
    """
    Generate parameters (lid_velocity, viscosity) with two separate ramp intervals.
    Also add sinusoidal variation and small noise.
    """
    # 1) Base velocity from the piecewise function
    base_lid = piecewise_lid_velocity(t)

    # 2) Sinusoidal variation
    period = 50
    angle = 2 * np.pi * (t / period)
    lid_amplitude = 0.1
    viscosity_amplitude = 5e-5

    lid_variation = lid_amplitude * np.sin(angle)
    viscosity_variation = viscosity_amplitude * np.cos(angle)

    # 3) Small random noise
    lid_noise = np.random.uniform(-0.01, 0.01)
    viscosity_noise = np.random.uniform(-1e-5, 1e-5)

    # 4) Combine for final parameters
    lid_velocity = base_lid + lid_variation + lid_noise
    viscosity = BASE_VISCOSITY + viscosity_variation + viscosity_noise

    return lid_velocity, viscosity

# -----------------------
# OpenFOAM Case Setup and Execution
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
    start_time = time.time()
    subprocess.run(["blockMesh"], cwd=case_dir, check=True)
    log_file = os.path.join(case_dir, "log")
    with open(log_file, "w") as log:
        subprocess.run(["icoFoam"], cwd=case_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.time() - start_time

# -----------------------
# Main Loop
# -----------------------
os.makedirs(output_base_dir, exist_ok=True)
total_simulation_time = 0.0

with open(csv_filename, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_step", "lid_velocity", "viscosity", "simulation_time_sec"])

    for t in time_steps:
        lid_velocity, viscosity = get_parameters_for_time_step(t)
        case_dir = create_case_directory(lid_velocity, viscosity, t)
        modify_velocity(case_dir, lid_velocity)
        modify_viscosity(case_dir, viscosity)

        sim_time = run_simulation(case_dir)
        total_simulation_time += sim_time

        csv_writer.writerow([t, lid_velocity, viscosity, sim_time])
        print(f"Time Step {t}: lid_velocity={lid_velocity:.2f}, viscosity={viscosity:.3e}, sim_time={sim_time:.2f}s")

print(f"\nTotal Simulation Time for {len(time_steps)} steps: {total_simulation_time:.2f} s")
script_end_time = time.time()
elapsed_time = script_end_time - script_start_time
print(f"Total Script Execution Time: {elapsed_time:.2f} s")
