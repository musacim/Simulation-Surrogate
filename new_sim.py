#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import time
import csv
import argparse
import shutil  # <-- Import shutil for directory removal

# -----------------------
# Parse Command‑Line Arguments
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simulation for a given range of time steps."
    )
    parser.add_argument("--start", type=int, default=1,
                        help="Starting time step (default: 1)")
    parser.add_argument("--end", type=int, default=600,
                        help="Ending time step (default: 600)")
    args = parser.parse_args()
else:
    # If imported, set default values.
    args = type("Args", (), {"start": 1, "end": 600})

# -----------------------
# Configuration
# -----------------------
script_start_time = time.time()

# Base directory for the initial cavity case
base_case_dir = "/home/musacim/simulation/openfoam/tutorials/incompressible/icoFoam/cavity/cavity"
output_base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Output CSV file (this script writes a temporary CSV; the final CSV is produced by dataset.py)
csv_filename = os.path.join(output_base_dir, "simulation_data_two_ramps.csv")

# -----------------------
# Ramp/Drift Intervals and Base Values
# -----------------------
RAMP1_START, RAMP1_END = 200, 300
RAMP2_START, RAMP2_END = 400, 500

BASE_LID_VEL_1 = 1.5   # Stable velocity before ramp 1
BASE_LID_VEL_2 = 2.0   # Stable velocity after ramp 1
BASE_LID_VEL_3 = 2.5   # Stable velocity after ramp 2

BASE_VISCOSITY = 1e-3

# -----------------------
# Helper: Piecewise Linear Drift
# -----------------------
def piecewise_lid_velocity(t):
    if t < RAMP1_START:
        return BASE_LID_VEL_1
    if RAMP1_START <= t < RAMP1_END:
        fraction = (t - RAMP1_START) / (RAMP1_END - RAMP1_START)
        return BASE_LID_VEL_1 + fraction * (BASE_LID_VEL_2 - BASE_LID_VEL_1)
    if RAMP1_END <= t < RAMP2_START:
        return BASE_LID_VEL_2
    if RAMP2_START <= t < RAMP2_END:
        fraction = (t - RAMP2_START) / (RAMP2_END - RAMP2_START)
        return BASE_LID_VEL_2 + fraction * (BASE_LID_VEL_3 - BASE_LID_VEL_2)
    return BASE_LID_VEL_3

# -----------------------
# Parameter Generation Function
# -----------------------
def get_parameters_for_time_step(t):
    base_lid = piecewise_lid_velocity(t)
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
# Clean Up Existing Output Directory
# -----------------------
if os.path.exists(output_base_dir):
    shutil.rmtree(output_base_dir)  # Remove existing output directory and its contents
os.makedirs(output_base_dir, exist_ok=True)

# -----------------------
# Main Loop: Run only for time steps in the provided range.
# -----------------------
total_simulation_time = 0.0

with open(csv_filename, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_step", "lid_velocity", "viscosity", "simulation_time_sec"])

    for t in range(args.start, args.end + 1):
        lid_velocity, viscosity = get_parameters_for_time_step(t)
        case_dir = create_case_directory(lid_velocity, viscosity, t)
        modify_velocity(case_dir, lid_velocity)
        modify_viscosity(case_dir, viscosity)

        sim_time = run_simulation(case_dir)
        total_simulation_time += sim_time

        csv_writer.writerow([t, lid_velocity, viscosity, sim_time])
        print(f"Time Step {t}: lid_velocity={lid_velocity:.2f}, viscosity={viscosity:.3e}, sim_time={sim_time:.2f}s")

print(f"\nTotal Simulation Time for steps {args.start} to {args.end}: {total_simulation_time:.2f} s")
script_end_time = time.time()
elapsed_time = script_end_time - script_start_time
print(f"Total Script Execution Time: {elapsed_time:.2f} s")
