#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import time
import csv
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BFS simulation for a given range of time steps with relaxed settings for faster execution."
    )
    parser.add_argument("--start", type=int, default=1, help="Starting time step (default: 1)")
    parser.add_argument("--end",   type=int, default=100, help="Ending time step (default: 100)")
    args = parser.parse_args()
else:
    # If imported, set default values.
    args = type("Args", (), {"start": 1, "end": 100})

# TOTAL_STEPS should match your overall workflow
TOTAL_STEPS = 1000

# Update BASE_CASE_DIR to your BFS base case location
BASE_CASE_DIR = "/home/musacim/simulation/openfoam/tutorials/incompressible/simpleFoam/backwardFacingStep2D"
OUTPUT_BASE_DIR  = "/home/musacim/simulation/openfoam/bfs_simulations"
CSV_FILENAME     = os.path.join(OUTPUT_BASE_DIR, "bfs_simulation_data_relaxed.csv")

# For faster runs, you might want to use a lower inlet velocity (or adjust as needed)
BASE_INLET_VEL_1   = 1.0
BASE_INLET_VEL_2   = 1.5
BASE_INLET_VEL_3   = 2.0
BASE_VISCOSITY     = 1e-3

def get_parameters_for_time_step(t):
    """
    Uses a piecewise ramp with sinusoidal variation.
    """
    RAMP1_START = int(TOTAL_STEPS * 0.250)
    RAMP1_END   = int(TOTAL_STEPS * 0.300)
    RAMP2_START = int(TOTAL_STEPS * 0.625)
    RAMP2_END   = int(TOTAL_STEPS * 0.675)

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

    period              = 50
    angle               = 2 * np.pi * (t / period)
    inlet_amplitude     = 0.1
    viscosity_amplitude = 5e-5
    inlet_variation     = inlet_amplitude * np.sin(angle)
    viscosity_variation = viscosity_amplitude * np.cos(angle)
    inlet_noise         = np.random.uniform(-0.01, 0.01)
    viscosity_noise     = np.random.uniform(-1e-5, 1e-5)

    inlet_velocity = base_inlet + inlet_variation + inlet_noise
    viscosity      = BASE_VISCOSITY + viscosity_variation + viscosity_noise
    return inlet_velocity, viscosity

def create_case_directory(inlet_velocity, viscosity, time_step):
    case_name = f"bfs_{inlet_velocity:.2f}ms_{viscosity:.3e}_t{time_step}"
    case_dir = os.path.join(OUTPUT_BASE_DIR, case_name)
    subprocess.run(["cp", "-r", BASE_CASE_DIR, case_dir], check=True)
    
    # Rename the initial conditions folder if necessary
    folder_0 = os.path.join(case_dir, "0")
    if not os.path.exists(folder_0):
        for alt in ["0.org", "0.orig"]:
            alt_path = os.path.join(case_dir, alt)
            if os.path.exists(alt_path):
                os.rename(alt_path, folder_0)
                print(f"Renamed '{alt_path}' to '{folder_0}'.")
                break
        else:
            os.makedirs(folder_0, exist_ok=True)
            print(f"Created empty '0' directory in {case_dir}. Please add initial conditions.")
    return case_dir

def modify_velocity(case_dir, inlet_velocity):
    u_file_path = os.path.join(case_dir, "0", "U")
    # Here we update only the inlet patch while preserving the original structure.
    # If you have an original file to use as reference, consider reading it and updating.
    content = f"""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2406                                  |
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
    inlet
    {{
        type            fixedValue;
        value           uniform ({inlet_velocity} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    lowerWallStartup
    {{
        type            symmetryPlane;
    }}
    upperWallStartup
    {{
        type            symmetryPlane;
    }}
    upperWall
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    lowerWall
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    front
    {{
        type            empty;
    }}
    back
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
"""
    with open(u_file_path, "w") as file:
        file.write(content)

def modify_viscosity(case_dir, viscosity):
    transport_file_path = os.path.join(case_dir, "constant", "transportProperties")
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
        # Use simpleFoam solver; note that we assume fvSolution and controlDict have been relaxed for faster runs.
        subprocess.run(["simpleFoam"], cwd=case_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.time() - start_time

# Clean out old directories before running
if os.path.exists(OUTPUT_BASE_DIR):
    shutil.rmtree(OUTPUT_BASE_DIR)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

total_sim_time = 0.0

with open(CSV_FILENAME, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["bfs_time_step", "inlet_velocity", "viscosity", "simulation_time_sec"])

    for t in range(args.start, args.end + 1):
        inlet_velocity, viscosity = get_parameters_for_time_step(t)
        case_dir = create_case_directory(inlet_velocity, viscosity, t)
        modify_velocity(case_dir, inlet_velocity)
        modify_viscosity(case_dir, viscosity)

        sim_time = run_simulation(case_dir)
        total_sim_time += sim_time

        csv_writer.writerow([t, inlet_velocity, viscosity, sim_time])
        print(f"Time Step {t}: inlet_velocity={inlet_velocity:.2f}, viscosity={viscosity:.3e}, sim_time={sim_time:.2f}s")

print(f"\nTotal Simulation Time for steps {args.start} to {args.end}: {total_sim_time:.2f} s")
