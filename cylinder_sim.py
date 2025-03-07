#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import time
import csv
import argparse
import shutil
import sys

# ------------------------------
# CONFIGURATION PARAMETERS
# ------------------------------
BASE_CASE_DIR = "/home/musacim/simulation/openfoam/tutorials/electromagnetics/mhdFoam/hartmann"
OUTPUT_BASE_DIR = "/home/musacim/simulation/openfoam/mhd/hartmann_simulations"
CSV_FILENAME = os.path.join(OUTPUT_BASE_DIR, "hartmann_simulation_data.csv")

def get_parameters_for_time_step(t, total_steps):
    """
    For the Hartmann MHD case:
      - Inlet velocity is fixed at 1 m/s.
      - B_magnitude (magnetic flux density) is 20 T for t < RAMP_START,
        then ramps linearly to 1 T, and remains 1 T afterward.
    """
    RAMP_START = int(total_steps * 0.5)
    RAMP_END   = int(total_steps * 0.6)
    inlet_velocity = 1.0

    if t < RAMP_START:
        B_magnitude = 20.0
    elif t < RAMP_END:
        fraction = (t - RAMP_START) / (RAMP_END - RAMP_START)
        B_magnitude = 20.0 + fraction * (1.0 - 20.0)
    else:
        B_magnitude = 1.0

    # Optional noise to B
    noise = np.random.uniform(-0.1, 0.1)
    B_magnitude += noise
    return inlet_velocity, B_magnitude

def create_case_directory(inlet_velocity, B_magnitude, time_step):
    case_name = f"hartmann_inlet{inlet_velocity:.2f}_B{B_magnitude:.2f}_t{time_step}"
    case_dir = os.path.join(OUTPUT_BASE_DIR, case_name)
    # Copy the base case into the new directory
    subprocess.run(["cp", "-r", BASE_CASE_DIR + "/", case_dir], check=True)
    return case_dir

def modify_inlet(case_dir, inlet_velocity):
    """
    Modify the inlet velocity in the 0/U file.
    Assumes the base case has a file named "0/U".
    """
    u_file_path = os.path.join(case_dir, "0", "U")
    if not os.path.exists(u_file_path):
        print(f"Error: The file {u_file_path} does not exist. Please ensure the base case has a '0/U' file.")
        sys.exit(1)

    try:
        with open(u_file_path, "r") as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading {u_file_path}: {e}")
        sys.exit(1)

    # Replace the inlet velocity string; adjust as needed for your base file format.
    new_inlet = f"uniform ({inlet_velocity} 0 0)"
    if "uniform (1 0 0)" in content:
        content = content.replace("uniform (1 0 0)", new_inlet)
    else:
        print(f"Warning: 'uniform (1 0 0)' not found in {u_file_path}. No changes made.")

    try:
        with open(u_file_path, "w") as file:
            file.write(content)
    except Exception as e:
        print(f"Error writing to {u_file_path}: {e}")
        sys.exit(1)

def modify_magnetic_field(case_dir, B_magnitude):
    """
    Modify the magnetic field in the 0/B file.
    Assumes the base case has a file named "0/B".
    """
    B_file_path = os.path.join(case_dir, "0", "B")
    if not os.path.exists(B_file_path):
        print(f"Error: The file {B_file_path} does not exist. Please ensure the base case has a '0/B' file.")
        sys.exit(1)

    try:
        with open(B_file_path, "r") as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading {B_file_path}: {e}")
        sys.exit(1)

    new_B = f"uniform (0 {B_magnitude} 0)"
    if "uniform (0 20 0)" in content:
        content = content.replace("uniform (0 20 0)", new_B)
    else:
        print(f"Warning: 'uniform (0 20 0)' not found in {B_file_path}. No changes made.")

    try:
        with open(B_file_path, "w") as file:
            file.write(content)
    except Exception as e:
        print(f"Error writing to {B_file_path}: {e}")
        sys.exit(1)

def run_simulation(case_dir):
    """
    Run blockMesh and mhdFoam, capturing the runtime.
    """
    start_time = time.time()
    subprocess.run(["blockMesh"], cwd=case_dir, check=True)
    log_file = os.path.join(case_dir, "log")
    with open(log_file, "w") as log:
        subprocess.run(["mhdFoam"], cwd=case_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.time() - start_time

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Hartmann MHD simulation for a given range of time steps.")
    parser.add_argument("--start", type=int, default=1, help="Starting time step (default: 1)")
    parser.add_argument("--end", type=int, default=100, help="Ending time step (default: 100)")
    parser.add_argument("--total_steps", type=int, default=1000, help="Total steps (from workflow2.py)")
    args = parser.parse_args()

    start = args.start
    end = args.end
    total_steps = args.total_steps

    # Clean out old simulation directories if they exist
    if os.path.exists(OUTPUT_BASE_DIR):
        shutil.rmtree(OUTPUT_BASE_DIR)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    total_sim_time = 0.0

    with open(CSV_FILENAME, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "time_step",
            "inlet_velocity",
            "B_magnitude",
            "velocity_magnitude",
            "pressure",
            "simulation_time_sec"
        ])

        for t in range(start, end + 1):
            inlet_velocity, B_magnitude = get_parameters_for_time_step(t, total_steps)
            case_dir = create_case_directory(inlet_velocity, B_magnitude, t)
            modify_inlet(case_dir, inlet_velocity)
            modify_magnetic_field(case_dir, B_magnitude)

            sim_time = run_simulation(case_dir)
            total_sim_time += sim_time

            # Placeholder values for velocity_magnitude and pressure
            avg_velocity_magnitude = 1.0
            avg_pressure = 100.0

            csv_writer.writerow([
                t,
                inlet_velocity,
                B_magnitude,
                avg_velocity_magnitude,
                avg_pressure,
                sim_time
            ])
            print(f"Time Step {t}: inlet_velocity={inlet_velocity:.2f}, B_magnitude={B_magnitude:.2f}, sim_time={sim_time:.2f} s")

    print(f"\nTotal Simulation Time for steps {start} to {end}: {total_sim_time:.2f} s")
    print(f"Script Execution Time: {time.time() - args.start:.2f} s")

if __name__ == "__main__":
    main()
