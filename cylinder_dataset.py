#!/usr/bin/env python3
import os
import pandas as pd
import argparse

# -----------------------
# Parse Command‑Line Arguments (optional)
# -----------------------
parser = argparse.ArgumentParser(
    description="Convert Hartmann simulation case directories into a CSV of simulation outputs."
)
parser.add_argument("--base_dir", type=str, default="/home/musacim/simulation/openfoam/mhd/hartmann_simulations",
                    help="Base directory where simulation case folders are stored.")
parser.add_argument("--output_csv", type=str, default="/home/musacim/simulation/openfoam/mhd/hartmann_simulation_output_data.csv",
                    help="Output CSV file path.")
args = parser.parse_args()

base_dir = args.base_dir
output_csv = args.output_csv

# List all case directories that follow the naming convention used in new_sim_mhd.py
case_dirs = [d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("hartmann_inlet")]

def parse_case_directory_name(case_name):
    # Expected format: hartmann_inlet{inlet_velocity}_B{B_magnitude}_t{time_step}
    try:
        parts = case_name.split("_")
        inlet_velocity = float(parts[1].replace("inlet", ""))
        B_magnitude = float(parts[2].replace("B", ""))
        time_step = int(parts[3].replace("t", ""))
        return inlet_velocity, B_magnitude, time_step
    except Exception as e:
        print(f"Error parsing directory {case_name}: {e}")
        return None, None, None

def parse_openfoam_vector_field(file_path):
    vectors = []
    with open(file_path, "r") as file:
        data_started = False
        for line in file:
            if "nonuniform List<vector>" in line:
                data_started = True
                continue
            if data_started:
                line = line.strip()
                if line.startswith("(") and line.endswith(")"):
                    components = line.strip("()").split()
                    try:
                        vector = tuple(float(c) for c in components)
                        vectors.append(vector)
                    except ValueError:
                        pass
                elif line == ");":
                    break
    return vectors

def parse_openfoam_scalar_field(file_path):
    scalars = []
    with open(file_path, 'r') as file:
        data_started = False
        for line in file:
            if "nonuniform List<scalar>" in line:
                data_started = True
                continue
            if data_started:
                line = line.strip()
                if line == ");":
                    break
                try:
                    scalars.append(float(line))
                except ValueError:
                    pass
    return scalars

def compute_velocity_magnitude(vectors):
    return [ (v[0]**2 + v[1]**2 + v[2]**2)**0.5 for v in vectors ]

data_list = []

for case_dir in case_dirs:
    inlet_velocity, B_magnitude, time_step = parse_case_directory_name(case_dir)
    if inlet_velocity is None:
        continue

    case_path = os.path.join(base_dir, case_dir)
    # Identify the latest time directory (assume numeric folder names)
    time_dirs = [d for d in os.listdir(case_path) if d.replace('.','',1).isdigit()]
    if not time_dirs:
        print(f"No time directories found in '{case_path}'.")
        continue
    latest_time = max(time_dirs, key=float)
    time_dir = os.path.join(case_path, latest_time)
    u_file = os.path.join(time_dir, "U")
    p_file = os.path.join(time_dir, "p")

    if not os.path.exists(u_file) or not os.path.exists(p_file):
        print(f"Missing U or p file in '{time_dir}'.")
        continue

    velocity_vectors = parse_openfoam_vector_field(u_file)
    pressure_data = parse_openfoam_scalar_field(p_file)

    if velocity_vectors:
        velocity_magnitudes = compute_velocity_magnitude(velocity_vectors)
        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
    else:
        avg_velocity_magnitude = None

    if pressure_data:
        avg_pressure = sum(pressure_data) / len(pressure_data)
    else:
        avg_pressure = None

    if avg_velocity_magnitude is not None and avg_pressure is not None:
        data_list.append({
            "time_step": time_step,
            "inlet_velocity": inlet_velocity,
            "B_magnitude": B_magnitude,
            "velocity_magnitude": avg_velocity_magnitude,
            "pressure": avg_pressure
        })

df = pd.DataFrame(data_list)
df = df.sort_values('time_step')
df.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")
