#!/usr/bin/env python3
import os
import math
import pandas as pd
import argparse

# -----------------------
# Parse Command‑Line Arguments (optional)
# -----------------------
parser = argparse.ArgumentParser(
    description="Convert simulation case directories into a CSV of simulation outputs."
)
parser.add_argument("--base_dir", type=str, default="/home/musacim/simulation/openfoam/cavity_simulations",
                    help="Base directory where simulation case folders are stored.")
parser.add_argument("--output_csv", type=str, default="/home/musacim/simulation/openfoam/simulation_output_data.csv",
                    help="Output CSV file path.")
args = parser.parse_args()

# -----------------------
# Directories and Files
# -----------------------
base_dir = args.base_dir
output_csv = args.output_csv

# List all case directories in the base directory (directories with '_t' in their name)
case_dirs = [d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d)) and '_t' in d]

def parse_case_directory_name(case_name):
    # Expected format: cavity_{lid_velocity}ms_{viscosity}_t{time_step}
    parts = case_name.split('_')
    if len(parts) != 4:
        print(f"Skipping directory '{case_name}' due to unexpected format")
        return None, None, None
    try:
        lid_velocity_str = parts[1][:-2]  # Remove 'ms'
        viscosity_str = parts[2]
        lid_time_step_str = parts[3][1:]  # Remove 't'

        lid_velocity = float(lid_velocity_str)
        viscosity = float(viscosity_str)
        lid_time_step = int(lid_time_step_str)

        return lid_velocity, viscosity, lid_time_step
    except (ValueError, IndexError) as e:
        print(f"Error parsing case directory '{case_name}': {e}")
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

# -----------------------
# Collect Data
# -----------------------
data_list = []

for case_dir in case_dirs:
    lid_velocity, viscosity, lid_time_step = parse_case_directory_name(case_dir)
    if lid_velocity is None:
        continue

    case_path = os.path.join(base_dir, case_dir)
    simulation_times = [d for d in os.listdir(case_path)
                        if os.path.isdir(os.path.join(case_path, d)) and d.replace('.', '', 1).isdigit()]
    if not simulation_times:
        print(f"No simulation time directories found in '{case_path}'.")
        continue
    latest_sim_time = max(simulation_times, key=float)
    time_dir = os.path.join(case_path, latest_sim_time)
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
            "lid_velocity": lid_velocity,
            "viscosity": viscosity,
            "lid_time_step": lid_time_step,
            "velocity_magnitude": avg_velocity_magnitude,
            "pressure": avg_pressure
        })

df = pd.DataFrame(data_list)
df = df.sort_values('lid_time_step')
df.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")
