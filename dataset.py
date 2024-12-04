# dataset.py

import os
import math
import pandas as pd
import numpy as np

# Define the base directory where simulation data is located
base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Define the output CSV file
output_csv = "/home/musacim/simulation/openfoam/simulation_data.csv"

# Time steps (1-30)
time_steps = range(1, 31)

# Define regions for data labeling (optional)
def get_region(t):
    if 1 <= t <= 10:
        return "Region1_Training"
    elif 11 <= t <= 20:
        return "Region2_Shifting"
    elif 21 <= t <= 30:
        return "Region3_Shifting"
    else:
        return "Unknown"

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

# Helper functions to parse OpenFOAM files
def parse_openfoam_vector_field(file_path):
    """Parse a volVectorField from an OpenFOAM file and return a list of vectors (3D tuples)."""
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
                        pass  # Ignore lines that don't contain numeric data
                elif line == ");":
                    break
    return vectors

def parse_openfoam_scalar_field(file_path):
    """Parse a volScalarField from an OpenFOAM file and return a list of scalar values."""
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
                    pass  # Ignore lines that don't contain numeric data
    return scalars

def compute_velocity_magnitude(vectors):
    """Compute the magnitude of each vector and return a list of magnitudes."""
    return [math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in vectors]

# Collect data in a list
data_list = []

for t in time_steps:
    lid_velocity, viscosity = get_parameters_for_time_step(t)
    region = get_region(t)
    
    # Generate case directory names based on current parameters
    case_velocities = [lid_velocity]
    case_viscosities = [viscosity]
    
    for lid_velocity in case_velocities:
        for viscosity in case_viscosities:
            case_name = f"cavity_{lid_velocity:.2f}ms_{viscosity:.3e}_t{t}".replace('+', '')
            case_path = os.path.join(base_dir, case_name)
    
            # Check if the case directory exists
            if not os.path.exists(case_path):
                print(f"Case directory '{case_path}' does not exist. Skipping.")
                continue
    
            # Process the latest simulation time directory
            simulation_times = [d for d in os.listdir(case_path)
                                if os.path.isdir(os.path.join(case_path, d)) and d.replace('.', '', 1).isdigit()]
            if not simulation_times:
                print(f"No simulation time directories found in '{case_path}'. Skipping.")
                continue
            latest_sim_time = max(simulation_times, key=float)
    
            time_dir = os.path.join(case_path, latest_sim_time)
            u_file = os.path.join(time_dir, "U")
            p_file = os.path.join(time_dir, "p")
    
            # Check if U and p files exist
            if not os.path.exists(u_file) or not os.path.exists(p_file):
                print(f"Missing U or p file in '{time_dir}'. Skipping.")
                continue
    
            # Extract data from U and p files
            velocity_vectors = parse_openfoam_vector_field(u_file)
            pressure_data = parse_openfoam_scalar_field(p_file)
    
            # Calculate average velocity magnitude and pressure
            if velocity_vectors:
                velocity_magnitudes = compute_velocity_magnitude(velocity_vectors)
                avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
            else:
                avg_velocity_magnitude = None
    
            if pressure_data:
                avg_pressure = sum(pressure_data) / len(pressure_data)
            else:
                avg_pressure = None
    
            # Append data to the list
            if avg_velocity_magnitude is not None and avg_pressure is not None:
                data_list.append({
                    "lid_velocity": lid_velocity,
                    "viscosity": viscosity,
                    "lid_time_step": t,
                    "simulation_time": float(latest_sim_time),
                    "velocity_magnitude": avg_velocity_magnitude,
                    "pressure": avg_pressure,
                    "region": region
                })

# Create DataFrame and save to CSV
df = pd.DataFrame(data_list)
df.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")
