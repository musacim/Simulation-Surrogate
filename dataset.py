import os
import math
import pandas as pd
import numpy as np

# Define the base directory where simulation data is located
base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Define the output CSV file
output_csv = "/home/musacim/simulation/openfoam/simulation_data.csv"

# Time steps (assuming time steps from 1 to 10)
time_steps = range(1, 11)

# Function to generate parameters for time step, matching sim_time_series.py
def get_parameters_for_time_step(t):
    if t <= 5:
        lid_velocities = np.arange(1.0, 3.1, 0.5)  # Lid velocities from 1.0 to 3.0 m/s
        viscosities = np.logspace(-3, -2, num=5)    # Viscosities from 1e-3 to 1e-2 m²/s
    else:
        lid_velocities = np.arange(3.5, 5.1, 0.5)  # Lid velocities from 3.5 to 5.0 m/s
        viscosities = np.logspace(-4, -3, num=5)    # Viscosities from 1e-4 to 1e-3 m²/s
    return lid_velocities, viscosities

# Generate list of case directories to process
case_dirs = []
for t in time_steps:
    lid_velocities, viscosities = get_parameters_for_time_step(t)
    for lid_velocity in lid_velocities:
        for viscosity in viscosities:
            case_name = f"cavity_{lid_velocity:.2f}ms_{viscosity:.3e}_t{t}".replace('+', '')
            case_dirs.append(case_name)

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
    return [math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in vectors]

# Collect data in a list
data_list = []

for case_dir in case_dirs:
    # Parse parameters from directory name
    parts = case_dir.split('_')
    try:
        lid_velocity = float(parts[1][:-2])  # Remove 'ms'
        viscosity = float(parts[2])
        lid_time_step = int(parts[3][1:])  # Remove 't'
    except (IndexError, ValueError) as e:
        print(f"Error parsing case directory '{case_dir}': {e}")
        continue

    # Path to the simulation case
    case_path = os.path.join(base_dir, case_dir)

    # Check if the case directory exists
    if not os.path.exists(case_path):
        print(f"Case directory '{case_path}' does not exist.")
        continue

    # Process the latest simulation time directory
    simulation_times = [d for d in os.listdir(case_path)
                        if os.path.isdir(os.path.join(case_path, d)) and d.replace('.', '', 1).isdigit()]
    if not simulation_times:
        print(f"No simulation time directories found in '{case_path}'.")
        continue
    latest_sim_time = max(simulation_times, key=float)

    time_dir = os.path.join(case_path, latest_sim_time)
    u_file = os.path.join(time_dir, "U")
    p_file = os.path.join(time_dir, "p")

    # Check if U and p files exist
    if not os.path.exists(u_file) or not os.path.exists(p_file):
        print(f"Missing U or p file in '{time_dir}'.")
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
            "lid_time_step": lid_time_step,
            "simulation_time": float(latest_sim_time),
            "velocity_magnitude": avg_velocity_magnitude,
            "pressure": avg_pressure
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data_list)
df.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")
