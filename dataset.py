import os
import math
import pandas as pd

# Define the base directory where simulation data is located
base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Define the output CSV file
output_csv = "/home/musacim/simulation/openfoam/simulation_output_data.csv"

# List all case directories in the base directory
# Only include directories that contain '_t' indicating time step
case_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and '_t' in d]

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
    return [math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in vectors]

# Collect data in a list
data_list = []

for case_dir in case_dirs:
    # Parse parameters from directory name
    lid_velocity, viscosity, lid_time_step = parse_case_directory_name(case_dir)
    if lid_velocity is None:
        continue  # Skip cases where parsing failed

    # Path to the simulation case
    case_path = os.path.join(base_dir, case_dir)

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
            "velocity_magnitude": avg_velocity_magnitude,
            "pressure": avg_pressure
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data_list)

# Sort the DataFrame by 'lid_time_step' to maintain order
df = df.sort_values('lid_time_step')

df.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")
