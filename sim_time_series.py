#guide for starting the environment 
#source /lib/openfoam/openfoam2406/etc/bashrc
#
#
#


import os
import subprocess
import numpy as np
import time  # Imported for timing measurements
script_start_time = time.time()


total_simulation_time = 0.0
# Base directory for the initial cavity case
base_case_dir = "/home/musacim/simulation/openfoam/tutorials/incompressible/icoFoam/cavity/cavity"
output_base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Total time steps
time_steps = range(1, 600)  # Time steps from 1 to 55 (25 + 15 + 15)
import numpy as np

def get_parameters_for_time_step(t):
    # No data drift scenario, but with stable periodic variations and mild noise.
    # Base values
    base_lid_velocity = 1.5      # mean lid_velocity
    base_viscosity = 1e-3        # mean viscosity

    # Introduce a sinusoidal variation (period = 50 steps) around the base values.
    # For instance, ±0.1 variation in lid_velocity and ±0.00005 in viscosity.
    lid_amplitude = 0.1
    viscosity_amplitude = 5e-5

    # Period for sine wave
    period = 50
    angle = 2 * np.pi * (t / period)

    # Sinusoidal variations
    lid_velocity_variation = lid_amplitude * np.sin(angle)
    viscosity_variation = viscosity_amplitude * np.cos(angle)

    # Optional small random noise (if desired), very small to not cause drift
    # e.g., noise ±0.01 for lid_velocity and ±1e-5 for viscosity
    lid_noise = np.random.uniform(-0.01, 0.01)
    viscosity_noise = np.random.uniform(-1e-5, 1e-5)

    # Combine everything
    lid_velocity = base_lid_velocity + lid_velocity_variation + lid_noise
    viscosity = base_viscosity + viscosity_variation + viscosity_noise

    return lid_velocity, viscosity




# Helper functions (same as before)
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

# Main loop to create and run simulations over time steps
os.makedirs(output_base_dir, exist_ok=True)

total_simulation_time = 0.0  # Initialize total simulation time

for t in time_steps:
    lid_velocity, viscosity = get_parameters_for_time_step(t)
    case_dir = create_case_directory(lid_velocity, viscosity, t)
    modify_velocity(case_dir, lid_velocity)
    modify_viscosity(case_dir, viscosity)
    
    sim_time = run_simulation(case_dir)  # Run simulation and get elapsed time
    total_simulation_time += sim_time  # Accumulate total simulation time
    
    print(f"Completed simulation for lid velocity {lid_velocity:.2f} m/s, viscosity {viscosity:.3e} m²/s at time step {t} in {sim_time:.2f} seconds.")

print(f"\nTotal Simulation Time for all {len(time_steps)} time steps: {total_simulation_time:.2f} seconds.")

script_end_time = time.time()
elapsed_time = script_end_time - script_start_time
print(f"Total Script Execution Time (sim_time_series.py): {elapsed_time:.2f} seconds.")
