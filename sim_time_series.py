import os
import subprocess
import numpy as np

# Base directory for the initial cavity case
base_case_dir = "/home/musacim/simulation/openfoam/tutorials/incompressible/icoFoam/cavity/cavity"
output_base_dir = "/home/musacim/simulation/openfoam/cavity_simulations"

# Total time steps
time_steps = range(1, 56)  # Time steps from 1 to 55 (25 + 15 + 15)

# Function to generate parameters for time steps with gradual shifts
def get_parameters_for_time_step(t):
    np.random.seed(t)  # Ensure reproducibility for each time step
    if t <= 25:
        # Region 1: Stable parameters with slight random noise
        lid_velocity = 1.0 + 0.03 * (t - 1) + np.random.uniform(-0.02, 0.02)
        viscosity = 1e-3 + 2e-5 * (t - 1) + np.random.uniform(-1e-5, 1e-5)
    elif 26 <= t <= 40:
        # Region 2: Noticeable jump in range with gradual change
        lid_velocity = 2.0 + 0.1 * (t - 25) + np.random.uniform(-0.03, 0.03)
        viscosity = 5e-4 + 1e-5 * (t - 25) + np.random.uniform(-2e-5, 2e-5)
    else:
        # Region 3: Distinct change with larger variability
        lid_velocity = 3.5 + 0.08 * (t - 40) + np.random.uniform(-0.05, 0.05)
        viscosity = 3e-4 - 1e-5 * (t - 40) + np.random.uniform(-2e-5, 2e-5)
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
    # Generate mesh
    subprocess.run(["blockMesh"], cwd=case_dir, check=True)
    # Run icoFoam solver and save output to log file
    log_file = os.path.join(case_dir, "log")
    with open(log_file, "w") as log:
        subprocess.run(["icoFoam"], cwd=case_dir, stdout=log, stderr=subprocess.STDOUT, check=True)

# Main loop to create and run simulations over time steps
os.makedirs(output_base_dir, exist_ok=True)
for t in time_steps:
    lid_velocity, viscosity = get_parameters_for_time_step(t)
    case_dir = create_case_directory(lid_velocity, viscosity, t)
    modify_velocity(case_dir, lid_velocity)
    modify_viscosity(case_dir, viscosity)
    run_simulation(case_dir)
    print(f"Completed simulation for lid velocity {lid_velocity:.2f} m/s, viscosity {viscosity:.3e} m²/s at time step {t}")
