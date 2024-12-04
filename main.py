# main.py
import subprocess

# Step 1: Run simulations over time
subprocess.run(["python", "sim_time_series.py"])

# Step 2: Process the simulation data into a dataset
subprocess.run(["python", "dataset.py"])

# Step 3: Train the surrogate model incrementally
subprocess.run(["python", "train_incremental.py"])

# Step 4: Compare the simulation and surrogate model
subprocess.run(["python", "compare.py"])
