import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
import time  # Imported for timing measurements
script_start_time = time.time()
# Ensure the 'plots' directory exists
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv('/home/musacim/simulation/openfoam/simulation_data.csv')

# Define regions
regions = [
    (1, 25),   # Region 1: Time steps 1-25
    (26, 40),  # Region 2: Time steps 26-40
    (41, 55)   # Region 3: Time steps 41-55
]

# Initialize models and scaler
velocity_model = GradientBoostingRegressor(random_state=42)
pressure_model = GradientBoostingRegressor(random_state=42)
scaler = StandardScaler()

# Retraining parameters
accuracy_threshold = 80.0  # Retrain if accuracy falls below 80%
evaluation_interval = 5    # Evaluate performance every 5 time steps

def calculate_accuracy_cumulative(correct_predictions, total_predictions):
    """
    Calculate cumulative accuracy.
    """
    if total_predictions == 0:
        return 0.0
    return (correct_predictions / total_predictions) * 100

# Initialize global lists to store cumulative accuracy metrics
global_accuracy = {
    'time_steps': [],
    'velocity_accuracy': [],
    'pressure_accuracy': [],
    'retrain_steps': []
}

# Initialize global counters for cumulative accuracy
global_velocity_correct = 0
global_pressure_correct = 0
global_total_velocity = 0
global_total_pressure = 0

# Initialize total training time
total_training_time = 0.0

# Iterate through regions
for region_num, (start_time, end_time) in enumerate(regions, start=1):
    region_key = f'Region {region_num}'
    print(f"\n=== Processing {region_key}: Time Steps {start_time}-{end_time} ===\n")

    # Define initial training time steps
    initial_train_size = 10
    initial_train_time_steps = range(start_time, start_time + initial_train_size)

    # Initial training data
    training_data = data[data['lid_time_step'].isin(initial_train_time_steps)]
    X_train = training_data[['lid_velocity', 'viscosity']]
    y_velocity_train = training_data['velocity_magnitude']
    y_pressure_train = training_data['pressure']

    # Start timing for initial training
    train_start_time = time.time()

    # Fit scaler and train models
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    velocity_model.fit(X_train_scaled, y_velocity_train)
    pressure_model.fit(X_train_scaled, y_pressure_train)

    # End timing for initial training
    train_end_time = time.time()
    train_elapsed_time = train_end_time - train_start_time
    total_training_time += train_elapsed_time

    print(f"Initial training completed on time steps {list(initial_train_time_steps)} in {train_elapsed_time:.2f} seconds.")

    # Iterate through subsequent time steps
    for current_time_step in range(start_time + initial_train_size, end_time + 1):
        # Prepare test data
        test_data = data[data['lid_time_step'] == current_time_step]
        if test_data.empty:
            print(f"No data available for time step {current_time_step}.\n")
            continue

        X_test = test_data[['lid_velocity', 'viscosity']]
        y_velocity_test = test_data['velocity_magnitude'].values[0]
        y_pressure_test = test_data['pressure'].values[0]
        X_test_scaled = scaler.transform(X_test)

        # Predict
        y_velocity_pred = velocity_model.predict(X_test_scaled)[0]
        y_pressure_pred = pressure_model.predict(X_test_scaled)[0]

        # Determine if predictions are within tolerance
        velocity_within_tolerance = (np.abs(y_velocity_test - y_velocity_pred) / y_velocity_test <= 0.1) if y_velocity_test != 0 else False
        pressure_within_tolerance = (np.abs(y_pressure_test - y_pressure_pred) / y_pressure_test <= 0.1) if y_pressure_test != 0 else False

        # Update global counters
        global_velocity_correct += int(velocity_within_tolerance)
        global_pressure_correct += int(pressure_within_tolerance)
        global_total_velocity += 1
        global_total_pressure += 1

        # Calculate cumulative accuracy
        cumulative_velocity_accuracy = calculate_accuracy_cumulative(global_velocity_correct, global_total_velocity)
        cumulative_pressure_accuracy = calculate_accuracy_cumulative(global_pressure_correct, global_total_pressure)

        # Record metrics
        global_accuracy['time_steps'].append(current_time_step)
        global_accuracy['velocity_accuracy'].append(cumulative_velocity_accuracy)
        global_accuracy['pressure_accuracy'].append(cumulative_pressure_accuracy)

        print(f"Time Step: {current_time_step}")
        print(f"Velocity Model - Cumulative Accuracy: {cumulative_velocity_accuracy:.2f}%")
        print(f"Pressure Model - Cumulative Accuracy: {cumulative_pressure_accuracy:.2f}%\n")

        # Determine if it's time to evaluate and possibly retrain
        steps_since_initial = current_time_step - start_time
        if (steps_since_initial % evaluation_interval) == 0:
            print(f"--- Evaluating model performance up to time step {current_time_step} ---")

            # Prepare cumulative data for evaluation
            eval_time_steps = range(start_time, current_time_step + 1)
            eval_data = data[data['lid_time_step'].isin(eval_time_steps)]
            X_eval = eval_data[['lid_velocity', 'viscosity']]
            y_velocity_eval = eval_data['velocity_magnitude']
            y_pressure_eval = eval_data['pressure']

            # Scale evaluation data
            X_eval_scaled = scaler.transform(X_eval)

            # Predict on evaluation data
            y_velocity_eval_pred = velocity_model.predict(X_eval_scaled)
            y_pressure_eval_pred = pressure_model.predict(X_eval_scaled)

            # Calculate overall accuracy
            overall_velocity_correct = np.sum((np.abs(y_velocity_eval.values - y_velocity_eval_pred) / y_velocity_eval.values) <= 0.1)
            overall_pressure_correct = np.sum((np.abs(y_pressure_eval.values - y_pressure_eval_pred) / y_pressure_eval.values) <= 0.1)
            overall_velocity_total = len(y_velocity_eval)
            overall_pressure_total = len(y_pressure_eval)

            overall_velocity_accuracy = calculate_accuracy_cumulative(overall_velocity_correct, overall_velocity_total)
            overall_pressure_accuracy = calculate_accuracy_cumulative(overall_pressure_correct, overall_pressure_total)

            print(f"Overall Velocity Model - Cumulative Accuracy: {overall_velocity_accuracy:.2f}%")
            print(f"Overall Pressure Model - Cumulative Accuracy: {overall_pressure_accuracy:.2f}%\n")

            # Determine if retraining is needed
            if (overall_velocity_accuracy < accuracy_threshold) or (overall_pressure_accuracy < accuracy_threshold):
                print(f"Accuracy fell below {accuracy_threshold}%. Retraining models with data up to time step {current_time_step}.\n")

                # Start timing for retraining
                retrain_start_time = time.time()

                retrain_time_steps = range(start_time, current_time_step + 1)
                retrain_data = data[data['lid_time_step'].isin(retrain_time_steps)]
                X_retrain = retrain_data[['lid_velocity', 'viscosity']]
                y_velocity_retrain = retrain_data['velocity_magnitude']
                y_pressure_retrain = retrain_data['pressure']

                # Retrain models
                scaler.fit(X_retrain)
                X_retrain_scaled = scaler.transform(X_retrain)
                velocity_model.fit(X_retrain_scaled, y_velocity_retrain)
                pressure_model.fit(X_retrain_scaled, y_pressure_retrain)

                # End timing for retraining
                retrain_end_time = time.time()
                retrain_elapsed_time = retrain_end_time - retrain_start_time
                total_training_time += retrain_elapsed_time

                print(f"Retrained models on time steps {list(retrain_time_steps)} in {retrain_elapsed_time:.2f} seconds.\n")

                # Reset global accuracy counters after retraining
                global_velocity_correct = 0
                global_pressure_correct = 0
                global_total_velocity = 0
                global_total_pressure = 0

            else:
                print(f"Model performance is satisfactory. No retraining needed.\n")

print(f"Completed processing for all regions.\n")
print(f"Total Training Time (Initial Training + Retraining): {total_training_time:.2f} seconds.\n")



script_end_time = time.time()
elapsed_time = script_end_time - script_start_time
print(f"Total Script Execution Time (train_incremental.py): {elapsed_time:.2f} seconds.\n")
print(f"Total Script Execution Time (train_incremental.py): {elapsed_time:.2f} seconds.\n")
print(f"Total Script Execution Time (train_incremental.py): {elapsed_time:.2f} seconds.\n")






def plot_cumulative_accuracy(global_accuracy, plots_dir):
    """
    Plot cumulative accuracy for velocity and pressure models over time and save the plot.
    """
    plt.figure(figsize=(14, 7))

    plt.plot(
        global_accuracy['time_steps'],
        global_accuracy['velocity_accuracy'],
        label='Velocity Model',
        color='blue',
        linewidth=2,
        marker='o',
        markersize=5
    )

    plt.plot(
        global_accuracy['time_steps'],
        global_accuracy['pressure_accuracy'],
        label='Pressure Model',
        color='green',
        linewidth=2,
        marker='s',
        markersize=5
    )

    plt.title('Cumulative Accuracy of Velocity and Pressure Models Over Time', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Cumulative Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 100)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plot_path = os.path.join(plots_dir, 'cumulative_accuracy_over_time.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")

plot_cumulative_accuracy(global_accuracy, plots_dir)
