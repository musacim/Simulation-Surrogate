import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib

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

# Loop over regions
for region_num, (start_time, end_time) in enumerate(regions, start=1):
    print(f"\n=== Processing Region {region_num}: Time Steps {start_time}-{end_time} ===\n")

    # Define initial training time steps
    initial_train_size = 10
    initial_train_time_steps = range(start_time, start_time + initial_train_size)

    # Initial training data
    training_data = data[data['lid_time_step'].isin(initial_train_time_steps)]
    X_train = training_data[['lid_velocity', 'viscosity']]
    y_velocity_train = training_data['velocity_magnitude']
    y_pressure_train = training_data['pressure']

    # Fit scaler and train models
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    velocity_model.fit(X_train_scaled, y_velocity_train)
    pressure_model.fit(X_train_scaled, y_pressure_train)
    print(f"Initial training completed on time steps {list(initial_train_time_steps)}.")

    # Initialize counters for cumulative accuracy
    velocity_correct = 0
    pressure_correct = 0
    total_velocity = 0
    total_pressure = 0

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
        velocity_within_tolerance = np.abs(y_velocity_test - y_velocity_pred) / y_velocity_test <= 0.1 if y_velocity_test != 0 else False
        pressure_within_tolerance = np.abs(y_pressure_test - y_pressure_pred) / y_pressure_test <= 0.1 if y_pressure_test != 0 else False

        # Update counters
        velocity_correct += int(velocity_within_tolerance)
        pressure_correct += int(pressure_within_tolerance)
        total_velocity += 1
        total_pressure += 1

        # Calculate cumulative accuracy
        cumulative_velocity_accuracy = calculate_accuracy_cumulative(velocity_correct, total_velocity)
        cumulative_pressure_accuracy = calculate_accuracy_cumulative(pressure_correct, total_pressure)

        # Calculate MAPE
        velocity_mape = mean_absolute_percentage_error([y_velocity_test], [y_velocity_pred])
        pressure_mape = mean_absolute_percentage_error([y_pressure_test], [y_pressure_pred])

        print(f"Time Step: {current_time_step}")
        print(f"Velocity Model - Cumulative Accuracy: {cumulative_velocity_accuracy:.2f}% (MAPE: {velocity_mape:.4f})")
        print(f"Pressure Model - Cumulative Accuracy: {cumulative_pressure_accuracy:.2f}% (MAPE: {pressure_mape:.4f})\n")

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
            overall_velocity_correct = np.sum(np.abs(y_velocity_eval.values - y_velocity_eval_pred) / y_velocity_eval.values <= 0.1)
            overall_pressure_correct = np.sum(np.abs(y_pressure_eval.values - y_pressure_eval_pred) / y_pressure_eval.values <= 0.1)
            overall_velocity_total = len(y_velocity_eval)
            overall_pressure_total = len(y_pressure_eval)

            overall_velocity_accuracy = calculate_accuracy_cumulative(overall_velocity_correct, overall_velocity_total)
            overall_pressure_accuracy = calculate_accuracy_cumulative(overall_pressure_correct, overall_pressure_total)

            # Calculate overall MAPE
            overall_velocity_mape = mean_absolute_percentage_error(y_velocity_eval, y_velocity_eval_pred)
            overall_pressure_mape = mean_absolute_percentage_error(y_pressure_eval, y_pressure_eval_pred)

            print(f"Overall Velocity Model - Cumulative Accuracy: {overall_velocity_accuracy:.2f}% (MAPE: {overall_velocity_mape:.4f})")
            print(f"Overall Pressure Model - Cumulative Accuracy: {overall_pressure_accuracy:.2f}% (MAPE: {overall_pressure_mape:.4f})\n")

            # Check if accuracy falls below the threshold
            if (overall_velocity_accuracy < accuracy_threshold) or (overall_pressure_accuracy < accuracy_threshold):
                print(f"Accuracy fell below {accuracy_threshold}%. Retraining models with data up to time step {current_time_step}.\n")

                # Retrain on all data up to current_time_step
                retrain_time_steps = range(start_time, current_time_step + 1)
                retrain_data = data[data['lid_time_step'].isin(retrain_time_steps)]
                X_retrain = retrain_data[['lid_velocity', 'viscosity']]
                y_velocity_retrain = retrain_data['velocity_magnitude']
                y_pressure_retrain = retrain_data['pressure']

                # Fit scaler and retrain models
                scaler.fit(X_retrain)
                X_retrain_scaled = scaler.transform(X_retrain)
                velocity_model.fit(X_retrain_scaled, y_velocity_retrain)
                pressure_model.fit(X_retrain_scaled, y_pressure_retrain)
                print(f"Retrained models on time steps {list(retrain_time_steps)}.\n")

                # Reset cumulative counters after retraining
                velocity_correct = 0
                pressure_correct = 0
                total_velocity = 0
                total_pressure = 0
            else:
                print(f"Model performance is satisfactory. No retraining needed.\n")

    print(f"Completed processing for Region {region_num}.\n")
