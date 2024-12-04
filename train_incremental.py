import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
velocity_model = RandomForestRegressor(random_state=42)
pressure_model = RandomForestRegressor(random_state=42)
scaler = StandardScaler()

# Define retraining intervals
train_interval = 5  # Train after every 5 time steps
first_training_point = 15  # Train the first surrogate model at time step 15

# Tolerance for accuracy calculation (percentage)
accuracy_tolerance = 0.05  # Within 5% of the true value

def calculate_accuracy(y_true, y_pred, tolerance):
    """Calculate accuracy as the percentage of predictions within a tolerance of the true value."""
    return np.mean(np.abs(y_true - y_pred) / y_true <= tolerance) * 100

# Loop over regions
for region_num, (start_time, end_time) in enumerate(regions, start=1):
    print(f"\n=== Processing Region {region_num}: Time Steps {start_time}-{end_time} ===\n")

    # Determine the initial training time step
    if region_num == 1:
        current_time_step = first_training_point
    else:
        current_time_step = start_time + train_interval

    while current_time_step <= end_time:
        # Use all data up to the current time step for training
        train_time_steps = range(start_time, current_time_step)
        test_time_steps = range(current_time_step, min(current_time_step + train_interval, end_time + 1))

        # Prepare training data
        training_data = data[data['lid_time_step'].isin(train_time_steps)]
        X_train = training_data[['lid_velocity', 'viscosity']]
        y_velocity_train = training_data['velocity_magnitude']
        y_pressure_train = training_data['pressure']

        # Fit scaler and transform training data
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        # Train surrogate models with all data so far
        velocity_model.fit(X_train_scaled, y_velocity_train)
        pressure_model.fit(X_train_scaled, y_pressure_train)

        # Save the models and scaler
        joblib.dump({"velocity": velocity_model, "pressure": pressure_model}, "/home/musacim/simulation/openfoam/surrogate_model.joblib")
        joblib.dump(scaler, "/home/musacim/simulation/openfoam/scaler.joblib")
        print(f"Trained surrogate models with data up to time step {current_time_step - 1}.")

        # Prepare test data
        test_data = data[data['lid_time_step'].isin(test_time_steps)]
        if test_data.shape[0] < 1:
            print(f"Not enough test samples to compute accuracy at time steps {list(test_time_steps)}.\n")
        else:
            X_test = test_data[['lid_velocity', 'viscosity']]
            y_velocity_test = test_data['velocity_magnitude']
            y_pressure_test = test_data['pressure']

            # Transform test data using the scaler
            X_test_scaled = scaler.transform(X_test)

            # Predict using the surrogate models
            y_velocity_pred = velocity_model.predict(X_test_scaled)
            y_pressure_pred = pressure_model.predict(X_test_scaled)

            # Evaluate the models
            velocity_accuracy = calculate_accuracy(y_velocity_test, y_velocity_pred, accuracy_tolerance)
            pressure_accuracy = calculate_accuracy(y_pressure_test, y_pressure_pred, accuracy_tolerance)

            print(f"Test Time Steps: {list(test_time_steps)}")
            print(f"Velocity Model - Accuracy: {velocity_accuracy:.2f}%")
            print(f"Pressure Model - Accuracy: {pressure_accuracy:.2f}%\n")

        # Increment the current time step by the training interval
        current_time_step += train_interval

    print(f"Completed processing for Region {region_num}.\n")
