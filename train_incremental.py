import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('/home/musacim/simulation/openfoam/simulation_data.csv')

# Sort data by time step
data = data.sort_values('lid_time_step')

# Initialize models and scaler
velocity_model = RandomForestRegressor(random_state=42)
pressure_model = RandomForestRegressor(random_state=42)
scaler = StandardScaler()

# Define accuracy threshold
accuracy_threshold = 95.0  # 95%

# Split data into initial training data (Region 1) and new data (Regions 2 and 3)
initial_training_data = data[data['lid_time_step'] <= 4]
new_data = data[data['lid_time_step'] > 4]

# Prepare the initial training data
X_train_initial = initial_training_data[['lid_velocity', 'viscosity']]
y_velocity_train_initial = initial_training_data['velocity_magnitude']
y_pressure_train_initial = initial_training_data['pressure']

# Scale the data
scaler.fit(X_train_initial)
X_train_initial_scaled = scaler.transform(X_train_initial)

# Train the models on initial data
velocity_model.fit(X_train_initial_scaled, y_velocity_train_initial)
pressure_model.fit(X_train_initial_scaled, y_pressure_train_initial)

# Save the initial models and scaler
joblib.dump({"velocity": velocity_model, "pressure": pressure_model}, "/home/musacim/simulation/openfoam/surrogate_model.joblib")
joblib.dump(scaler, "/home/musacim/simulation/openfoam/scaler.joblib")

print("Initial surrogate models trained on data from Region 1 (Time Steps 1-4).\n")

# Loop over each time step in new data
unique_time_steps = new_data['lid_time_step'].unique()
for current_time in unique_time_steps:
    # Get data for the current time step
    current_data = new_data[new_data['lid_time_step'] == current_time]
    
    # Prepare the data
    X_current = current_data[['lid_velocity', 'viscosity']]
    y_velocity_current = current_data['velocity_magnitude']
    y_pressure_current = current_data['pressure']
    
    # Scale the data using the existing scaler
    X_current_scaled = scaler.transform(X_current)
    
    # Predict using the surrogate models
    y_velocity_pred = velocity_model.predict(X_current_scaled)
    y_pressure_pred = pressure_model.predict(X_current_scaled)
    
    # Evaluate the models
    velocity_r2 = r2_score(y_velocity_current, y_velocity_pred)
    pressure_r2 = r2_score(y_pressure_current, y_pressure_pred)
    
    velocity_accuracy = velocity_r2 * 100
    pressure_accuracy = pressure_r2 * 100
    
    print(f"Time Step: {current_time}")
    print(f"Velocity Model - Accuracy: {velocity_accuracy:.2f}%")
    print(f"Pressure Model - Accuracy: {pressure_accuracy:.2f}%\n")
    
    # Check if accuracy drops below threshold
    if velocity_accuracy < accuracy_threshold or pressure_accuracy < accuracy_threshold:
        print("Accuracy below threshold. Retraining the model with new data...\n")
        # Retrain the models with all available data up to current time
        combined_data = data[data['lid_time_step'] <= current_time]
        X_combined = combined_data[['lid_velocity', 'viscosity']]
        y_velocity_combined = combined_data['velocity_magnitude']
        y_pressure_combined = combined_data['pressure']
        
        # Update scaler with new data
        scaler.fit(X_combined)
        X_combined_scaled = scaler.transform(X_combined)
        
        # Retrain the models
        velocity_model.fit(X_combined_scaled, y_velocity_combined)
        pressure_model.fit(X_combined_scaled, y_pressure_combined)
        
        # Save the updated models and scaler
        joblib.dump({"velocity": velocity_model, "pressure": pressure_model}, "/home/musacim/simulation/openfoam/surrogate_model.joblib")
        joblib.dump(scaler, "/home/musacim/simulation/openfoam/scaler.joblib")
        
        print("Retraining completed.\n")
    else:
        print("Accuracy is acceptable. No retraining needed.\n")
