# train_incremental.py

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

# Define regions
def get_region(t):
    if 1 <= t <= 10:
        return "Region1_Training"
    elif 11 <= t <= 20:
        return "Region2_Shifting"
    elif 21 <= t <= 30:
        return "Region3_Shifting"
    else:
        return "Unknown"

data['region'] = data['lid_time_step'].apply(get_region)

# Initialize models and scaler
velocity_model = RandomForestRegressor(random_state=42)
pressure_model = RandomForestRegressor(random_state=42)
scaler = StandardScaler()

# Define accuracy threshold
accuracy_threshold = 95.0  # 95%

# Initialize training flag
initial_training_done = False

# Loop over each time step
for t in sorted(data['lid_time_step'].unique()):
    current_data = data[data['lid_time_step'] == t]
    
    # Prepare features and targets
    X_current = current_data[['lid_velocity', 'viscosity', 'lid_time_step']]
    y_velocity_current = current_data['velocity_magnitude']
    y_pressure_current = current_data['pressure']
    
    if not initial_training_done:
        if t <= 5:
            # Initial Training on first 5 time steps
            training_data = data[data['lid_time_step'] <= 5]
            X_train = training_data[['lid_velocity', 'viscosity', 'lid_time_step']]
            y_velocity_train = training_data['velocity_magnitude']
            y_pressure_train = training_data['pressure']
            
            # Fit scaler
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            
            # Train models
            velocity_model.fit(X_train_scaled, y_velocity_train)
            pressure_model.fit(X_train_scaled, y_pressure_train)
            
            # Save models and scaler
            joblib.dump({"velocity": velocity_model, "pressure": pressure_model}, "/home/musacim/simulation/openfoam/surrogate_model.joblib")
            joblib.dump(scaler, "/home/musacim/simulation/openfoam/scaler.joblib")
            
            # Predict on current data
            X_current_scaled = scaler.transform(X_current)
            y_velocity_pred = velocity_model.predict(X_current_scaled)
            y_pressure_pred = pressure_model.predict(X_current_scaled)
            
            # Evaluate
            velocity_r2 = r2_score(y_velocity_current, y_velocity_pred)
            pressure_r2 = r2_score(y_pressure_current, y_pressure_pred)
            
            velocity_accuracy = velocity_r2 * 100
            pressure_accuracy = pressure_r2 * 100
            
            print(f"Initial Training Up to Time Step: {t}")
            print(f"Velocity Model - Accuracy: {velocity_accuracy:.2f}%")
            print(f"Pressure Model - Accuracy: {pressure_accuracy:.2f}%\n")
            
            # Set flag
            initial_training_done = True
            continue
        else:
            # Proceed to training if t > 5 without initial training (safety)
            pass
    
    # Predict using existing models
    X_current_scaled = scaler.transform(X_current)
    y_velocity_pred = velocity_model.predict(X_current_scaled)
    y_pressure_pred = pressure_model.predict(X_current_scaled)
    
    # Evaluate
    velocity_r2 = r2_score(y_velocity_current, y_velocity_pred)
    pressure_r2 = r2_score(y_pressure_current, y_pressure_pred)
    
    velocity_accuracy = velocity_r2 * 100
    pressure_accuracy = pressure_r2 * 100
    
    print(f"Time Step: {t}")
    print(f"Velocity Model - Accuracy: {velocity_accuracy:.2f}%")
    print(f"Pressure Model - Accuracy: {pressure_accuracy:.2f}%")
    
    # Check if accuracy drops below threshold
    if velocity_accuracy < accuracy_threshold or pressure_accuracy < accuracy_threshold:
        print("Accuracy below threshold. Retraining the model with new data...\n")
        # Retrain with all data up to current time step
        training_data = data[data['lid_time_step'] <= t]
        X_train = training_data[['lid_velocity', 'viscosity', 'lid_time_step']]
        y_velocity_train = training_data['velocity_magnitude']
        y_pressure_train = training_data['pressure']
        
        # Re-fit scaler
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        
        # Retrain models
        velocity_model.fit(X_train_scaled, y_velocity_train)
        pressure_model.fit(X_train_scaled, y_pressure_train)
        
        # Save updated models and scaler
        joblib.dump({"velocity": velocity_model, "pressure": pressure_model}, "/home/musacim/simulation/openfoam/surrogate_model.joblib")
        joblib.dump(scaler, "/home/musacim/simulation/openfoam/scaler.joblib")
        
        print("Retraining completed.\n")
    else:
        print("Accuracy is acceptable. No retraining needed.\n")
