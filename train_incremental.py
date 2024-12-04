import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib

# Ensure the 'plots' directory exists
plots_dir = 'plots_2'
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

# Initialize dictionaries to store accuracy metrics per region
accuracy_metrics = {
    'Region 1': {
        'time_steps': [],
        'velocity_accuracy': [],
        'pressure_accuracy': [],
        'velocity_mape': [],
        'pressure_mape': [],
        'retrain_steps': []
    },
    'Region 2': {
        'time_steps': [],
        'velocity_accuracy': [],
        'pressure_accuracy': [],
        'velocity_mape': [],
        'pressure_mape': [],
        'retrain_steps': []
    },
    'Region 3': {
        'time_steps': [],
        'velocity_accuracy': [],
        'pressure_accuracy': [],
        'velocity_mape': [],
        'pressure_mape': [],
        'retrain_steps': []
    }
}

# Loop over regions
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
        velocity_within_tolerance = (np.abs(y_velocity_test - y_velocity_pred) / y_velocity_test <= 0.1) if y_velocity_test != 0 else False
        pressure_within_tolerance = (np.abs(y_pressure_test - y_pressure_pred) / y_pressure_test <= 0.1) if y_pressure_test != 0 else False

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

        # Record metrics
        accuracy_metrics[region_key]['time_steps'].append(current_time_step)
        accuracy_metrics[region_key]['velocity_accuracy'].append(cumulative_velocity_accuracy)
        accuracy_metrics[region_key]['pressure_accuracy'].append(cumulative_pressure_accuracy)
        accuracy_metrics[region_key]['velocity_mape'].append(velocity_mape)
        accuracy_metrics[region_key]['pressure_mape'].append(pressure_mape)

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
            overall_velocity_correct = np.sum((np.abs(y_velocity_eval.values - y_velocity_eval_pred) / y_velocity_eval.values) <= 0.1)
            overall_pressure_correct = np.sum((np.abs(y_pressure_eval.values - y_pressure_eval_pred) / y_pressure_eval.values) <= 0.1)
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

                # Record retraining step
                accuracy_metrics[region_key]['retrain_steps'].append(current_time_step)

                # Reset cumulative counters after retraining
                velocity_correct = 0
                pressure_correct = 0
                total_velocity = 0
                total_pressure = 0
            else:
                print(f"Model performance is satisfactory. No retraining needed.\n")

    print(f"Completed processing for all regions.\n")

    # Function to plot accuracy metrics and save the plots
    def plot_accuracy_metrics(accuracy_metrics, plots_dir):
        """
        Plot cumulative accuracy over time for velocity and pressure models across regions
        and save the plots to the specified directory.
        """
        regions = accuracy_metrics.keys()

        # Create a figure with subplots for velocity and pressure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Define colors and markers for different regions
        colors = ['blue', 'green', 'red']
        markers = ['o', 's', '^']

        for idx, region in enumerate(regions):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # Plot Velocity Accuracy
            axes[0].plot(
                accuracy_metrics[region]['time_steps'],
                accuracy_metrics[region]['velocity_accuracy'],
                label=region,
                color=color,
                marker=marker
            )

            # Plot Pressure Accuracy
            axes[1].plot(
                accuracy_metrics[region]['time_steps'],
                accuracy_metrics[region]['pressure_accuracy'],
                label=region,
                color=color,
                marker=marker
            )

            # Plot Retraining Points
            for retrain_step in accuracy_metrics[region]['retrain_steps']:
                axes[0].axvline(x=retrain_step, color=color, linestyle='--', alpha=0.7, label=f'{region} Retrain' if idx == 0 else "")
                axes[1].axvline(x=retrain_step, color=color, linestyle='--', alpha=0.7, label=f'{region} Retrain' if idx == 0 else "")

        # Configure Velocity Accuracy Plot
        axes[0].set_title('Cumulative Velocity Model Accuracy Over Time')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_ylim(0, 100)  # Since accuracy is in percentage

        # Configure Pressure Accuracy Plot
        axes[1].set_title('Cumulative Pressure Model Accuracy Over Time')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_ylim(0, 100)  # Since accuracy is in percentage

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plot_path = os.path.join(plots_dir, 'cumulative_accuracy_over_time.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Close the plot to free memory
        plt.close()

    # Function to plot accuracy and MAPE metrics and save the plots
    def plot_accuracy_and_mape(accuracy_metrics, plots_dir):
        """
        Plot cumulative accuracy and MAPE over time for velocity and pressure models across regions
        and save the plots to the specified directory.
        """
        regions = accuracy_metrics.keys()

        # Create a figure with four subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

        # Define colors and markers for different regions
        colors = ['blue', 'green', 'red']
        markers = ['o', 's', '^']

        for idx, region in enumerate(regions):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # Plot Velocity Accuracy
            axes[0].plot(
                accuracy_metrics[region]['time_steps'],
                accuracy_metrics[region]['velocity_accuracy'],
                label=region,
                color=color,
                marker=marker
            )

            # Plot Pressure Accuracy
            axes[1].plot(
                accuracy_metrics[region]['time_steps'],
                accuracy_metrics[region]['pressure_accuracy'],
                label=region,
                color=color,
                marker=marker
            )

            # Plot Velocity MAPE
            axes[2].plot(
                accuracy_metrics[region]['time_steps'],
                accuracy_metrics[region]['velocity_mape'],
                label=region,
                color=color,
                marker=marker
            )

            # Plot Pressure MAPE
            axes[3].plot(
                accuracy_metrics[region]['time_steps'],
                accuracy_metrics[region]['pressure_mape'],
                label=region,
                color=color,
                marker=marker
            )

            # Plot Retraining Points for Velocity and Pressure Accuracy
            for retrain_step in accuracy_metrics[region]['retrain_steps']:
                axes[0].axvline(x=retrain_step, color=color, linestyle='--', alpha=0.7, label=f'{region} Retrain' if idx == 0 else "")
                axes[1].axvline(x=retrain_step, color=color, linestyle='--', alpha=0.7, label=f'{region} Retrain' if idx == 0 else "")
                axes[2].axvline(x=retrain_step, color=color, linestyle='--', alpha=0.7, label=f'{region} Retrain' if idx == 0 else "")
                axes[3].axvline(x=retrain_step, color=color, linestyle='--', alpha=0.7, label=f'{region} Retrain' if idx == 0 else "")

        # Configure Velocity Accuracy Plot
        axes[0].set_title('Cumulative Velocity Model Accuracy Over Time')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_ylim(0, 100)

        # Configure Pressure Accuracy Plot
        axes[1].set_title('Cumulative Pressure Model Accuracy Over Time')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_ylim(0, 100)

        # Configure Velocity MAPE Plot
        axes[2].set_title('Cumulative Velocity Model MAPE Over Time')
        axes[2].set_ylabel('MAPE')
        axes[2].legend()
        axes[2].grid(True)

        # Configure Pressure MAPE Plot
        axes[3].set_title('Cumulative Pressure Model MAPE Over Time')
        axes[3].set_xlabel('Time Steps')
        axes[3].set_ylabel('MAPE')
        axes[3].legend()
        axes[3].grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plot_path = os.path.join(plots_dir, 'cumulative_accuracy_and_mape_over_time.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Close the plot to free memory
        plt.close()

    # Plot the cumulative accuracy and save the plot
    plot_accuracy_metrics(accuracy_metrics, plots_dir)

    # Optionally, plot both accuracy and MAPE and save the plot
    plot_accuracy_and_mape(accuracy_metrics, plots_dir)
