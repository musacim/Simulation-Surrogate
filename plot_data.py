# plot_data.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset
data_csv = "/home/musacim/simulation/openfoam/simulation_data.csv"

# Load the dataset
df = pd.read_csv(data_csv)

# Define regions based on 'lid_time_step'
def get_region(t):
    if 1 <= t <= 10:
        return "Region1_Training"
    elif 11 <= t <= 20:
        return "Region2_Shifting"
    elif 21 <= t <= 30:
        return "Region3_Shifting"
    else:
        return "Unknown"

df['region'] = df['lid_time_step'].apply(get_region)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# 1. Scatter Plot: Lid Velocity vs. Viscosity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='lid_velocity', y='viscosity', hue='region', palette=['blue', 'orange', 'green'])
plt.title('Lid Velocity vs. Viscosity Across Regions')
plt.xlabel('Lid Velocity (m/s)')
plt.ylabel('Viscosity (m²/s)')
plt.legend(title='Region')
plt.tight_layout()
plt.savefig('/home/musacim/simulation/openfoam/lid_velocity_vs_viscosity_regions.png')
plt.show()

# 2. Histogram: Lid Velocity
plt.figure(figsize=(10, 6))
sns.histplot(df[df['region'] == 'Region1_Training']['lid_velocity'], color='blue', label='Region1_Training', kde=True, stat="density", linewidth=0)
sns.histplot(df[df['region'] == 'Region2_Shifting']['lid_velocity'], color='orange', label='Region2_Shifting', kde=True, stat="density", linewidth=0, alpha=0.7)
sns.histplot(df[df['region'] == 'Region3_Shifting']['lid_velocity'], color='green', label='Region3_Shifting', kde=True, stat="density", linewidth=0, alpha=0.5)
plt.title('Distribution of Lid Velocity Across Regions')
plt.xlabel('Lid Velocity (m/s)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('/home/musacim/simulation/openfoam/lid_velocity_distribution_regions.png')
plt.show()

# 3. Histogram: Viscosity
plt.figure(figsize=(10, 6))
sns.histplot(df[df['region'] == 'Region1_Training']['viscosity'], color='blue', label='Region1_Training', kde=True, stat="density", linewidth=0)
sns.histplot(df[df['region'] == 'Region2_Shifting']['viscosity'], color='orange', label='Region2_Shifting', kde=True, stat="density", linewidth=0, alpha=0.7)
sns.histplot(df[df['region'] == 'Region3_Shifting']['viscosity'], color='green', label='Region3_Shifting', kde=True, stat="density", linewidth=0, alpha=0.5)
plt.title('Distribution of Viscosity Across Regions')
plt.xlabel('Viscosity (m²/s)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('/home/musacim/simulation/openfoam/viscosity_distribution_regions.png')
plt.show()

# 4. Box Plot: Lid Velocity
plt.figure(figsize=(8, 6))
sns.boxplot(x='region', y='lid_velocity', data=df, palette=['blue', 'orange', 'green'])
plt.title('Box Plot of Lid Velocity by Region')
plt.xlabel('Region')
plt.ylabel('Lid Velocity (m/s)')
plt.tight_layout()
plt.savefig('/home/musacim/simulation/openfoam/lid_velocity_boxplot_regions.png')
plt.show()

# 5. Box Plot: Viscosity
plt.figure(figsize=(8, 6))
sns.boxplot(x='region', y='viscosity', data=df, palette=['blue', 'orange', 'green'])
plt.title('Box Plot of Viscosity by Region')
plt.xlabel('Region')
plt.ylabel('Viscosity (m²/s)')
plt.tight_layout()
plt.savefig('/home/musacim/simulation/openfoam/viscosity_boxplot_regions.png')
plt.show()

# 6. Velocity Magnitude vs. Lid Velocity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='lid_velocity', y='velocity_magnitude', hue='region', palette=['blue', 'orange', 'green'])
plt.title('Velocity Magnitude vs. Lid Velocity Across Regions')
plt.xlabel('Lid Velocity (m/s)')
plt.ylabel('Velocity Magnitude')
plt.legend(title='Region')
plt.tight_layout()
plt.savefig('/home/musacim/simulation/openfoam/velocity_magnitude_vs_lid_velocity_regions.png')
plt.show()

# 7. Pressure vs. Viscosity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='viscosity', y='pressure', hue='region', palette=['blue', 'orange', 'green'])
plt.title('Pressure vs. Viscosity Across Regions')
plt.xlabel('Viscosity (m²/s)')
plt.ylabel('Pressure')
plt.legend(title='Region')
plt.tight_layout()
plt.savefig('/home/musacim/simulation/openfoam/pressure_vs_viscosity_regions.png')
plt.show()
