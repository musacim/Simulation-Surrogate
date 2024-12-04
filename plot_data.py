import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the path to the dataset
data_csv = "/home/musacim/simulation/openfoam/simulation_data.csv"

# Load the dataset
df = pd.read_csv(data_csv)

# Ensure 'lid_time_step' is of integer type
df['lid_time_step'] = df['lid_time_step'].astype(int)

# Define regions based on 'lid_time_step'
def get_region(t):
    if 1 <= t <= 25:
        return "Region1_Training"
    elif 26 <= t <= 40:
        return "Region2_Shifting"
    elif 41 <= t <= 55:
        return "Region3_Shifting"
    else:
        return "Unknown"

df['region'] = df['lid_time_step'].apply(get_region)

# Verify that all regions are correctly assigned
unknown_regions = df[df['region'] == 'Unknown']
if not unknown_regions.empty:
    print("Warning: Some data points have been assigned to 'Unknown' region.")
    print(unknown_regions)

# Filter out 'Unknown' regions for plotting
df_plot = df[df['region'] != 'Unknown']

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Define color palette for regions
palette_dict = {
    "Region1_Training": "blue",
    "Region2_Shifting": "orange",
    "Region3_Shifting": "green"
}

# Define the order of regions for consistent color mapping
regions_order = ["Region1_Training", "Region2_Shifting", "Region3_Shifting"]
colors = [palette_dict[region] for region in regions_order]

# Ensure the output directory exists
output_dir = "/home/musacim/simulation/openfoam/plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Scatter Plot: Lid Velocity vs. Viscosity
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_plot,
    x='lid_velocity',
    y='viscosity',
    hue='region',
    palette=palette_dict,
    s=100
)
plt.title('Lid Velocity vs. Viscosity Across Regions', fontsize=16)
plt.xlabel('Lid Velocity (m/s)', fontsize=14)
plt.ylabel('Viscosity (m²/s)', fontsize=14)
plt.legend(title='Region', fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lid_velocity_vs_viscosity_regions.png'))
plt.close()

# 2. Histogram: Lid Velocity
plt.figure(figsize=(10, 6))
for region, color in palette_dict.items():
    sns.histplot(
        data=df_plot[df_plot['region'] == region],
        x='lid_velocity',
        label=region,
        color=color,
        kde=True,
        stat="density",
        linewidth=0,
        alpha=0.6
    )
plt.title('Distribution of Lid Velocity Across Regions', fontsize=16)
plt.xlabel('Lid Velocity (m/s)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(title='Region', fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lid_velocity_distribution_regions.png'))
plt.close()

# 3. Histogram: Viscosity
plt.figure(figsize=(10, 6))
for region, color in palette_dict.items():
    sns.histplot(
        data=df_plot[df_plot['region'] == region],
        x='viscosity',
        label=region,
        color=color,
        kde=True,
        stat="density",
        linewidth=0,
        alpha=0.6
    )
plt.title('Distribution of Viscosity Across Regions', fontsize=16)
plt.xlabel('Viscosity (m²/s)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(title='Region', fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'viscosity_distribution_regions.png'))
plt.close()

# 4. Box Plot: Lid Velocity
plt.figure(figsize=(8, 6))
sns.boxplot(
    x='region',
    y='lid_velocity',
    hue='region',
    data=df_plot,
    palette=palette_dict,
    order=regions_order
)
plt.title('Box Plot of Lid Velocity by Region', fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Lid Velocity (m/s)', fontsize=14)
plt.legend([], [], frameon=False)  # Remove the legend
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lid_velocity_boxplot_regions.png'))
plt.close()

# 5. Box Plot: Viscosity
plt.figure(figsize=(8, 6))
sns.boxplot(
    x='region',
    y='viscosity',
    hue='region',
    data=df_plot,
    palette=palette_dict,
    order=regions_order
)
plt.title('Box Plot of Viscosity by Region', fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Viscosity (m²/s)', fontsize=14)
plt.legend([], [], frameon=False)  # Remove the legend
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'viscosity_boxplot_regions.png'))
plt.close()

# 6. Velocity Magnitude vs. Lid Velocity
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_plot,
    x='lid_velocity',
    y='velocity_magnitude',
    hue='region',
    palette=palette_dict,
    s=100
)
plt.title('Velocity Magnitude vs. Lid Velocity Across Regions', fontsize=16)
plt.xlabel('Lid Velocity (m/s)', fontsize=14)
plt.ylabel('Velocity Magnitude', fontsize=14)
plt.legend(title='Region', fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'velocity_magnitude_vs_lid_velocity_regions.png'))
plt.close()

# 7. Pressure vs. Viscosity
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_plot,
    x='viscosity',
    y='pressure',
    hue='region',
    palette=palette_dict,
    s=100
)
plt.title('Pressure vs. Viscosity Across Regions', fontsize=16)
plt.xlabel('Viscosity (m²/s)', fontsize=14)
plt.ylabel('Pressure', fontsize=14)
plt.legend(title='Region', fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pressure_vs_viscosity_regions.png'))
plt.close()
