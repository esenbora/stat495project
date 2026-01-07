"""
Configuration file for STAT495 Earthquake Analysis Project
Contains paths, constants, and color schemes.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_PATH, "data/raw")
DATA_PROCESSED = os.path.join(BASE_PATH, "data/processed")
NOTEBOOKS_PATH = os.path.join(BASE_PATH, "notebooks")
FIGURES_PATH = os.path.join(BASE_PATH, "reports/figures")
TABLES_PATH = os.path.join(BASE_PATH, "reports/tables")

# Data file paths
EARTHQUAKE_DATA = os.path.join(DATA_RAW, "afad_full_historical_1990_2025.csv")
SOIL_PROVINCE_DATA = os.path.join(DATA_RAW, "soil/turkey_soil_classification_81provinces.csv")
SOIL_DISTRICT_DATA = os.path.join(DATA_RAW, "soil/turkey_soil_classification_973districts.csv")
FAULT_DATA = os.path.join(DATA_RAW, "tectonic/turkey_active_faults.csv")
GPS_DATA = os.path.join(DATA_RAW, "tectonic/turkey_gps_velocities.csv")
MOON_DATA = os.path.join(DATA_RAW, "lunar/moon_daily_data_1990_2025.csv")
MOON_PHASES_DATA = os.path.join(DATA_RAW, "lunar/moon_phases_1990_2025.csv")
PRESSURE_DATA = os.path.join(DATA_RAW, "tectonic/turkey_atmospheric_pressure_1990_2025.csv")
POPULATION_DATA = os.path.join(DATA_RAW, "population/turkey_population_density.xlsx")

# =============================================================================
# CONSTANTS
# =============================================================================
MAGNITUDE_THRESHOLD = 4.0  # Earthquake (>=4.0) vs Tremor (<4.0)
EARTH_RADIUS_KM = 6371.0

# Turkey bounding box
TURKEY_BOUNDS = {
    'lat_min': 35.5,
    'lat_max': 42.5,
    'lon_min': 25.5,
    'lon_max': 45.0
}

# =============================================================================
# COLOR SCHEMES
# =============================================================================
COLORS = {
    'earthquake': '#e74c3c',
    'tremor': '#3498db',
    'primary': '#2c3e50',
    'secondary': '#7f8c8d',
    'accent': '#f39c12',
    'success': '#27ae60',
    'warning': '#f1c40f',
    'danger': '#c0392b'
}

SOIL_COLORS = {
    'ZA': '#27ae60',  # Hard rock - green
    'ZB': '#3498db',  # Soft rock - blue
    'ZC': '#f39c12',  # Dense soil - orange
    'ZD': '#e74c3c',  # Soft soil - red
    'ZE': '#9b59b6'   # Very soft soil - purple
}

HAZARD_COLORS = {
    'Very High': '#c0392b',
    'High': '#e74c3c',
    'Medium': '#f39c12',
    'Low': '#3498db',
    'Very Low': '#27ae60'
}

MOON_PHASE_COLORS = {
    'New Moon': '#2c3e50',
    'Waxing': '#3498db',
    'Full Moon': '#f1c40f',
    'Waning': '#e67e22'
}

CLUSTER_CMAP = 'tab10'
DENSITY_CMAP = 'YlOrRd'
DEPTH_CMAP = 'viridis'

# =============================================================================
# VISUALIZATION DEFAULTS
# =============================================================================
FIGURE_DPI = 150
FIGURE_SIZE_DEFAULT = (12, 8)
FIGURE_SIZE_LARGE = (14, 10)
FIGURE_SIZE_WIDE = (16, 6)
FONT_SIZE = 11
TITLE_SIZE = 14
LABEL_SIZE = 12
