"""
AgriSense-MM: System Constants and Domain Definitions.
Defines stress classes, sensor features, field dimensions, and domain mappings.
"""

from typing import Dict, List, Tuple

# 6 Primary Crop Stress Classes
STRESS_CLASSES: List[str] = [
    "Healthy",
    "Water Stress",
    "Nutrient Deficiency",
    "Heat Stress",
    "Disease Stress",
    "Pest Damage",
]

NUM_CLASSES: int = len(STRESS_CLASSES)

STRESS_CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(STRESS_CLASSES)}
IDX_TO_STRESS_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(STRESS_CLASSES)}

# Color coding for visualization and dashboard alerts (Hex)
STRESS_COLORS: Dict[int, str] = {
    0: "#2ecc71",  # Healthy: Bright Emerald Green
    1: "#3498db",  # Water Stress: Ocean Blue
    2: "#f39c12",  # Nutrient Deficiency: Amber Orange
    3: "#e74c3c",  # Heat Stress: Crimson Red
    4: "#9b59b6",  # Disease Stress: Purple
    5: "#d35400",  # Pest Damage: Dark Rust
}

# Domain recommended agronomic actions per stress class
RECOMMENDED_ACTIONS: Dict[int, str] = {
    0: "Field condition nominal. Continue standard monitoring schedule.",
    1: "Water stress detected. Trigger variable-rate precision irrigation for zone.",
    2: "Nutrient deficiency suspected. Recommend soil nitrogen/phosphorus/potassium foliar spray.",
    3: "Extreme heat stress. Deploy shade netting or canopy cooling misting cycle.",
    4: "Pathogen infestation detected. Quarantine zone and apply targeted bio-fungicide.",
    5: "Pest damage observed. Deploy targeted pheromone traps and beneficial predator biocontrol.",
}

# Mobile Platform Motion States (IMU classification)
PLATFORM_STATES: List[str] = [
    "cruising",
    "turning",
    "hovering",
    "high_vibration",
]
NUM_PLATFORM_STATES: int = len(PLATFORM_STATES)
PLATFORM_STATE_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(PLATFORM_STATES)}

# Environmental Time-Series Features (14-day sequence)
ENV_FEATURE_NAMES: List[str] = [
    "temperature_c",          # Ambient temperature in Celsius
    "humidity_pct",           # Relative humidity in %
    "soil_moisture_pct",      # Volumetric soil moisture in %
    "rainfall_mm",            # Daily precipitation in mm
    "solar_radiation_wm2",    # Solar irradiance in W/m^2
    "wind_speed_ms",          # Wind velocity in m/s
    "ndvi",                   # Normalized Difference Vegetation Index [-0.1 to 1.0]
]
NUM_ENV_FEATURES: int = len(ENV_FEATURE_NAMES)
ENV_WINDOW_DAYS: int = 14

# Motion / IMU Features (100 Hz, 1-second window = 100 samples)
IMU_FEATURE_NAMES: List[str] = [
    "accel_x",  # m/s^2
    "accel_y",  # m/s^2
    "accel_z",  # m/s^2
    "gyro_x",   # rad/s
    "gyro_y",   # rad/s
    "gyro_z",   # rad/s
]
NUM_IMU_CHANNELS: int = len(IMU_FEATURE_NAMES)
IMU_SAMPLE_POINTS: int = 100  # 1 second at 100 Hz

# Static Geographic and Edaphic (Soil) Features
STATIC_FEATURE_NAMES: List[str] = [
    "latitude",            # Deg N
    "longitude",           # Deg E
    "elevation_m",         # Altitude in meters
    "soil_ph",             # Soil pH [4.5 - 8.5]
    "organic_carbon_gkg",  # Soil Organic Carbon in g/kg
]
NUM_STATIC_FEATURES: int = len(STATIC_FEATURE_NAMES)

# Field Grid Zone Spec (Default 4x4 Grid = 16 Inspection Zones)
FIELD_GRID_ROWS: int = 4
FIELD_GRID_COLS: int = 4
TOTAL_ZONES: int = FIELD_GRID_ROWS * FIELD_GRID_COLS

# Default Field Center (Realistic test farm coordinates in agricultural belt)
DEFAULT_FARM_ORIGIN: Tuple[float, float] = (28.6139, 77.2090)  # Latitude, Longitude
ZONE_SIZE_METERS: float = 25.0  # 25m x 25m zones

