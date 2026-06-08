import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BUOY_CONFIG = {
    'update_interval': 60,
    'max_buoys': 50,
    'default_buoys': [
        {'id': '46001', 'name': 'Eastern Pacific', 'lat': 45.0, 'lon': -130.0},
        {'id': '46002', 'name': 'Western Pacific', 'lat': 35.0, 'lon': -150.0},
        {'id': '41001', 'name': 'Atlantic North', 'lat': 35.0, 'lon': -70.0},
        {'id': '42001', 'name': 'Gulf of Mexico', 'lat': 25.0, 'lon': -90.0},
        {'id': '44004', 'name': 'Atlantic Mid', 'lat': 40.0, 'lon': -65.0},
        {'id': '44007', 'name': 'Coastal Maine', 'lat': 43.0, 'lon': -69.0},
        {'id': '46006', 'name': 'Pacific Northwest', 'lat': 48.0, 'lon': -125.0},
        {'id': '46012', 'name': 'California Coast', 'lat': 37.0, 'lon': -123.0},
        {'id': '41046', 'name': 'Caribbean Sea', 'lat': 18.0, 'lon': -80.0},
        {'id': '42039', 'name': 'Florida Keys', 'lat': 24.5, 'lon': -81.0},
    ]
}

QUALITY_CONTROL = {
    'temperature_range': (-2.0, 40.0),
    'salinity_range': (28.0, 40.0),
    'pressure_range': (0.0, 10000.0),
    'current_speed_max': 3.0,
    'max_consecutive_outliers': 3,
    'iqr_factor': 1.5,
}

ALERT_CONFIG = {
    'alert_cooldown': 300,
    'enable_email': False,
    'enable_console': True,
    'alert_file': os.path.join(LOG_DIR, 'alerts.log'),
}

VISUALIZATION_CONFIG = {
    'default_template': 'plotly_dark',
    'colorscale_temp': 'thermal',
    'colorscale_salinity': 'viridis',
    'colorscale_current': 'speed',
}
