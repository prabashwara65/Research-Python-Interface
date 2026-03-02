# solar_predictor.py

import numpy as np
import tensorflow as tf
import joblib
import os
import pandas as pd  # Needed for DataFrame to match scaler columns

# ---------------- CONSTANTS ----------------
PANEL_AREA = 6.5          # m²
EFFICIENCY = 0.18
LUX_TO_IRRADIANCE = 126   # lux → W/m²
# -------------------------------------------

# Get folder of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build path to model folder
MODEL_PATH = os.path.join(BASE_DIR, "model", "solar_lstm.keras")
SCALER_X_PATH = os.path.join(BASE_DIR, "model", "scaler_X.save")
SCALER_Y_PATH = os.path.join(BASE_DIR, "model", "scaler_y.save")

# Load model & scalers
_model = tf.keras.models.load_model(MODEL_PATH)
_scaler_X = joblib.load(SCALER_X_PATH)
_scaler_y = joblib.load(SCALER_Y_PATH)

# Columns that scaler was originally fitted on
SCALER_COLUMNS = ["light_intensity", "temperature", "humidity"]

# ---------------- FUNCTIONS ----------------
def lux_to_irradiance(lux: float) -> float:
    """Convert LUX to solar irradiance (W/m²)."""
    return lux / LUX_TO_IRRADIANCE


def calculate_physical_energy(lux: float) -> float:
    """Physics-based instantaneous energy (kWh)."""
    irradiance = lux_to_irradiance(lux)
    return (irradiance * PANEL_AREA * EFFICIENCY) / 1000


def predict_24h_energy(sequence_24h):
    """
    Predict next 24h solar energy using LSTM.

    sequence_24h shape: (24, 3)
    columns: [irradiance, temperature, humidity]
    """
    # Convert to numpy array
    X = np.array(sequence_24h).reshape(24, 3)
    
    # Convert to DataFrame with original scaler columns to avoid feature name mismatch
    X_df = pd.DataFrame(X, columns=SCALER_COLUMNS)
    
    # Scale
    X_scaled = _scaler_X.transform(X_df).reshape(1, 24, 3)
    
    # Predict
    pred_norm = _model.predict(X_scaled, verbose=0)[0]
    
    # Inverse scale
    energy_kwh = _scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    
    # Physics constraints
    irradiance_seq = X[:, 0]
    energy_kwh[irradiance_seq == 0] = 0
    energy_kwh[energy_kwh < 0] = 0
    
    return energy_kwh


def predict_24h_from_single_reading(temp: float, hum: float, lux: float):
    """
    Predict 24-hour solar energy using a single sensor reading.

    - Fills a 24-hour buffer with the same reading.
    - Returns energy_kwh array (length 24) and total predicted energy.
    """
    irradiance = lux_to_irradiance(lux)
    sequence_24h = [[irradiance, temp, hum]] * 24

    energy_24h = predict_24h_energy(sequence_24h)
    total_energy = np.sum(energy_24h)

    return energy_24h, total_energy
