import numpy as np
import tensorflow as tf
import datetime as dt
import os

# ------------------------------- Load trained model -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sync_scheduler_model.keras")

print("📦 Loading sync scheduler model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------- Constants -------------------------------
FEATURES = [
    "battery_level",
    "signal_strength",
    "cloud_cover",
    "temp",
    "solar_irradiance",
    "forecast_sunlight_hours",
    "app_open_count_last_hour",
    "panel_current",
    "battery_efficiency",
    "hour_of_day",
    "humidity"
]

IMMEDIATE_THRESHOLD = 0.8
SCHEDULED_THRESHOLD = 0.3
MAX_SYNC_INTERVAL = 20
EPS = 1e-6

NIGHT_START = 18
NIGHT_END = 6
CRITICAL_BATTERY = 0.35

PANEL_VOLTAGE_NOMINAL = 5.0      # Volts
PANEL_CURRENT_RATED = 0.1 * 2    # 2 panels in parallel

# ------------------------------- Utility Functions -------------------------------
def is_night(hour):
    return hour >= NIGHT_START or hour <= NIGHT_END

def calculate_forecast_sunlight_hours(hour_of_day, solar_irradiance):
    DAYLIGHT_START = 6
    DAYLIGHT_END = 18
    MIN_DAYLIGHT_WM2 = 50

    if solar_irradiance < MIN_DAYLIGHT_WM2 or hour_of_day >= DAYLIGHT_END:
        return 0.0

    remaining_hours = DAYLIGHT_END - hour_of_day
    sunlight_factor = min(solar_irradiance / 1000.0, 1.0)
    return float(max(remaining_hours * sunlight_factor, 0))

# ------------------------------- Input Functions -------------------------------
def get_manual_input():
    """Fallback manual input if Arduino data not provided."""
    print("\nEnter sensor values manually:")

    battery_level_now = float(input("battery_level (0-1): "))
    battery_level_prev = float(input("battery_level_prev (0-1): "))
    panel_voltage = float(input("panel_voltage: "))
    panel_current = np.clip((panel_voltage / PANEL_VOLTAGE_NOMINAL) * PANEL_CURRENT_RATED, 0, PANEL_CURRENT_RATED)

    soc_delta = battery_level_now - battery_level_prev
    panel_power_w = panel_voltage * panel_current
    battery_efficiency = np.clip(soc_delta * 100 / (panel_power_w + EPS), 0, 0.95)

    signal_strength = float(input("signal_strength (-60 to -120 dBm): "))
    cloud_cover = float(input("cloud_cover (0-1): "))
    temp = float(input("temp (°C): "))
    solar_irradiance = float(input("solar_irradiance (W/m²): "))
    app_open_count = float(input("app_open_count_last_hour: "))
    humidity = float(input("humidity (%): "))

    hour_of_day = dt.datetime.now().hour
    forecast_sunlight_hours = calculate_forecast_sunlight_hours(hour_of_day, solar_irradiance)

    values = [
        battery_level_now,
        signal_strength,
        cloud_cover,
        temp,
        solar_irradiance,
        forecast_sunlight_hours,
        app_open_count,
        panel_current,
        battery_efficiency,
        hour_of_day,
        humidity
    ]

    return np.array([values], dtype=np.float32), battery_level_now, hour_of_day

# ------------------------------- Sync Decision -------------------------------
def run_sync_decision(battery_level, panel_voltage, temp, humidity, solar_irradiance,
                      signal_strength=-80, cloud_cover=0.2, app_open_count=0, current_load=None):
    """
    Run sync decision automatically from Arduino readings.

    'current_load' is optional for future-proofing.
    """
    hour_of_day = dt.datetime.now().hour

    # Panel current calculation
    if current_load is None:
        panel_current = np.clip((panel_voltage / PANEL_VOLTAGE_NOMINAL) * PANEL_CURRENT_RATED, 0, PANEL_CURRENT_RATED)
    else:
        panel_current = current_load

    battery_efficiency = 0.5  # default if no previous SOC info
    forecast_sunlight_hours = calculate_forecast_sunlight_hours(hour_of_day, solar_irradiance)

    values = [
        battery_level,
        signal_strength,
        cloud_cover,
        temp,
        solar_irradiance,
        forecast_sunlight_hours,
        app_open_count,
        panel_current,
        battery_efficiency,
        hour_of_day,
        humidity
    ]

    X = np.array([values], dtype=np.float32)
    decision, final_time, prob, raw_time = predict_and_decide(X, battery_level, hour_of_day)

    print("\n=== SYNC DECISION ===")
    print(f"Immediate sync probability: {prob:.2f}")
    if final_time == 0:
        print(decision)
    elif final_time is not None:
        print(decision)
        print(f"Scheduled sync time: {final_time:.1f} minutes")
    else:
        print(decision)

    return decision, final_time, prob, raw_time

# ------------------------------- Prediction Logic -------------------------------
def predict_and_decide(X, battery_level, hour_of_day):
    prediction = model.predict(X, verbose=0)

    # Multi-output handling
    if isinstance(prediction, dict):
        raw_sync_time = float(prediction["reg"][0])
        immediate_prob = float(prediction["cls"][0])
    else:
        raw_sync_time = float(prediction.squeeze())
        immediate_prob = 0.5

    raw_sync_time = np.clip(raw_sync_time, 0, MAX_SYNC_INTERVAL)

    if is_night(hour_of_day):
        if battery_level > CRITICAL_BATTERY:
            return "Night mode - defer sync", None, immediate_prob, raw_sync_time
        else:
            return "Emergency night sync (critical battery)", 0, immediate_prob, raw_sync_time

    if immediate_prob >= IMMEDIATE_THRESHOLD:
        return "Sync now sensor data", 0, immediate_prob, raw_sync_time
    elif immediate_prob >= SCHEDULED_THRESHOLD:
        return "Sync data at next sync time", raw_sync_time, immediate_prob, raw_sync_time
    else:
        return "Defer sync - do not sync now", None, immediate_prob, raw_sync_time

# ------------------------------- Main -------------------------------
if __name__ == "__main__":
    print("⚡ Running Sync Scheduler")
    choice = input("Use Arduino readings? (y/n): ").strip().lower()
    if choice == 'y':
        print("❌ Please run this from main_monitor.py with Arduino option.")
    else:
        X_manual, battery_level, hour_of_day = get_manual_input()
        run_sync_decision(
            battery_level=battery_level,
            panel_voltage=float(input("panel_voltage (V): ")),
            temp=float(input("temp (°C): ")),
            humidity=float(input("humidity (%): ")),
            solar_irradiance=float(input("solar_irradiance (W/m²): "))
        )
