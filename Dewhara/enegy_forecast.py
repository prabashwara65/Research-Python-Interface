# -------------------------------
# Energy Forecast: Battery SOC, SoH, and Runtime Prediction for LED Load
# Wrapped in run_forecast() for menu integration
# -------------------------------

import os
import serial
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

def run_forecast():
    # -------------------------------
    # Step 1: Model & Scaler Paths
    # -------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "battery_soc_led_model")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_led.pkl")

    print("📦 Loading LSTM model and scaler...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded.\n")

    # -------------------------------
    # Step 2: Constants
    # -------------------------------
    BATTERY_CAPACITY_AH = 1.8  # 1800 mAh
    READINGS = 30
    ARDUINO_PORT = 'COM9'
    BAUD_RATE = 9600

    # LED current mapping (A)
    LED_CURRENT_MAP = {0: 0.0, 1: 0.02, 2: 0.04}

    # -------------------------------
    # Step 3: Read Battery Voltage and LED Status from Arduino
    # -------------------------------
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"📡 Reading battery data from Arduino ({READINGS} readings)...")
    except Exception as e:
        print(f"❌ Could not connect to Arduino: {e}")
        return

    voltages = []
    time_values = []
    current_load = []
    led_counts = []

    count = 0

    try:
        while count < READINGS:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    parts = line.split(',')

                    # Skip lines that don't have at least voltage + 2 LEDs
                    if len(parts) < 3:
                        print(f"⚠️ Invalid Arduino line: '{line}'")
                        continue

                    # Battery voltage: 3rd-to-last column (robust for extra sensors)
                    voltage = float(parts[-3])
                    voltages.append(voltage)
                    time_values.append(count)

                    # LED states: last 2 columns
                    led_count = int(parts[-2]) + int(parts[-1])
                    if led_count > 2:
                        led_count = 2
                    led_counts.append(led_count)

                    current = LED_CURRENT_MAP[led_count]
                    current_load.append(current)

                    count += 1
                    print(f"Reading {count:02d}/{READINGS}: Voltage = {voltage:.2f} V, LED count = {led_count}, Current = {current:.2f} A")

                except (ValueError, IndexError):
                    print(f"⚠️ Invalid Arduino line: '{line}'")
            else:
                time.sleep(0.05)
    finally:
        arduino.close()
        print("\n✅ Arduino connection closed.\n")

    voltages = np.array(voltages)
    time_values = np.array(time_values)
    current_load = np.array(current_load)
    led_counts = np.array(led_counts)

    # -------------------------------
    # Step 4: Prepare Features for LSTM
    # -------------------------------
    features = np.column_stack([voltages, current_load, time_values])
    features_scaled = scaler.transform(features)

    def create_sequences(X, time_steps=10):
        seq = []
        for i in range(len(X) - time_steps):
            seq.append(X[i:i + time_steps])
        return np.array(seq)

    TIME_STEPS = 10
    X_seq = create_sequences(features_scaled, TIME_STEPS)

    # -------------------------------
    # Step 5: Predict SOC using LSTM
    # -------------------------------
    soc_pred_raw = model.predict(X_seq, verbose=0).flatten()

    # -------------------------------
    # Step 6: Compute SoH and Runtime dynamically using LED count
    # -------------------------------
    remaining_capacity_ah = (soc_pred_raw[0] / 100) * BATTERY_CAPACITY_AH
    initial_capacity = remaining_capacity_ah

    soc_adjusted = []
    soh = []
    runtime = []

    for i in range(len(soc_pred_raw) - 1):
        delta_t = time_values[min(i + 1, READINGS - 1)] - time_values[i]
        delta_t = max(delta_t, 1)

        # Use LED-based current for this time step
        current = current_load[i]
        discharge_ah = current * (delta_t / 3600)
        remaining_capacity_ah -= discharge_ah
        remaining_capacity_ah = max(remaining_capacity_ah, 0)

        soc_val = (remaining_capacity_ah / BATTERY_CAPACITY_AH) * 100
        soc_adjusted.append(soc_val)

        soh_val = (remaining_capacity_ah / initial_capacity) * 100
        soh.append(soh_val)

        runtime_sec = (remaining_capacity_ah / current) * 3600 if current > 0 else 0
        runtime.append(runtime_sec)

        if remaining_capacity_ah <= 0:
            break

    soc_adjusted = np.array(soc_adjusted)
    soh = np.array(soh)
    runtime = np.array(runtime)

    # -------------------------------
    # Step 7: Display Results (with LED count)
    # -------------------------------
    last_led_count = led_counts[-1] if len(led_counts) > 0 else 0

    print("\n========== BATTERY STATUS ==========")
    print(f"LEDs ON (last reading) : {last_led_count}")
    print(f"Predicted SOC           : {soc_adjusted[-1]:.2f} %")
    print(f"Estimated SoH           : {soh[-1]:.2f} %")
    print(f"Estimated Runtime       : {runtime[-1] / 3600:.2f} hours")
    print("===================================\n")

    # -------------------------------
    # Step 8: Plot Results
    # -------------------------------
    plt.figure(figsize=(12,5))
    plt.plot(soc_adjusted, label="SOC (%)")
    plt.title("SOC Prediction")
    plt.xlabel("Time")
    plt.ylabel("SOC (%)")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(soh, label="SoH (%)", color="orange")
    plt.title("SoH Estimation")
    plt.xlabel("Time")
    plt.ylabel("SoH (%)")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(runtime, label="Runtime (seconds)", color="green")
    plt.title("Remaining Battery Runtime")
    plt.xlabel("Time")
    plt.ylabel("Seconds")
    plt.grid()
    plt.legend()
    plt.show()


# -------------------------------
# Optional: run directly
# -------------------------------
if __name__ == "__main__":
    run_forecast()
