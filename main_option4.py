import serial
import time
import os
import sys
import numpy as np
from Sithmi.solar_predictor import (
    lux_to_irradiance,
    calculate_physical_energy,
    predict_24h_from_single_reading
)

# ---------------- SETTINGS ----------------
ARDUINO_PORT = 'COM9'
BAUD_RATE = 9600
READINGS_PER_MIN = 60
HALF_MIN = READINGS_PER_MIN // 2
SYNC_READINGS = 30  # Number of readings for sync scheduler
# ------------------------------------------

# Add Hasara folder to path for sync_scheduler import
HASARA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hasara")
if HASARA_PATH not in sys.path:
    sys.path.append(HASARA_PATH)

import sync_scheduler  # Your modified sync_scheduler.py

# ---------------- Import Prabashwara/imageObjectDetection ----------------
PRABASHWARA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Prabashwara")
if PRABASHWARA_PATH not in sys.path:
    sys.path.append(PRABASHWARA_PATH)

try:
    from image_object_detection import run_light_detection
    print("✅ Imported run_light_detection successfully")
except ModuleNotFoundError:
    print("❌ Could not find image_object_detection.py in Prabashwara folder")
    run_light_detection = None

# ---------------- ARDUINO FUNCTIONS ----------------
def read_arduino_data():
    """Read 1-minute data from Arduino and return averaged temp, hum, lux, solar, battery."""
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
    time.sleep(2)
    print("✅ Arduino connected, reading sensors...")

    sensor_data_list = []

    try:
        count = 0
        while count < READINGS_PER_MIN:
            if arduino.in_waiting:
                line = arduino.readline().decode('utf-8').strip()
                try:
                    temp, hum, lux, solar, battery = map(float, line.split(','))
                except:
                    continue

                sensor_data_list.append((temp, hum, lux, solar, battery))
                count += 1

                if count <= HALF_MIN:
                    print(
                        f"{count:02d}: "
                        f"Temp={temp:.1f}C | "
                        f"Hum={hum:.1f}% | "
                        f"Lux={lux:.1f} | "
                        f"Solar={solar:.2f}V | "
                        f"Battery={battery:.2f}V"
                    )

        # Compute 1-minute average
        avg_temp = sum(d[0] for d in sensor_data_list) / READINGS_PER_MIN
        avg_hum = sum(d[1] for d in sensor_data_list) / READINGS_PER_MIN
        avg_lux = sum(d[2] for d in sensor_data_list) / READINGS_PER_MIN
        avg_solar = sum(d[3] for d in sensor_data_list) / READINGS_PER_MIN
        avg_battery = sum(d[4] for d in sensor_data_list) / READINGS_PER_MIN

        return avg_temp, avg_hum, avg_lux, avg_solar, avg_battery

    finally:
        arduino.close()
        print("Arduino connection closed.")


def read_arduino_n_readings(n):
    """Read n readings from Arduino, return lists of temp, hum, lux, solar, battery"""
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
    time.sleep(0.2)
    temps, hums, luxes, solars, batteries = [], [], [], [], []
    try:
        count = 0
        while count < n:
            if arduino.in_waiting:
                line = arduino.readline().decode('utf-8').strip()
                try:
                    temp, hum, lux, solar, battery = map(float, line.split(','))
                except:
                    continue
                temps.append(temp)
                hums.append(hum)
                luxes.append(lux)
                solars.append(solar)
                batteries.append(battery)
                count += 1
                print(f"Reading {count}/{n}: Temp={temp:.1f}, Hum={hum:.1f}, Lux={lux:.1f}, Solar={solar:.2f}, Battery={battery:.2f}")
    finally:
        arduino.close()
        print("Arduino connection closed after readings.")

    return temps, hums, luxes, solars, batteries

# ---------------- MAIN MENU ----------------
def main_menu():
    while True:
        print("\n====== SOLAR ENERGY MONITOR ======")
        print("1. Read 1-minute sensor data & show physics energy")
        print("2. Predict 24-hour solar energy (LSTM)")
        print("3. Trigger Sync Scheduler (30 readings aggregated)")
        print("4. Run Light Detection (CNN) on test images")
        print("5. Exit")
        choice = input("Select an option (1-5): ")

        if choice == '1':
            avg_temp, avg_hum, avg_lux, avg_solar, avg_battery = read_arduino_data()
            avg_irradiance = lux_to_irradiance(avg_lux)
            instant_energy = calculate_physical_energy(avg_lux)

            print("\n--- 1 MINUTE AVERAGE ---")
            print(
                f"Temp: {avg_temp:.1f}C | "
                f"Humidity: {avg_hum:.1f}% | "
                f"Lux: {avg_lux:.1f} | "
                f"Irradiance: {avg_irradiance:.2f} W/m² | "
                f"Solar: {avg_solar:.2f}V | "
                f"Battery: {avg_battery:.2f}V"
            )
            print(f"⚡ Instant Energy (physics): {instant_energy:.4f} kWh")

        elif choice == '2':
            avg_temp, avg_hum, avg_lux, avg_solar, avg_battery = read_arduino_data()
            energy_24h, total_energy = predict_24h_from_single_reading(avg_temp, avg_hum, avg_lux)

            print("\n--- 24-HOUR PREDICTION ---")
            print("Hour | Predicted Energy (kWh)")
            print("-----------------------------")
            for i, e in enumerate(energy_24h, 1):
                print(f"{i:02d}   | {e:.3f}")

            print(f"\n🔋 Total predicted energy (24h): {total_energy:.2f} kWh")

        elif choice == '3':
            print("📦 Triggering Sync Scheduler with 30 readings (aggregated for 24h prediction)...")
            temps, hums, luxes, solars, batteries = read_arduino_n_readings(SYNC_READINGS)
            avg_temp = np.mean(temps)
            avg_hum = np.mean(hums)
            avg_solar_voltage = np.mean(solars)
            avg_battery = np.mean(batteries)
            avg_lux = np.mean(luxes)
            avg_irradiance = lux_to_irradiance(avg_lux)

            print("\n--- 30 READINGS AGGREGATED ---")
            print(
                f"Avg Temp: {avg_temp:.1f}C | "
                f"Avg Humidity: {avg_hum:.1f}% | "
                f"Avg Lux: {avg_lux:.1f} | "
                f"Avg Irradiance: {avg_irradiance:.2f} W/m² | "
                f"Avg Solar Voltage: {avg_solar_voltage:.2f}V | "
                f"Avg Battery: {avg_battery:.2f}V"
            )

            try:
                sync_scheduler.run_sync_decision(
                    battery_level=avg_battery,
                    panel_voltage=avg_solar_voltage,
                    temp=avg_temp,
                    humidity=avg_hum,
                    solar_irradiance=avg_irradiance
                )
            except Exception as e:
                print(f"❌ Error running sync scheduler: {e}")

        elif choice == '4':
            if run_light_detection is not None:
                run_light_detection()
            else:
                print("❌ Light detection module not available. Check Prabashwara/imageObjectDetection.py")

        elif choice == '5':
            print("Exiting program. Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main_menu()
# main.py
import time
import numpy as np
from solar_predictor import calculate_physical_energy, predict_24h_from_single_reading

# ---------------- MOCK SENSOR DATA ----------------
def read_sample_sensor_data(n=1):
    """
    Simulate reading sensor data.
    Returns temp (°C), hum (%), lux.
    If n>1, returns list of tuples.
    """
    import random
    if n == 1:
        temp = round(random.uniform(25, 35), 1)
        hum = round(random.uniform(40, 60), 1)
        lux = round(random.uniform(100, 1200), 0)
        return temp, hum, lux
    else:
        data = []
        for _ in range(n):
            temp = round(random.uniform(25, 35), 1)
            hum = round(random.uniform(40, 60), 1)
            lux = round(random.uniform(100, 1200), 0)
            data.append((temp, hum, lux))
        return data

# ---------------- MENU ----------------
def main_menu():
    while True:
        print("\n====== SOLAR ENERGY MONITOR ======")
        print("1. Read 1-minute sensor data & show physics energy")
        print("2. Predict 24-hour solar energy (LSTM)")
        print("3. Trigger Sync Scheduler (30 readings aggregated)")
        print("4. Run Light Detection (CNN) on test images")
        print("5. Exit")
        choice = input("Select an option (1-5): ").strip()

        if choice == '1':
            # Option 1: Physics-based energy
            temp, hum, lux = read_sample_sensor_data()
            energy = calculate_physical_energy(lux)
            print("\n📡 Sensor Reading:")
            print(f"Temp: {temp} °C | Humidity: {hum}% | Lux: {lux}")
            print(f"Instantaneous energy (physics): {energy:.4f} kWh")

        elif choice == '2':
            # Option 2: LSTM 24-hour prediction
            temp, hum, lux = read_sample_sensor_data()
            energy_24h, total_energy = predict_24h_from_single_reading(temp, hum, lux)
            print("\n📡 Sensor Reading:")
            print(f"Temp: {temp} °C | Humidity: {hum}% | Lux: {lux}")
            print("\nPredicted 24h energy (kWh per hour):")
            for i, e in enumerate(energy_24h, 1):
                print(f"Hour {i:02d}: {e:.4f} kWh")
            print(f"Total predicted energy: {total_energy:.4f} kWh")

        elif choice == '3':
            # Option 3: Sync Scheduler (30 readings)
            readings = read_sample_sensor_data(n=30)
            print("\n📡 Triggering Sync Scheduler with 30 readings:")
            print(f"{'No.':<4}{'Temp(°C)':<10}{'Hum(%)':<8}{'Lux':<8}{'Energy(kWh)':<12}")
            print("-"*45)
            for i, (temp, hum, lux) in enumerate(readings, 1):
                energy = calculate_physical_energy(lux)
                print(f"{i:<4}{temp:<10}{hum:<8}{lux:<8}{energy:<12.4f}")
            print("✅ Sync Scheduler complete.")

        elif choice == '4':
            # Option 4: CNN Light Detection (requires Arduino)
            try:
                from light_detection_module import connect_arduino, run_light_detection
                ARDUINO_PORT = 'COM9'
                BAUD_RATE = 9600
                arduino = connect_arduino(ARDUINO_PORT, BAUD_RATE)
                run_light_detection(arduino)
                arduino.close()
            except Exception as e:
                print(f"❌ Light detection failed: {e}")

        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("❌ Invalid option. Choose 1-5.")

if __name__ == "__main__":
    main_menu()
