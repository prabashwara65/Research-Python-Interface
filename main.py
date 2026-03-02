# main_monitor.py

import time
import numpy as np
import os
import sys
import random
import datetime as dt

# ---------------- SETTINGS ----------------
ARDUINO_PORT = 'COM9'  # Update your COM port
BAUD_RATE = 9600
READINGS_PER_MIN = 30
SYNC_READINGS = 30
# ------------------------------------------

# Add Hasara folder to path for sync_scheduler import
HASARA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hasara")
if HASARA_PATH not in sys.path:
    sys.path.append(HASARA_PATH)

import sync_scheduler  # Your sync_scheduler.py

# Add Dewhara folder to path
DEWHARA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dewhara")
if DEWHARA_PATH not in sys.path:
    sys.path.append(DEWHARA_PATH)

try:
    from Sithmi.solar_predictor import predict_24h_from_single_reading
except ImportError:
    print("⚠️ solar_predictor not found. Option 2 will fail if used.")

# ---------------- ARDUINO ----------------
try:
    import serial
    ARDUINO_AVAILABLE = True
except ImportError:
    print("⚠️ PySerial not installed, will use mock data")
    ARDUINO_AVAILABLE = False

# ---------------- LED CURRENT MAP ----------------
LED_CURRENT_MAP = {0: 0.0, 1: 0.02, 2: 0.04, 3: 0.06}  # Current in A per LEDs ON

# ---------------- ARDUINO DATA ----------------
def read_arduino_data(show_live=True, readings=READINGS_PER_MIN, include_leds=True):
    """
    Read sensor data from Arduino. Returns averages of temp, hum, lux, solar, battery.
    """
    if not ARDUINO_AVAILABLE:
        return read_mock_data(show_live=show_live, readings=readings, include_leds=include_leds)

    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        print(f"✅ Arduino connected on {ARDUINO_PORT}, reading {readings} readings...")
    except Exception as e:
        print(f"⚠️ Could not connect to Arduino: {e}")
        return read_mock_data(show_live=show_live, readings=readings, include_leds=include_leds)

    data_list = []
    header = f"{'No.':>3} | {'Temp(C)':>7} | {'Hum(%)':>6} | {'Lux':>6} | {'Solar(V)':>8} | {'Batt(V)':>7}"
    if include_leds:
        header += f" | {'LEDs':>4} | {'Current(A)':>10}"
    print(header)
    print("-"*80)

    count = 0
    try:
        while count < readings:
            line = arduino.readline().decode(errors='ignore').strip()
            if not line:
                continue
            try:
                parts = list(map(float, line.split(',')))
                temp, hum, lux, solar, battery = parts[:5]

                if include_leds:
                    leds = parts[5:]
                    led_count = int(sum(leds))
                    current_load = LED_CURRENT_MAP.get(led_count, 0.0)
                    data_list.append((temp, hum, lux, solar, battery, led_count, current_load))
                    if show_live:
                        print(f"{count+1:03d} | {temp:7.1f} | {hum:6.1f} | {lux:6.0f} | {solar:8.2f} | {battery:7.2f} | {led_count:>4} | {current_load:10.3f}")
                else:
                    data_list.append((temp, hum, lux, solar, battery))
                    if show_live:
                        print(f"{count+1:03d} | {temp:7.1f} | {hum:6.1f} | {lux:6.0f} | {solar:8.2f} | {battery:7.2f}")

                count += 1
            except:
                continue
    finally:
        arduino.close()
        print("✅ Arduino connection closed.\n")

    data_array = np.array(data_list)
    avg_values = np.mean(data_array, axis=0)
    return tuple(avg_values)

# ---------------- MOCK DATA ----------------
def read_mock_data(show_live=True, readings=READINGS_PER_MIN, include_leds=True):
    data_list = []
    print("⚠️ Using mock sensor data")
    header = f"{'No.':>3} | {'Temp(C)':>7} | {'Hum(%)':>6} | {'Lux':>6} | {'Solar(V)':>8} | {'Batt(V)':>7}"
    if include_leds:
        header += f" | {'LEDs':>4} | {'Current(A)':>10}"
    print(header)
    print("-"*80)

    for i in range(readings):
        temp = round(random.uniform(25, 35), 1)
        hum = round(random.uniform(40, 60), 1)
        lux = round(random.uniform(100, 1200), 1)
        solar = round(random.uniform(3.5, 5.0), 2)
        battery = round(random.uniform(3.6, 4.2), 2)
        if include_leds:
            led_count = random.randint(0, 3)
            current_load = LED_CURRENT_MAP[led_count]
            data_list.append((temp, hum, lux, solar, battery, led_count, current_load))
            if show_live:
                print(f"{i+1:03d} | {temp:7.1f} | {hum:6.1f} | {lux:6.0f} | {solar:8.2f} | {battery:7.2f} | {led_count:>4} | {current_load:10.3f}")
        else:
            data_list.append((temp, hum, lux, solar, battery))
            if show_live:
                print(f"{i+1:03d} | {temp:7.1f} | {hum:6.1f} | {lux:6.0f} | {solar:8.2f} | {battery:7.2f}")
        time.sleep(0.05)

    data_array = np.array(data_list)
    avg_values = np.mean(data_array, axis=0)
    return tuple(avg_values)

# ---------------- MAIN MENU ----------------
def main_menu():
    while True:
        print("\n====== SOLAR ENERGY MONITOR ======")
        print("1. Run Dewhara Energy Forecast")
        print("2. Predict 24-hour solar energy (LSTM)")
        print("3. Trigger Sync Scheduler (30 readings aggregated)")
        print("4. Exit")
        choice = input("Select an option (1-4): ").strip()

        if choice == '1':
            try:
                import enegy_forecast
                enegy_forecast.run_forecast()
            except Exception as e:
                print(f"❌ Error running Dewhara enegy_forecast: {e}")

        elif choice == '2':
            # Solar energy prediction using solar voltage
            avg_temp, avg_hum, _, avg_solar_voltage, _ = read_arduino_data(show_live=True, include_leds=False)
            try:
                energy_24h, total_energy = predict_24h_from_single_reading(avg_temp, avg_hum, avg_solar_voltage)
                print("\n--- 24-HOUR SOLAR ENERGY PREDICTION ---")
                print("Hour | Predicted Energy (kWh)")
                print("-----------------------------")
                for i, e in enumerate(energy_24h, 1):
                    print(f"{i:02d}   | {e:.3f}")
                print(f"\n🔋 Total predicted energy (24h): {total_energy:.2f} kWh")
            except Exception as e:
                print(f"❌ Error predicting 24h solar energy: {e}")

        elif choice == '3':
            # Sync Scheduler with aggregated readings (automatic, no user input)
            print(f"📦 Triggering Sync Scheduler with {SYNC_READINGS} readings...")
            avg_temp, avg_hum, _, avg_solar_voltage, avg_battery_voltage = read_arduino_data(
                show_live=True, readings=SYNC_READINGS, include_leds=False
            )

            print("\n--- 30 READINGS AGGREGATED ---")
            print(
                f"Avg Temp: {avg_temp:.1f}C | "
                f"Avg Humidity: {avg_hum:.1f}% | "
                f"Avg Solar Voltage: {avg_solar_voltage:.2f}V | "
                f"Avg Battery: {avg_battery_voltage:.2f}V"
            )

            try:
                # Automatically pass default values for optional parameters
                sync_scheduler.run_sync_decision(
                    battery_level=avg_battery_voltage,
                    panel_voltage=avg_solar_voltage,
                    temp=avg_temp,
                    humidity=avg_hum,
                    solar_irradiance=avg_solar_voltage * 100  # approximate W/m²
                )
            except Exception as e:
                print(f"❌ Error running sync scheduler: {e}")

        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main_menu()
