import serial
import time
import sys
import numpy as np

# ---------------- CONFIG ----------------
ARDUINO_PORT = 'COM9'
BAUD_RATE = 9600
READINGS = 30  # Number of readings to collect
RETRY_INTERVAL = 2  # seconds to wait if COM port is busy
# ----------------------------------------

def connect_arduino(port, baud):
    """Try connecting to Arduino. Retry until successful."""
    while True:
        try:
            arduino = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # allow Arduino to reset
            print(f"✅ Connected to Arduino on {port}")
            return arduino
        except serial.SerialException as e:
            print(f"❌ Could not open {port}: {e}")
            print(f"⏳ Retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)

def read_battery_data(arduino, readings=30):
    """Read battery voltage from Arduino and return list of values."""
    voltages = []
    count = 0
    print(f"\n📡 Reading battery voltage from Arduino ({readings} readings)...")
    while count < readings:
        try:
            if arduino.in_waiting:
                line = arduino.readline().decode('utf-8').strip()
                if not line:
                    continue
                # Arduino sends CSV: TEMP,HUM,LUX,SOLAR,BATTERY
                parts = line.split(',')
                if len(parts) == 5:
                    try:
                        temp = float(parts[0])
                        hum = float(parts[1])
                        lux = float(parts[2])
                        solar = float(parts[3])
                        battery = float(parts[4])
                        voltages.append(battery)
                        count += 1
                        print(f"{count:02d}: Temp={temp:.1f}C | Hum={hum:.1f}% | Lux={lux:.1f} | Solar={solar:.2f}V | Battery={battery:.2f}V")
                    except ValueError:
                        continue
        except Exception as e:
            print("❌ Error reading from Arduino:", e)
            break
    return voltages

def main():
    arduino = connect_arduino(ARDUINO_PORT, BAUD_RATE)
    try:
        voltages = read_battery_data(arduino, READINGS)
        avg_voltage = np.mean(voltages)
        print(f"\n🔋 Average battery voltage: {avg_voltage:.2f} V")
    finally:
        arduino.close()
        print("🔌 Arduino connection closed.")

if __name__ == "__main__":
    main()
