import cv2
import numpy as np
import serial
import time
from collections import deque
from threading import Thread
from queue import Queue, Empty

# ---------------- CONFIG ----------------
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

PAN_MIN, PAN_MAX = 10, 80
TILT_MIN, TILT_MAX = 10, 80

SERVO_UPDATE_INTERVAL = 0.05
MAX_SERVO_STEP = 6
BRIGHTNESS_THRESHOLD = 240
INVERT_PAN = False
INVERT_TILT = True
POSITION_HISTORY = 3
DEBUG = True
# ---------------------------------------

# ---------------- GLOBAL VARIABLES ----------------
D10_GLOBAL = 0  # ARM LEFT/RIGHT
D12_GLOBAL = 0  # ARM UP/DOWN

# Connect to Arduino
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("âœ… Arduino connected")
except:
    arduino = None
    print("âš ï¸ Arduino not connected (visual mode)")

# Shared queue for servo commands
servo_queue = Queue(maxsize=1)

# Thread to send angles to Arduino non-blocking
def serial_thread():
    global D10_GLOBAL, D12_GLOBAL
    while True:
        angles = servo_queue.get()
        if angles is None:
            break  # exit signal
        if arduino:
            try:
                cmd = f"{angles[0]},{angles[1]}\n"
                arduino.write(cmd.encode())
            except:
                pass

        # ------------------- READ D10/D12 -------------------
        if arduino:
            while arduino.in_waiting:
                line = arduino.readline().decode().strip()
                if line:  # Expecting CSV like "temp,hum,lux,D10,D12"
                    parts = line.split(',')
                    if len(parts) >= 5:
                        try:
                            D10_GLOBAL = int(parts[3])
                            D12_GLOBAL = int(parts[4])
                            if DEBUG:
                                print(f"ðŸŒŸ Updated globals: D10={D10_GLOBAL}, D12={D12_GLOBAL}")
                        except ValueError:
                            pass

# Start serial thread
thread = Thread(target=serial_thread, daemon=True)
thread.start()

# ---------------- Helper functions ----------------
def map_value(val, old_min, old_max, new_min, new_max):
    return int((val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min)

def smooth_move(current, target):
    smoothed = []
    for c, t in zip(current, target):
        diff = t - c
        step = np.clip(diff, -MAX_SERVO_STEP, MAX_SERVO_STEP)
        smoothed.append(c + step)
    return smoothed

def send_servo_latest(angles):
    if servo_queue.full():
        try:
            servo_queue.get_nowait()  # drop old command
        except Empty:
            pass
    servo_queue.put(angles)

def find_camera(max_index=3):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"âœ… Camera found at index {i}")
            return cap
        cap.release()
    return None

# ---------------- Main loop ----------------
def run():
    global D10_GLOBAL, D12_GLOBAL
    cap = find_camera()
    if cap is None:
        print("âŒ No camera found")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    position_history = deque(maxlen=POSITION_HISTORY)
    current_servo = [45, 45]
    target_servo = [45, 45]
    last_servo_update = time.time()

    print("ðŸ”† Headless threaded light tracker running (Ctrl+C to stop)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(mask)

            if coords is not None:
                cx = int(coords[:, :, 0].mean())
                cy = int(coords[:, :, 1].mean())
                position_history.append((cx, cy))
                avg_cx = int(sum(p[0] for p in position_history) / len(position_history))
                avg_cy = int(sum(p[1] for p in position_history) / len(position_history))

                pan = map_value(avg_cx, 0, w, PAN_MAX, PAN_MIN) if INVERT_PAN else map_value(avg_cx, 0, w, PAN_MIN, PAN_MAX)
                tilt = map_value(avg_cy, 0, h, TILT_MAX, TILT_MIN) if INVERT_TILT else map_value(avg_cy, 0, h, TILT_MIN, TILT_MAX)
                target_servo = [np.clip(pan, PAN_MIN, PAN_MAX), np.clip(tilt, TILT_MIN, TILT_MAX)]

            now = time.time()
            if now - last_servo_update > SERVO_UPDATE_INTERVAL:
                current_servo = smooth_move(current_servo, target_servo)
                send_servo_latest(current_servo)
                last_servo_update = now

                if DEBUG:
                    print(f"Servo H:{current_servo[0]} V:{current_servo[1]} | Light: {(avg_cx, avg_cy) if coords is not None else 'None'}")
                    print(f"ðŸ’¡ Global ARM positions: D10={D10_GLOBAL}, D12={D12_GLOBAL}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nðŸ‘‹ Stopping tracker...")

    finally:
        cap.release()
        send_servo_latest([45, 45])  # return to center
        servo_queue.put(None)  # signal serial thread to exit
        thread.join()
        if arduino:
            arduino.close()
        print("âœ… Tracker stopped")

if __name__ == "__main__":
    run()