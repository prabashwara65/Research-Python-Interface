import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import serial
import time

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "light_segmentation_cnn.h5")
IMG_SIZE = 128
ARDUINO_PORT = 'COM9'
BAUD_RATE = 9600
PAN_MIN, PAN_MAX = 10, 80
TILT_MIN, TILT_MAX = 10, 80
# ---------------------------------------

# Load CNN model
model = load_model(MODEL_PATH)
print("✅ CNN Model loaded")

# Connect to Arduino
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
    time.sleep(2)
    print("✅ Arduino connected")
except Exception as e:
    print(f"❌ Cannot connect to Arduino: {e}")
    arduino = None

def map_value(val, old_min, old_max, new_min, new_max):
    """Map value from one range to another."""
    return int((val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min)

def run_light_detection():
    cap = cv2.VideoCapture(0)  # usually 0 for default webcam
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🔆 Running Light Detection (ESC to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Preprocess for CNN
        img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_input = np.expand_dims(img_resized / 255.0, axis=0)

        # CNN prediction
        mask_pred = model.predict(img_input, verbose=0)[0, :, :, 0]
        mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_pred, (w, h))

        coords = cv2.findNonZero(mask_resized)
        if coords is not None:
            cx = int(coords[:, 0, 0].mean())
            cy = int(coords[:, 0, 1].mean())
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            text = f"px=({cx},{cy})"

            # Servo control
            if arduino is not None:
                angleH = map_value(cx, 0, w, PAN_MAX, PAN_MIN)
                angleV = map_value(cy, 0, h, TILT_MAX, TILT_MIN)
                cmd = f"{angleH},{angleV}\n"
                arduino.write(cmd.encode())
                arduino.flush()  # Important!
                time.sleep(0.05)  # Small delay for servo movement
                print(f"{text} | Servo H:{angleH} V:{angleV}")
        else:
            text = "Light not detected"

        # Display
        cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hybrid ML + CV", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    if arduino is not None:
        arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_light_detection()
