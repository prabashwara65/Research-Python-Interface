



# solar_prediction_server.py
from pathlib import Path
import json
import ssl
import time
import numpy as np
import tensorflow as tf
import joblib
import psutil
import os
import paho.mqtt.client as mqtt

# ===============================
# CONFIGURATION
# ===============================
AWS_ENDPOINT = "ajmja1mzmi1j4-ats.iot.eu-north-1.amazonaws.com"
PORT = 8883
REQUEST_TOPIC = "solar/prediction/request"
RESPONSE_TOPIC = "solar/prediction/response"

# Root folder = Main_Program
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # src/... -> Main_Program

CERT_PATH = ROOT_DIR / "certs/6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-certificate.pem.crt"
KEY_PATH = ROOT_DIR / "certs/6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-private.pem.key"
ROOT_CA_PATH = ROOT_DIR / "certs/AmazonRootCA1.pem"

# Model paths (relative to this script)
MODEL_DIR = Path(__file__).parent / "model"
KERAS_MODEL_PATH = MODEL_DIR / "solar_lstm.keras"
SCALER_X_PATH = MODEL_DIR / "scaler_X.save"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.save"


# ===============================
# SOLAR PREDICTOR CLASS
# ===============================
class SolarPredictor:
    def __init__(self):
        """Initialize the solar predictor with model and scalers"""
        print("🔄 Loading solar prediction model...")
        
        # Load model
        self.model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), compile=False)
        
        # Load scalers
        self.scaler_X = joblib.load(str(SCALER_X_PATH))
        self.scaler_y = joblib.load(str(SCALER_Y_PATH))
        
        # Model info
        self.model_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
        
        print(f"✅ Model loaded successfully")
        print(f"   Model size: {self.model_size:.2f} MB")
    
    def prepare_input_data(self, recent_data):
        """Prepare and scale input data for prediction"""
        # Expected shape: (24, 3) for 24 hours with [irradiance, temp, humidity]
        if len(recent_data.shape) == 2:
            recent_data = recent_data.reshape(1, 24, 3)
        
        # Scale the data
        recent_data_scaled = self.scaler_X.transform(
            recent_data.reshape(24, 3)
        ).reshape(1, 24, 3)
        
        return recent_data_scaled
    
    def predict(self, recent_data):
        """Make prediction for 24-hour energy output"""
        process = psutil.Process(os.getpid())
        
        # Prepare input
        recent_data_scaled = self.prepare_input_data(recent_data)
        
        # Measure inference time
        start = time.time()
        predictions = self.model.predict(recent_data_scaled, verbose=0)
        inference_time = (time.time() - start) * 1000  # Convert to ms
        
        # Get memory usage
        memory_usage = process.memory_info().rss / (1024 * 1024)
        cpu_usage = psutil.cpu_percent()
        
        # Inverse transform predictions
        energy_kwh = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Apply physics constraints
        irradiance = recent_data[0, :, 0].flatten() if len(recent_data.shape) == 3 else recent_data[:, 0]
        energy_kwh[irradiance == 0] = 0
        energy_kwh[energy_kwh < 0] = 0
        
        # Calculate total
        total_energy = np.sum(energy_kwh)
        
        # Prepare hourly breakdown
        hourly_breakdown = []
        for i in range(24):
            hourly_breakdown.append({
                "hour": i + 1,
                "energy_kwh": float(energy_kwh[i]),
                "irradiance": float(irradiance[i]) if i < len(irradiance) else None
            })
        
        result = {
            "total_energy_kwh": float(total_energy),
            "hourly_breakdown": hourly_breakdown,
            "performance": {
                "inference_time_ms": float(inference_time),
                "memory_usage_mb": float(memory_usage),
                "cpu_usage_percent": float(cpu_usage),
                "model_size_mb": float(self.model_size)
            }
        }
        
        return result
    
    def get_default_data(self):
        """Return default test data (24 hours of sample data)"""
        return np.array([
            [0.00, 0.55, 0.85], [0.00, 0.54, 0.86], [0.00, 0.53, 0.87],
            [0.00, 0.52, 0.88], [0.00, 0.51, 0.89], [0.00, 0.50, 0.90],
            [0.15, 0.55, 0.85], [0.30, 0.60, 0.80], [0.50, 0.65, 0.75],
            [0.70, 0.70, 0.70], [0.90, 0.75, 0.65], [1.00, 0.78, 0.60],
            [0.95, 0.77, 0.62], [0.80, 0.74, 0.65], [0.60, 0.70, 0.70],
            [0.40, 0.65, 0.75], [0.20, 0.60, 0.80], [0.05, 0.58, 0.82],
            [0.00, 0.55, 0.85], [0.00, 0.54, 0.86], [0.00, 0.53, 0.87],
            [0.00, 0.52, 0.88], [0.00, 0.51, 0.89], [0.00, 0.50, 0.90],
        ])


# ===============================
# MQTT PREDICTION SERVER
# ===============================
class SolarPredictionServer:
    def __init__(self, client_id="solar_predictor"):
        self.client_id = client_id
        self.predictor = SolarPredictor()
        self.client = None
        self.running = False
        
    def create_mqtt_client(self):
        """Create and configure MQTT client"""
        client = mqtt.Client(client_id=self.client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        
        # Configure TLS
        client.tls_set(ca_certs=str(ROOT_CA_PATH),
                       certfile=str(CERT_PATH),
                       keyfile=str(KEY_PATH),
                       cert_reqs=ssl.CERT_REQUIRED,
                       tls_version=ssl.PROTOCOL_TLSv1_2,
                       ciphers=None)
        
        client.tls_insecure_set(False)
        
        # Set callbacks
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.on_disconnect = self.on_disconnect
        
        return client
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"✅ Connected to AWS IoT Core")
            # Subscribe to request topic
            client.subscribe(REQUEST_TOPIC)
            print(f"📡 Subscribed to {REQUEST_TOPIC}")
        else:
            print(f"❌ Connection failed with code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        print(f"📡 Disconnected from MQTT broker")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            print(f"\n📥 Received message on {msg.topic}")
            
            # Parse the request
            payload = json.loads(msg.payload.decode())
            print(f"   Request ID: {payload.get('request_id', 'N/A')}")
            print(f"   Device ID: {payload.get('deviceId', 'unknown')}")
            
            # Process the prediction request
            response = self.process_request(payload)
            
            # Publish response
            response_payload = json.dumps(response, indent=2)
            client.publish(RESPONSE_TOPIC, response_payload, qos=1)
            print(f"📤 Published response to {RESPONSE_TOPIC}")
            print(f"   Total energy: {response['total_energy_kwh']:.2f} kWh")
            print(f"   Inference time: {response['performance']['inference_time_ms']:.2f} ms")
            
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            # Send error response
            error_response = {
                "status": "error",
                "message": str(e),
                "timestamp": time.time(),
                "request_id": payload.get('request_id', None) if 'payload' in locals() else None
            }
            client.publish(RESPONSE_TOPIC, json.dumps(error_response), qos=1)
    
    def process_request(self, request):
        """Process prediction request and return result"""
        # Check if sensor data is provided
        if "sensor_data" in request:
            # Convert sensor data to numpy array
            # Expected format: {"sensor_data": [[irr, temp, hum], ...]}
            sensor_data = np.array(request["sensor_data"])
            
            # Validate shape
            if sensor_data.shape != (24, 3):
                raise ValueError(f"Expected sensor_data shape (24, 3), got {sensor_data.shape}")
            
            print(f"   Using provided sensor data")
            # Make prediction
            result = self.predictor.predict(sensor_data)
            
        else:
            # Use default test data
            print(f"   No sensor data provided, using default test data")
            default_data = self.predictor.get_default_data()
            result = self.predictor.predict(default_data)
        
        # Add metadata to response
        result["status"] = "success"
        result["timestamp"] = time.time()
        result["request_id"] = request.get("request_id", None)
        result["deviceId"] = request.get("deviceId", "unknown")
        
        return result
    
    def start(self):
        """Start the prediction server"""
        print("\n" + "="*50)
        print("🚀 SOLAR PREDICTION SERVER")
        print("="*50)
        
        # Create MQTT client
        self.client = self.create_mqtt_client()
        
        # Connect to AWS IoT
        print(f"📡 Connecting to {AWS_ENDPOINT}:{PORT}")
        self.client.connect(AWS_ENDPOINT, PORT)
        
        # Start network loop in background thread
        self.client.loop_start()
        self.running = True
        
        print(f"\n✅ Server is running")
        print(f"   Listening on: {REQUEST_TOPIC}")
        print(f"   Responding on: {RESPONSE_TOPIC}")
        print(f"\n   Press Ctrl+C to stop")
        print("="*50 + "\n")
    
    def stop(self):
        """Stop the prediction server"""
        if self.client and self.running:
            self.client.loop_stop()
            self.client.disconnect()
            self.running = False
            print("\n🛑 Solar Prediction Server stopped")


# ===============================
# TEST CLIENT FUNCTION
# ===============================
def send_test_request():
    """Send a test prediction request (can be used for testing)"""
    print("\n" + "="*50)
    print("🧪 SENDING TEST PREDICTION REQUEST")
    print("="*50)
    
    # Create a test client
    client = mqtt.Client(client_id="test_client", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    
    # Configure TLS
    client.tls_set(ca_certs=str(ROOT_CA_PATH),
                   certfile=str(CERT_PATH),
                   keyfile=str(KEY_PATH),
                   cert_reqs=ssl.CERT_REQUIRED,
                   tls_version=ssl.PROTOCOL_TLSv1_2)
    
    # Connect
    client.connect(AWS_ENDPOINT, PORT)
    client.loop_start()
    
    # Create test request with sample data
    test_request = {
        "request_id": f"test_{int(time.time())}",
        "deviceId": "test_device",
        "sensor_data": [
            [0.00, 0.55, 0.85], [0.00, 0.54, 0.86], [0.00, 0.53, 0.87],
            [0.00, 0.52, 0.88], [0.00, 0.51, 0.89], [0.00, 0.50, 0.90],
            [0.15, 0.55, 0.85], [0.30, 0.60, 0.80], [0.50, 0.65, 0.75],
            [0.70, 0.70, 0.70], [0.90, 0.75, 0.65], [1.00, 0.78, 0.60],
            [0.95, 0.77, 0.62], [0.80, 0.74, 0.65], [0.60, 0.70, 0.70],
            [0.40, 0.65, 0.75], [0.20, 0.60, 0.80], [0.05, 0.58, 0.82],
            [0.00, 0.55, 0.85], [0.00, 0.54, 0.86], [0.00, 0.53, 0.87],
            [0.00, 0.52, 0.88], [0.00, 0.51, 0.89], [0.00, 0.50, 0.90]
        ]
    }
    
    # Publish request
    print(f"📤 Publishing test request to {REQUEST_TOPIC}")
    client.publish(REQUEST_TOPIC, json.dumps(test_request), qos=1)
    
    # Wait for response (simplified - in production use proper callback)
    time.sleep(3)
    
    client.loop_stop()
    client.disconnect()
    print("✅ Test request sent")


# ===============================
# MAIN ENTRY POINT
# ===============================
def main():
    """Main entry point - start the prediction server"""
    server = SolarPredictionServer()
    
    try:
        server.start()
        # Keep the program running
        while server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Send a test request
        send_test_request()
    else:
        # Start the server
        main()
