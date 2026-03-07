#!/usr/bin/env python3
"""
Solar Prediction Main Program
Handles both incoming prediction requests and outgoing sensor data
"""

import os
import json
import time
import threading
import queue
import numpy as np
import tensorflow as tf
import joblib
import boto3
import ssl
import signal
import sys
import psutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from decimal import Decimal

# AWS IoT Core SDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Paho MQTT for client publishing
import paho.mqtt.client as mqtt

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =============================
# CONFIGURATION
# =============================
class Config:
    """Central configuration class"""
    
    # AWS IoT Core Settings (Server/Listener)
    SERVER_CLIENT_ID = "solar_predictor_server"
    AWS_IOT_ENDPOINT = "ajmja1mzmi1j4-ats.iot.eu-north-1.amazonaws.com"
    REQUEST_TOPIC = "solar/prediction/request"
    RESPONSE_TOPIC = "solar/prediction/response"
    
    # Client Publisher Settings
    CLIENT_ID = "hasara_client"
    CLIENT_PUBLISH_TOPIC = "solar/prediction/result"
    
    # DynamoDB
    TABLE_NAME = "SithmiSolarPredictResults"
    REGION = "eu-north-1"
    ENABLE_DYNAMODB = True  # Set to False if you don't have DynamoDB
    
    # Paths
    ROOT_DIR = Path(__file__).resolve().parent
    CERTS_DIR = ROOT_DIR / "certs"
    MODEL_DIR = ROOT_DIR / "model"
    
    # Certificate paths (Server)
    SERVER_ROOT_CA = CERTS_DIR / "AmazonRootCA1.pem"
    SERVER_CERT = CERTS_DIR / "certificate.pem.crt"
    SERVER_PRIVATE_KEY = CERTS_DIR / "private.pem.key"
    
    # Certificate paths (Client) - Using your specific client certs
    CLIENT_CERT = CERTS_DIR / "6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-certificate.pem.crt"
    CLIENT_KEY = CERTS_DIR / "6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-private.pem.key"
    CLIENT_ROOT_CA = CERTS_DIR / "AmazonRootCA1.pem"
    
    # Model paths
    KERAS_MODEL_PATH = MODEL_DIR / "solar_lstm.keras"
    SCALER_X_PATH = MODEL_DIR / "scaler_X.save"
    SCALER_Y_PATH = MODEL_DIR / "scaler_y.save"
    
    # Performance monitoring
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_LOG_INTERVAL = 3600  # Log performance every hour
    
    # Threading
    NUM_WORKER_THREADS = 4
    REQUEST_QUEUE_SIZE = 100

# =============================
# LOGGING SETUP
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solar_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================
# DYNAMODB HELPER
# =============================
def convert_floats_to_decimal(obj):
    """
    Recursively convert floats to Decimal for DynamoDB compatibility
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_floats_to_decimal(item) for item in obj]
    elif isinstance(obj, np.floating):
        return Decimal(str(float(obj)))
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

# =============================
# MODEL MANAGER
# =============================
class ModelManager:
    """Singleton class to manage model and scalers"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.scaler_X = None
        self.scaler_Y = None
        self.model_size_mb = 0
        self.load_time = 0
        self._last_inference_time = 0
        self._initialized = True
        
    def load_models(self):
        """Load all models and scalers"""
        logger.info("=" * 50)
        logger.info("LOADING SOLAR PREDICTION MODEL")
        logger.info("=" * 50)
        
        try:
            # Check if model files exist
            if not Config.KERAS_MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {Config.KERAS_MODEL_PATH}")
            if not Config.SCALER_X_PATH.exists():
                raise FileNotFoundError(f"Scaler X not found at {Config.SCALER_X_PATH}")
            if not Config.SCALER_Y_PATH.exists():
                raise FileNotFoundError(f"Scaler Y not found at {Config.SCALER_Y_PATH}")
            
            # Load model with timing
            start_time = time.time()
            
            logger.info(f"Loading Keras model from: {Config.KERAS_MODEL_PATH}")
            self.model = tf.keras.models.load_model(Config.KERAS_MODEL_PATH, compile=False)
            
            logger.info(f"Loading X scaler from: {Config.SCALER_X_PATH}")
            self.scaler_X = joblib.load(Config.SCALER_X_PATH)
            
            logger.info(f"Loading Y scaler from: {Config.SCALER_Y_PATH}")
            self.scaler_Y = joblib.load(Config.SCALER_Y_PATH)
            
            self.load_time = time.time() - start_time
            self.model_size_mb = os.path.getsize(Config.KERAS_MODEL_PATH) / (1024 * 1024)
            
            logger.info(f"✅ Model loaded successfully in {self.load_time:.2f} seconds")
            logger.info(f"📊 Model size: {self.model_size_mb:.2f} MB")
            
            # Log model summary
            self.model.summary(print_fn=lambda x: logger.info(f"  {x}"))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def prepare_input(self, recent_data: np.ndarray) -> np.ndarray:
        """Scale and reshape input data"""
        try:
            # Ensure data is in the right shape for scaling
            if len(recent_data.shape) == 3:
                # Data is already batched: (batch, 24, 3)
                batch_size = recent_data.shape[0]
                # Reshape to (batch*24, 3) for scaling
                reshaped = recent_data.reshape(-1, 3)
                scaled = self.scaler_X.transform(reshaped)
                # Reshape back to (batch, 24, 3)
                return scaled.reshape(batch_size, 24, 3)
            elif len(recent_data.shape) == 2:
                # Data is (24, 3) - single sample
                scaled = self.scaler_X.transform(recent_data)
                return scaled.reshape(1, 24, 3)
            else:
                raise ValueError(f"Unexpected input shape: {recent_data.shape}")
                
        except Exception as e:
            logger.error(f"Error preparing input: {e}")
            raise
    
    def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """Run prediction on input data and return formatted results"""
        try:
            # Prepare input
            scaled_data = self.prepare_input(data)

            # Run prediction and capture inference time
            start_time = time.time()
            predictions = self.model.predict(scaled_data, verbose=0)
            inference_time = (time.time() - start_time) * 1000

            # Debug: print raw predictions
            logger.debug(f"Raw predictions: {predictions}")

            # Inverse transform
            energy_kwh = self.scaler_Y.inverse_transform(predictions.reshape(-1, 1)).flatten()

            # Debug: print after inverse scaling
            logger.debug(f"Inverse scaled predictions: {energy_kwh}")

            # Extract irradiance
            if len(data.shape) == 3:
                irradiance = data[0, :, 0].flatten()
            else:
                irradiance = data[:, 0]

            # Debug: print irradiance
            logger.debug(f"Irradiance values: {irradiance}")

            # Apply physics constraints carefully
            # Only zero out when irradiance is truly zero, not when scaler mismatch
            for i in range(len(energy_kwh)):
                if irradiance[i] <= 0:
                    energy_kwh[i] = 0
                elif energy_kwh[i] < 0:
                    energy_kwh[i] = 0

            # Calculate total energy
            total_energy = float(np.sum(energy_kwh))

            # Format response
            hourly_data = [{"hour": i+1, "energy_kwh": float(energy_kwh[i])} for i in range(24)]
            hourly_output = [f"{i+1:02d} | {energy_kwh[i]:.3f}" for i in range(24)]

            return {
                "total_energy_kwh": total_energy,
                "total_energy_formatted": f"Total 24h Energy: {total_energy:.2f} kWh",
                "hourly_energy": [float(e) for e in energy_kwh],
                "hourly_data": hourly_data,
                "hourly_output": hourly_output,
                "irradiance": [float(i) for i in irradiance],
                "performance": self.get_prediction_performance(inference_time)
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    
    def get_prediction_performance(self, inference_time_ms: float) -> Dict[str, Any]:
        """Get performance metrics for the prediction"""
        process = psutil.Process(os.getpid())
        
        return {
            "inference_time_ms": inference_time_ms,
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "ram_usage_mb": process.memory_info().rss / (1024 * 1024)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        process = psutil.Process(os.getpid())
        
        return {
            "model_size_mb": self.model_size_mb,
            "load_time_seconds": self.load_time,
            "ram_usage_mb": process.memory_info().rss / (1024 * 1024),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "thread_count": threading.active_count()
        }

# =============================
# MQTT SERVER (Request Handler)
# =============================
class SolarPredictionServer:
    """Handles incoming prediction requests via AWS IoT Core"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.mqtt_client = None
        self.running = False
        self.request_queue = queue.Queue(maxsize=Config.REQUEST_QUEUE_SIZE)
        self.worker_threads = []
        self.dynamodb = None
        self.table = None
        
        # Initialize DynamoDB only if enabled
        if Config.ENABLE_DYNAMODB:
            try:
                self.dynamodb = boto3.resource("dynamodb", region_name=Config.REGION)
                self.table = self.dynamodb.Table(Config.TABLE_NAME)
                logger.info("✅ DynamoDB initialized")
            except Exception as e:
                logger.warning(f"⚠️ DynamoDB initialization failed: {e}")
                self.dynamodb = None
        else:
            logger.info("ℹ️ DynamoDB logging is disabled")
    
    def connect(self):
        """Connect to AWS IoT Core"""
        try:
            self.mqtt_client = AWSIoTMQTTClient(Config.SERVER_CLIENT_ID)
            
            # Configure endpoint
            self.mqtt_client.configureEndpoint(Config.AWS_IOT_ENDPOINT, 8883)
            
            # Configure credentials
            self.mqtt_client.configureCredentials(
                str(Config.SERVER_ROOT_CA),
                str(Config.SERVER_PRIVATE_KEY),
                str(Config.SERVER_CERT)
            )
            
            # Configure MQTT client
            self.mqtt_client.configureOfflinePublishQueueing(-1)  # Infinite queue
            self.mqtt_client.configureDrainingFrequency(2)  # 2 Hz
            self.mqtt_client.configureConnectDisconnectTimeout(10)
            self.mqtt_client.configureMQTTOperationTimeout(5)
            self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
            
            # Connect
            logger.info("📡 Connecting to AWS IoT Core...")
            self.mqtt_client.connect()
            logger.info("✅ Connected to AWS IoT Core")
            
            # Subscribe to request topic
            self.mqtt_client.subscribe(Config.REQUEST_TOPIC, 1, self._message_callback)
            logger.info(f"📡 Subscribed to topic: {Config.REQUEST_TOPIC}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to AWS IoT Core: {e}")
            return False
    
    def _message_callback(self, client, userdata, message):
        """Callback for incoming MQTT messages"""
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            request_id = payload.get('requestId') or payload.get('request_id')  # Handle both formats
            logger.info(f"📩 Request received with requestId: {request_id}")
            
            # Add to processing queue
            self.request_queue.put({
                'topic': message.topic,
                'payload': payload,
                'timestamp': time.time()
            })
            
            logger.debug(f"Queue size: {self.request_queue.qsize()}")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON payload: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")

    def _extract_sensor_array(self, payload: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract and normalize sensor data to shape (24, 3)."""
        source = payload
        if isinstance(payload.get("payload"), dict):
            source = payload["payload"]
        elif isinstance(payload.get("message"), dict):
            source = payload["message"]

        sensor_data = (
            source.get("sensor_data")
            or source.get("sensorData")
            or source.get("data")
        )
        if sensor_data is None:
            return None

        if isinstance(sensor_data, list) and sensor_data and isinstance(sensor_data[0], dict):
            normalized = []
            for row in sensor_data:
                irr = row.get("irradiance", row.get("irr", row.get("solar", 0)))
                temp = row.get("temperature", row.get("temp", 0))
                hum = row.get("humidity", row.get("hum", 0))
                normalized.append([irr, temp, hum])
            sensor_data = normalized

        sensor_array = np.array(sensor_data, dtype=np.float32)
        if sensor_array.shape == (1, 24, 3):
            sensor_array = sensor_array[0]
        return sensor_array
    
    def _process_request(self, request: Dict[str, Any]):
        """Process a single prediction request"""
        try:
            payload = request['payload']
            
            # Get requestId - check both formats (requestId or request_id)
            request_id = payload.get('requestId') or payload.get('request_id')
            if not request_id:
                request_id = f"auto_{int(time.time() * 1000)}"
                logger.warning(f"No requestId provided, generated: {request_id}")
            
            device_id = payload.get('deviceId') or payload.get('device_id') or 'unknown'
            
            logger.info(f"🔄 Processing request {request_id} from device {device_id}")
            
            sensor_array = self._extract_sensor_array(payload)
            if sensor_array is not None:
                # Validate shape
                if sensor_array.shape != (24, 3):
                    error_msg = (
                        f"Invalid sensor data shape: {sensor_array.shape}, expected (24, 3). "
                        f"Payload keys: {list(payload.keys())}"
                    )
                    logger.error(f"❌ {error_msg}")
                    self._send_error_response(request_id, device_id, error_msg)
                    return
            else:
                error_msg = (
                    "No sensor data found in payload. Expected one of: sensor_data, sensorData, data "
                    "(directly or inside payload/message wrapper). "
                    f"Payload keys: {list(payload.keys())}"
                )
                logger.error(f"❌ {error_msg}")
                self._send_error_response(request_id, device_id, error_msg)
                return
            
            # Run prediction
            result = self.model_manager.predict(sensor_array)
            
            # Prepare response with formatted data
            response = {
                "requestId": request_id,
                "deviceId": device_id,
                "timestamp": time.time(),
                "timestamp_str": datetime.fromtimestamp(time.time()).isoformat(),
                "inference_time_ms": result['performance']['inference_time_ms'],
                "status": "success",
                "total_energy_kwh": result['total_energy_kwh'],
                "total_energy_formatted": result['total_energy_formatted'],
                "hourly_energy": result['hourly_energy'],
                "hourly_output": result['hourly_output'],  # Formatted hourly output
                "irradiance": result['irradiance'],
                "performance": result['performance']
            }
            
            # Also create a simplified response with just the total energy if needed
            simple_response = {
                "requestId": request_id,
                "deviceId": device_id,
                "total_energy": result['total_energy_formatted'],
                "total_energy_kwh": round(result['total_energy_kwh'], 2),
                "timestamp": time.time()
            }
            
            # Publish full response to response topic
            self.mqtt_client.publish(
                Config.RESPONSE_TOPIC,
                json.dumps(response),
                1
            )
            logger.info(f"✅ Full response sent for request {request_id} (inference: {result['performance']['inference_time_ms']:.2f}ms)")
            
            # Also publish simplified response to result topic
            self.mqtt_client.publish(
                Config.CLIENT_PUBLISH_TOPIC,
                json.dumps(simple_response),
                1
            )
            logger.info(f"✅ Simplified result published to {Config.CLIENT_PUBLISH_TOPIC}: {result['total_energy_formatted']}")
            
            # Log to DynamoDB if available
            if self.dynamodb and self.table:
                try:
                    # Convert floats to Decimal for DynamoDB
                    dynamodb_item = convert_floats_to_decimal(response)
                    self.table.put_item(Item=dynamodb_item)
                    logger.info(f"💾 Prediction logged to DynamoDB for request {request_id}")
                except Exception as e:
                    logger.error(f"❌ DynamoDB logging failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            # Get request_id safely
            error_request_id = 'unknown'
            if 'payload' in locals():
                error_request_id = payload.get('requestId') or payload.get('request_id', 'unknown')
            error_device_id = (
                (payload.get('deviceId') or payload.get('device_id') or 'unknown')
                if 'payload' in locals()
                else 'unknown'
            )
            self._send_error_response(error_request_id, error_device_id, str(e))
    
    def _send_error_response(self, request_id: str, device_id: str, error_message: str):
        """Send error response"""
        try:
            error_response = {
                "requestId": request_id,
                "deviceId": device_id,
                "timestamp": time.time(),
                "status": "error",
                "error": error_message
            }
            
            # Send error to response topic
            self.mqtt_client.publish(
                Config.RESPONSE_TOPIC,
                json.dumps(error_response),
                1
            )
            
            # Also send to result topic
            self.mqtt_client.publish(
                Config.CLIENT_PUBLISH_TOPIC,
                json.dumps(error_response),
                1
            )
            
            logger.info(f"⚠️ Error response sent for request {request_id}: {error_message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send error response: {e}")
    
    def _worker_thread(self, thread_id: int):
        """Worker thread for processing requests"""
        logger.info(f"🧵 Worker thread {thread_id} started")
        
        while self.running:
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=1)
                self._process_request(request)
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Worker thread {thread_id} error: {e}")
        
        logger.info(f"🧵 Worker thread {thread_id} stopped")
    
    def start(self):
        """Start the server"""
        if not self.connect():
            logger.error("❌ Failed to start server")
            return False
        
        self.running = True
        
        # Start worker threads
        for i in range(Config.NUM_WORKER_THREADS):
            thread = threading.Thread(
                target=self._worker_thread,
                args=(i,),
                name=f"Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"🚀 Server started with {Config.NUM_WORKER_THREADS} worker threads")
        return True
    
    def stop(self):
        """Stop the server"""
        logger.info("🛑 Stopping server...")
        self.running = False
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        # Disconnect MQTT
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
                logger.info("✅ Disconnected from AWS IoT Core")
            except:
                pass
        
        logger.info("✅ Server stopped")

# =============================
# MQTT CLIENT (Data Publisher)
# =============================
class SolarDataPublisher:
    """Publishes sensor data to AWS IoT Core"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to AWS IoT Core as a publisher"""
        try:
            self.client = mqtt.Client(
                client_id=Config.CLIENT_ID,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            
            # Configure TLS
            self.client.tls_set(
                ca_certs=str(Config.CLIENT_ROOT_CA),
                certfile=str(Config.CLIENT_CERT),
                keyfile=str(Config.CLIENT_KEY),
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLSv1_2
            )
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            
            # Connect
            logger.info("📡 Connecting publisher to AWS IoT Core...")
            self.client.connect(Config.AWS_IOT_ENDPOINT, 8883, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info("✅ Publisher connected to AWS IoT Core")
                return True
            else:
                logger.error("❌ Publisher connection timeout")
                return False
            
        except Exception as e:
            logger.error(f"❌ Publisher connection failed: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Connection callback"""
        if rc == 0:
            self.connected = True
            logger.info("✅ Publisher connected successfully")
        else:
            logger.error(f"❌ Publisher connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Disconnection callback"""
        self.connected = False
        logger.warning("⚠️ Publisher disconnected")
    
    def _on_publish(self, client, userdata, mid, rc, properties=None):
        """Publish callback"""
        logger.debug(f"📤 Message {mid} published")
    
    def publish_sensor_data(self, sensor_data: List[List[float]], device_id: str = "Raspberry") -> bool:
        """
        Publish sensor data to AWS IoT Core
        
        Args:
            sensor_data: 24x3 array of sensor readings
            device_id: Device identifier
            
        Returns:
            bool: True if published successfully
        """
        if not self.connected:
            logger.error("❌ Publisher not connected")
            return False
        
        try:
            # Prepare payload
            payload = {
                "deviceId": device_id,
                "timestamp": time.time(),
                "data": sensor_data
            }
            
            # Publish
            result = self.client.publish(
                Config.CLIENT_PUBLISH_TOPIC,
                json.dumps(payload),
                qos=1
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"📤 Sensor data published to {Config.CLIENT_PUBLISH_TOPIC}")
                return True
            else:
                logger.error(f"❌ Publish failed with code: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to publish sensor data: {e}")
            return False
    
    def disconnect(self):
        """Disconnect publisher"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("✅ Publisher disconnected")

# =============================
# PERFORMANCE MONITOR
# =============================
class PerformanceMonitor:
    """Monitors and logs system performance"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.running = False
        self.monitor_thread = None
        self.metrics_history = []
        
    def start(self):
        """Start performance monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Performance monitor started")
        
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Save metrics to file
        if self.metrics_history:
            metrics_file = Path(f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"📊 Performance metrics saved to {metrics_file}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self.model_manager.get_performance_metrics()
                metrics['timestamp'] = time.time()
                metrics['timestamp_str'] = datetime.now().isoformat()
                
                self.metrics_history.append(metrics)
                
                # Log periodically
                logger.info("📊 Performance Metrics:")
                logger.info(f"  RAM Usage: {metrics['ram_usage_mb']:.2f} MB")
                logger.info(f"  CPU Usage: {metrics['cpu_percent']:.1f}%")
                logger.info(f"  Threads: {metrics['thread_count']}")
                
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Sleep for interval
                for _ in range(Config.PERFORMANCE_LOG_INTERVAL):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ Performance monitor error: {e}")
                time.sleep(60)

# =============================
# MAIN APPLICATION
# =============================
class SolarPredictionApp:
    """Main application coordinating all components"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.server = None
        self.publisher = None
        self.monitor = None
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("SOLAR PREDICTION SYSTEM - INITIALIZING")
        logger.info("=" * 60)
        
        # Load model
        if not self.model_manager.load_models():
            return False
        
        # Initialize server
        self.server = SolarPredictionServer(self.model_manager)
        
        # Initialize publisher
        self.publisher = SolarDataPublisher()
        
        # Initialize performance monitor
        if Config.ENABLE_PERFORMANCE_MONITORING:
            self.monitor = PerformanceMonitor(self.model_manager)
        
        logger.info("=" * 60)
        logger.info("✅ SYSTEM INITIALIZED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
    
    def start(self):
        """Start all components"""
        logger.info("🚀 STARTING SOLAR PREDICTION SYSTEM")
        
        # Start server
        if not self.server.start():
            logger.error("❌ Failed to start server")
            return
        
        # Connect publisher
        if not self.publisher.connect():
            logger.warning("⚠️ Publisher not connected - continuing without publishing")
        
        # Start performance monitor
        if self.monitor:
            self.monitor.start()
        
        self.running = True
        
        # Print system info
        self._print_system_info()
        
        # Main loop
        try:
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop all components"""
        logger.info("🛑 SHUTTING DOWN SOLAR PREDICTION SYSTEM")
        
        self.running = False
        
        # Stop server
        if self.server:
            self.server.stop()
        
        # Disconnect publisher
        if self.publisher:
            self.publisher.disconnect()
        
        # Stop monitor
        if self.monitor:
            self.monitor.stop()
        
        logger.info("👋 SYSTEM SHUTDOWN COMPLETE")
    
    def _print_system_info(self):
        """Print system information"""
        logger.info("=" * 60)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 60)
        logger.info(f"📡 Server listening on: {Config.REQUEST_TOPIC}")
        logger.info(f"📤 Publishing results to: {Config.RESPONSE_TOPIC}")
        logger.info(f"📤 Also publishing to: {Config.CLIENT_PUBLISH_TOPIC}")
        logger.info(f"🧵 Worker threads: {Config.NUM_WORKER_THREADS}")
        logger.info(f"📊 Performance monitoring: {'Enabled' if Config.ENABLE_PERFORMANCE_MONITORING else 'Disabled'}")
        logger.info(f"💾 DynamoDB logging: {'Enabled' if Config.ENABLE_DYNAMODB else 'Disabled'}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to shutdown")
        logger.info("=" * 60)

# =============================
# UTILITY FUNCTIONS
# =============================
def generate_test_sensor_data() -> List[List[float]]:
    """Generate test sensor data for 24 hours - matching the example"""
    
    # This matches the exact pattern from your example
    data = [
        [0.00, 0.55, 0.85],
        [0.00, 0.54, 0.86],
        [0.00, 0.53, 0.87],
        [0.00, 0.52, 0.88],
        [0.00, 0.51, 0.89],
        [0.00, 0.50, 0.90],
        [0.15, 0.55, 0.85],
        [0.30, 0.60, 0.80],
        [0.50, 0.65, 0.75],
        [0.70, 0.70, 0.70],
        [0.90, 0.75, 0.65],
        [1.00, 0.78, 0.60],
        [0.95, 0.77, 0.62],
        [0.80, 0.74, 0.65],
        [0.60, 0.70, 0.70],
        [0.40, 0.65, 0.75],
        [0.20, 0.60, 0.80],
        [0.05, 0.58, 0.82],
        [0.00, 0.55, 0.85],
        [0.00, 0.54, 0.86],
        [0.00, 0.53, 0.87],
        [0.00, 0.52, 0.88],
        [0.00, 0.51, 0.89],
        [0.00, 0.50, 0.90],
    ]
    
    return data

def send_test_prediction_request():
    """Send a test prediction request (for testing)"""
    
    publisher = SolarDataPublisher()
    if publisher.connect():
        # Generate test sensor data matching the example
        sensor_data = generate_test_sensor_data()
        
        # Generate a unique requestId
        request_id = f"test_{int(time.time() * 1000)}"
        
        # Send prediction request with requestId
        request = {
            "requestId": request_id,  # Using requestId format
            "deviceId": "Raspberry_Test",
            "sensor_data": sensor_data
        }
        
        # For testing, we publish to request topic
        client = mqtt.Client(client_id="test_client", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        client.tls_set(
            ca_certs=str(Config.CLIENT_ROOT_CA),
            certfile=str(Config.CLIENT_CERT),
            keyfile=str(Config.CLIENT_KEY),
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )
        
        client.connect(Config.AWS_IOT_ENDPOINT, 8883)
        client.loop_start()
        
        print(f"\n📤 Sending test request with requestId: {request_id}")
        print(f"📤 Using sensor data matching the example pattern")
        result = client.publish(Config.REQUEST_TOPIC, json.dumps(request), qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"✅ Test prediction request sent with requestId: {request_id}")
            print(f"✅ Request sent. Check response on topic: {Config.RESPONSE_TOPIC}")
            print(f"✅ Also check simplified result on: {Config.CLIENT_PUBLISH_TOPIC}")
        else:
            logger.error("❌ Failed to send test request")
        
        time.sleep(2)
        client.loop_stop()
        client.disconnect()
        
        publisher.disconnect()

# =============================
# MAIN ENTRY POINT
# =============================
def main():
    """Main entry point"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Solar Prediction System')
    parser.add_argument('--test', action='store_true', help='Send test prediction request')
    parser.add_argument('--publish-test-data', action='store_true', help='Publish test sensor data')
    parser.add_argument('--no-dynamodb', action='store_true', help='Disable DynamoDB logging')
    args = parser.parse_args()
    
    # Disable DynamoDB if requested
    if args.no_dynamodb:
        Config.ENABLE_DYNAMODB = False
        logger.info("ℹ️ DynamoDB logging disabled by command line")
    
    if args.test:
        # Just send a test request and exit
        send_test_prediction_request()
        return
    
    if args.publish_test_data:
        # Publish test sensor data
        publisher = SolarDataPublisher()
        if publisher.connect():
            test_data = generate_test_sensor_data()
            publisher.publish_sensor_data(test_data, "Raspberry_Test")
            publisher.disconnect()
        return
    
    # Run main application
    app = SolarPredictionApp()
    
    if app.initialize():
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            app.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start application
        app.start()
    else:
        logger.error("❌ Failed to initialize application")
        sys.exit(1)

if __name__ == "__main__":
    main()
