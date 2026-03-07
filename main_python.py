#!/usr/bin/env python3
"""
Main IoT Core Application - Loads all models and processes AWS IoT Core requests
"""

import json
import logging
import os
import sys
import threading
import queue
import time
from datetime import datetime
from typing import Dict, Any, Optional
import signal
from pathlib import Path

# AWS IoT Core imports
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Model imports (adjust these based on your actual models)
try:
    import tensorflow as tf
    import joblib
    import torch
    import numpy as np
    import pandas as pd
except ImportError as e:
    logging.error(f"Failed to import ML libraries: {e}")
    logging.error("Please install required packages: pip install tensorflow joblib torch numpy pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iot_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IoTModelProcessor:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the IoT Model Processor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.models = {}
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.processing_threads = []
        
        # AWS IoT Core client
        self.mqtt_client = None
        
        # Setup paths
        self.models_path = Path(self.config.get("models_path", "./models"))
        self.data_path = Path(self.config.get("data_path", "./data"))
        
        # Create directories if they don't exist
        self.models_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "aws_iot": {
                "endpoint": "YOUR_AWS_IOT_ENDPOINT",
                "port": 8883,
                "root_ca": "./certs/root-CA.crt",
                "private_key": "./certs/private.pem.key",
                "certificate": "./certs/certificate.pem.crt",
                "client_id": "iot_model_processor",
                "subscribe_topic": "iot/requests/#",
                "publish_topic": "iot/responses"
            },
            "models_path": "./models",
            "data_path": "./data",
            "num_processing_threads": 4,
            "model_timeout": 30,
            "enable_tensorflow_gpu": True
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Loaded configuration from {config_path}")
            else:
                # Save default config
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default configuration at {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def load_all_models(self):
        """Load all models from the models directory"""
        logger.info("Starting to load all models...")
        
        # Configure TensorFlow GPU if enabled
        if self.config.get("enable_tensorflow_gpu", True):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow GPU configured: {len(gpus)} GPU(s) available")
                except RuntimeError as e:
                    logger.error(f"Error configuring GPU: {e}")
        
        # Model loading functions based on file extensions
        model_loaders = {
            '.h5': self.load_tensorflow_model,
            '.pkl': self.load_sklearn_model,
            '.joblib': self.load_joblib_model,
            '.pt': self.load_pytorch_model,
            '.pth': self.load_pytorch_model,
            '.onnx': self.load_onnx_model,
        }
        
        # Scan models directory
        for model_file in self.models_path.glob("*"):
            if model_file.is_file():
                extension = model_file.suffix.lower()
                if extension in model_loaders:
                    try:
                        logger.info(f"Loading model: {model_file.name}")
                        model_name = model_file.stem
                        
                        # Load model using appropriate loader
                        model = model_loaders[extension](model_file)
                        
                        if model is not None:
                            self.models[model_name] = {
                                'model': model,
                                'type': extension,
                                'path': str(model_file),
                                'loaded_at': datetime.now().isoformat()
                            }
                            logger.info(f"✓ Successfully loaded model: {model_name}")
                        else:
                            logger.error(f"✗ Failed to load model: {model_file.name}")
                    
                    except Exception as e:
                        logger.error(f"✗ Error loading model {model_file.name}: {e}")
                else:
                    logger.warning(f"Skipping unsupported file: {model_file.name}")
        
        logger.info(f"Model loading complete. Loaded {len(self.models)} models.")
        
        # Print loaded models summary
        if self.models:
            logger.info("Loaded models:")
            for name, info in self.models.items():
                logger.info(f"  - {name} ({info['type']})")
        else:
            logger.warning("No models were loaded!")
    
    def load_tensorflow_model(self, model_path: Path):
        """Load TensorFlow/Keras model"""
        try:
            model = tf.keras.models.load_model(str(model_path))
            logger.info(f"TensorFlow model loaded: {model_path.name}")
            return model
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            return None
    
    def load_sklearn_model(self, model_path: Path):
        """Load scikit-learn model (using joblib)"""
        try:
            model = joblib.load(str(model_path))
            logger.info(f"Scikit-learn model loaded: {model_path.name}")
            return model
        except Exception as e:
            logger.error(f"Error loading scikit-learn model: {e}")
            return None
    
    def load_joblib_model(self, model_path: Path):
        """Load joblib model"""
        try:
            model = joblib.load(str(model_path))
            logger.info(f"Joblib model loaded: {model_path.name}")
            return model
        except Exception as e:
            logger.error(f"Error loading joblib model: {e}")
            return None
    
    def load_pytorch_model(self, model_path: Path):
        """Load PyTorch model"""
        try:
            # You'll need to define your model architecture before loading
            # This is a simplified example
            model = torch.load(str(model_path), map_location='cpu')
            model.eval()
            logger.info(f"PyTorch model loaded: {model_path.name}")
            return model
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return None
    
    def load_onnx_model(self, model_path: Path):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            model = ort.InferenceSession(str(model_path))
            logger.info(f"ONNX model loaded: {model_path.name}")
            return model
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            return None
    
    def setup_aws_iot(self):
        """Setup AWS IoT Core MQTT client"""
        aws_config = self.config.get("aws_iot", {})
        
        try:
            # Create MQTT client
            self.mqtt_client = AWSIoTMQTTClient(aws_config.get("client_id"))
            self.mqtt_client.configureEndpoint(
                aws_config.get("endpoint"),
                aws_config.get("port", 8883)
            )
            self.mqtt_client.configureCredentials(
                aws_config.get("root_ca"),
                aws_config.get("private_key"),
                aws_config.get("certificate")
            )
            
            # Configure MQTT client
            self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
            self.mqtt_client.configureOfflinePublishQueueing(-1)  # Infinite queue
            self.mqtt_client.configureDrainingFrequency(2)  # 2 Hz
            self.mqtt_client.configureConnectDisconnectTimeout(10)
            self.mqtt_client.configureMQTTOperationTimeout(5)
            
            # Connect and subscribe
            self.mqtt_client.connect()
            logger.info("Connected to AWS IoT Core")
            
            # Subscribe to topics
            subscribe_topic = aws_config.get("subscribe_topic")
            self.mqtt_client.subscribe(subscribe_topic, 1, self.mqtt_message_callback)
            logger.info(f"Subscribed to topic: {subscribe_topic}")
            
        except Exception as e:
            logger.error(f"Error setting up AWS IoT: {e}")
            raise
    
    def mqtt_message_callback(self, client, userdata, message):
        """Callback for incoming MQTT messages"""
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            logger.info(f"Received message on topic {message.topic}")
            
            # Add to processing queue
            self.request_queue.put({
                'topic': message.topic,
                'payload': payload,
                'timestamp': time.time()
            })
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON payload: {message.payload}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def process_request(self, request):
        """Process a single request with appropriate model"""
        try:
            payload = request['payload']
            
            # Extract request details
            model_name = payload.get('model')
            input_data = payload.get('data')
            request_id = payload.get('request_id', str(time.time()))
            
            if not model_name:
                return {
                    'request_id': request_id,
                    'status': 'error',
                    'message': 'No model specified'
                }
            
            if model_name not in self.models:
                return {
                    'request_id': request_id,
                    'status': 'error',
                    'message': f'Model {model_name} not found',
                    'available_models': list(self.models.keys())
                }
            
            # Get model info
            model_info = self.models[model_name]
            model = model_info['model']
            model_type = model_info['type']
            
            logger.info(f"Processing request {request_id} with model {model_name}")
            
            # Process based on model type
            if model_type in ['.h5']:  # TensorFlow
                result = self.process_tensorflow(model, input_data)
            elif model_type in ['.pkl', '.joblib']:  # sklearn/joblib
                result = self.process_sklearn(model, input_data)
            elif model_type in ['.pt', '.pth']:  # PyTorch
                result = self.process_pytorch(model, input_data)
            elif model_type == '.onnx':  # ONNX
                result = self.process_onnx(model, input_data)
            else:
                result = {'error': f'Unsupported model type: {model_type}'}
            
            # Prepare response
            response = {
                'request_id': request_id,
                'model': model_name,
                'status': 'success',
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Request {request_id} processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                'request_id': payload.get('request_id', 'unknown'),
                'status': 'error',
                'message': str(e)
            }
    
    def process_tensorflow(self, model, data):
        """Process with TensorFlow model"""
        # Convert input data to tensor
        input_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        
        # Add batch dimension if needed
        if len(input_tensor.shape) == 1:
            input_tensor = tf.expand_dims(input_tensor, 0)
        
        # Run inference
        predictions = model.predict(input_tensor)
        
        # Convert to list for JSON serialization
        return predictions.tolist()
    
    def process_sklearn(self, model, data):
        """Process with scikit-learn model"""
        # Convert to numpy array
        input_array = np.array(data)
        
        # Run prediction
        predictions = model.predict(input_array)
        
        return predictions.tolist()
    
    def process_pytorch(self, model, data):
        """Process with PyTorch model"""
        # Convert to tensor
        input_tensor = torch.tensor(data, dtype=torch.float32)
        
        # Run inference
        with torch.no_grad():
            predictions = model(input_tensor)
        
        return predictions.tolist()
    
    def process_onnx(self, model, data):
        """Process with ONNX model"""
        # Prepare input
        input_name = model.get_inputs()[0].name
        input_data = {input_name: np.array(data, dtype=np.float32)}
        
        # Run inference
        predictions = model.run(None, input_data)
        
        return predictions[0].tolist()
    
    def worker_thread(self):
        """Worker thread for processing requests"""
        while self.running:
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=1)
                
                # Process request
                response = self.process_request(request)
                
                # Send response via MQTT
                if self.mqtt_client and response:
                    publish_topic = self.config["aws_iot"]["publish_topic"]
                    self.mqtt_client.publish(
                        publish_topic,
                        json.dumps(response),
                        1
                    )
                    logger.info(f"Response sent for request {response['request_id']}")
                
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    def start(self):
        """Start the IoT Model Processor"""
        logger.info("Starting IoT Model Processor...")
        
        # Load all models
        self.load_all_models()
        
        # Setup AWS IoT
        try:
            self.setup_aws_iot()
        except Exception as e:
            logger.error(f"Failed to setup AWS IoT: {e}")
            logger.warning("Continuing without AWS IoT (local mode)")
        
        # Start processing threads
        self.running = True
        num_threads = self.config.get("num_processing_threads", 4)
        
        for i in range(num_threads):
            thread = threading.Thread(target=self.worker_thread, name=f"Worker-{i}")
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
            logger.info(f"Started worker thread: Worker-{i}")
        
        logger.info("IoT Model Processor started successfully")
        logger.info(f"Waiting for requests... (Press Ctrl+C to stop)")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the IoT Model Processor"""
        logger.info("Stopping IoT Model Processor...")
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5)
        
        # Disconnect MQTT
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
                logger.info("Disconnected from AWS IoT Core")
            except:
                pass
        
        logger.info("IoT Model Processor stopped")

def setup_signal_handlers(processor):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        processor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════╗
    ║     IoT Core Model Processor - v1.0          ║
    ║     Loading all models and processing         ║
    ║     requests from AWS IoT Core                ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # Get config path from command line or environment
    config_path = os.environ.get('CONFIG_PATH', 'config.json')
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create processor instance
    processor = IoTModelProcessor(config_path)
    
    # Setup signal handlers
    setup_signal_handlers(processor)
    
    # Start processor
    try:
        processor.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()