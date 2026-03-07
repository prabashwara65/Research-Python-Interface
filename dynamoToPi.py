#!/usr/bin/env python3
"""
Raspberry Pi - Retrieve userAppVisits data from DynamoDB via IoT Core
Specifically designed for your table structure
"""

import json
import time
from datetime import datetime
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# ================= CONFIGURATION =================
AWS_IOT_ENDPOINT = "your-iot-endpoint.iot.region.amazonaws.com"
CLIENT_ID = "raspberry_pi_1"
REQUEST_TOPIC = "raspberrypi/request"
RESPONSE_TOPIC = "raspberrypi/userappvisits/data"

# Certificate paths
CERT_PATH = "/home/pi/certs/device.pem.crt"
PRIVATE_KEY_PATH = "/home/pi/certs/private.pem.key"
ROOT_CA_PATH = "/home/pi/certs/AmazonRootCA1.pem"

class UserAppVisitsRetriever:
    def __init__(self):
        self.client = None
        self.visit_data = {}
        self.last_update_time = None
        
    def connect_mqtt(self):
        """Connect to AWS IoT Core"""
        self.client = AWSIoTMQTTClient(CLIENT_ID)
        self.client.configureEndpoint(AWS_IOT_ENDPOINT, 8883)
        self.client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        
        # Configure connection
        self.client.configureAutoReconnectBackoffTime(1, 32, 20)
        self.client.configureOfflinePublishQueueing(-1)
        self.client.configureDrainingFrequency(2)
        self.client.configureConnectDisconnectTimeout(10)
        self.client.configureMQTTOperationTimeout(5)
        
        # Connect
        self.client.connect()
        print("✅ Connected to AWS IoT Core")
        
        # Subscribe to response topic
        self.client.subscribe(RESPONSE_TOPIC, 1, self.data_callback)
        print(f"📡 Subscribed to {RESPONSE_TOPIC}")
    
    def data_callback(self, client, userdata, message):
        """Callback when userAppVisits data is received"""
        try:
            payload = json.loads(message.payload)
            print("\n" + "="*50)
            print("📨 Received userAppVisits data from DynamoDB:")
            print("="*50)
            
            # Extract the visit data
            if 'visitData' in payload:
                data = payload['visitData']
                
                # Check if it's a single item or multiple
                if isinstance(data, list):
                    for item in data:
                        self.process_visit_data(item)
                else:
                    self.process_visit_data(data)
            else:
                print("No visitData found in payload")
                print(json.dumps(payload, indent=2))
                
        except Exception as e:
            print(f"❌ Error processing data: {e}")
    
    def process_visit_data(self, data):
        """Process individual user visit data"""
        print(f"\n👤 User: {data.get('uid', 'Unknown')}")
        print("-" * 30)
        
        # Extract all fields
        uid = data.get('uid', 'N/A')
        daily_count = data.get('dailyAppOpenCount', 0)
        first_open = data.get('firstOpenTime', 'N/A')
        last_visit = data.get('lastVisit', 'N/A')
        updated_at = data.get('updatedAt', 'N/A')
        
        # Display the data
        print(f"📱 Daily App Opens: {daily_count}")
        print(f"🕐 First Open: {self.format_time(first_open)}")
        print(f"🕒 Last Visit: {self.format_time(last_visit)}")
        print(f"🔄 Last Updated: {self.format_time(updated_at)}")
        
        # Take actions based on the data
        self.take_actions_based_on_data(uid, daily_count, last_visit)
        
        # Store for later use
        self.visit_data[uid] = {
            'daily_count': daily_count,
            'last_visit': last_visit,
            'updated_at': updated_at
        }
        
        self.last_update_time = datetime.now()
    
    def format_time(self, timestamp):
        """Format ISO timestamp to readable format"""
        if timestamp == 'N/A':
            return timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp
    
    def take_actions_based_on_data(self, uid, daily_count, last_visit):
        """Take actions based on user app visit data"""
        
        # Example 1: If user is very active, optimize system
        if daily_count > 10:
            print(f"⚡ High activity user! Optimizing solar tracking for peak usage")
            self.optimize_for_active_user()
        
        # Example 2: Check if user hasn't visited in a while
        if last_visit != 'N/A':
            try:
                last = datetime.fromisoformat(last_visit.replace('Z', '+00:00'))
                now = datetime.now()
                hours_since = (now - last).total_seconds() / 3600
                
                if hours_since > 24:
                    print(f"😴 User inactive for {hours_since:.1f} hours - entering power save mode")
                    self.enter_power_save_mode()
                elif hours_since < 1:
                    print(f"🟢 User active within last hour - maintaining high performance")
                    self.maintain_performance_mode()
            except:
                pass
        
        # Example 3: Adjust based on daily count for multiple users
        self.adjust_based_on_aggregate_data()
    
    def optimize_for_active_user(self):
        """Optimize system for active user"""
        print("   → Optimizing tracking speed and accuracy")
        # Send command to Arduino for faster tracking
        self.send_arduino_command("high_performance")
    
    def enter_power_save_mode(self):
        """Enter power save mode when user inactive"""
        print("   → Entering power save mode")
        # Reduce tracking frequency, save power
        self.send_arduino_command("power_save")
    
    def maintain_performance_mode(self):
        """Maintain normal performance mode"""
        print("   → Maintaining normal performance")
        self.send_arduino_command("normal_mode")
    
    def adjust_based_on_aggregate_data(self):
        """Make decisions based on all users' data"""
        if not self.visit_data:
            return
        
        total_daily_opens = sum(v['daily_count'] for v in self.visit_data.values())
        active_users = len(self.visit_data)
        
        print(f"\n📊 AGGREGATE STATISTICS:")
        print(f"   Active Users: {active_users}")
        print(f"   Total Daily Opens: {total_daily_opens}")
        
        # If many users are active, ensure system is ready
        if total_daily_opens > 50:
            print("   🔥 Peak usage detected - ensuring maximum performance")
            self.send_arduino_command("peak_performance")
    
    def send_arduino_command(self, mode):
        """Send command to Arduino based on user activity"""
        try:
            import serial
            ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            time.sleep(1)
            
            if mode == "high_performance":
                # Faster tracking, higher update rate
                cmd = "MODE:FAST\n"
            elif mode == "power_save":
                # Slower updates, sleep modes
                cmd = "MODE:POWER_SAVE\n"
            elif mode == "peak_performance":
                # Maximum performance
                cmd = "MODE:PEAK\n"
            else:
                cmd = "MODE:NORMAL\n"
            
            ser.write(cmd.encode())
            ser.close()
            print(f"   ✓ Arduino command sent: {mode}")
        except Exception as e:
            print(f"   ✗ Failed to send Arduino command: {e}")
    
    def request_data(self, uid=None):
        """Request userAppVisits data"""
        if self.client:
            # Create request message
            request = {
                "request_id": str(int(time.time())),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # If specific UID provided, request that user
            if uid:
                request["uid"] = uid
                print(f"📤 Requesting data for user {uid}")
            else:
                print("📤 Requesting all user data")
            
            # Publish request
            self.client.publish(REQUEST_TOPIC, json.dumps(request), 1)
    
    def run(self):
        """Main loop"""
        print("="*50)
        print("🚀 User App Visits Data Retriever")
        print("="*50)
        print(f"📱 Monitoring userAppVisits table")
        print(f"📡 Request Topic: {REQUEST_TOPIC}")
        print(f"📡 Response Topic: {RESPONSE_TOPIC}")
        print("="*50)
        
        # Connect to MQTT
        self.connect_mqtt()
        
        # Request initial data for a specific user
        # Replace with your actual user UID
        specific_uid = "IW0CcUjaCNYgvt2zmTwXSMCl7bE2"
        self.request_data(specific_uid)
        
        # Keep running and request periodically
        try:
            while True:
                time.sleep(300)  # Request every 5 minutes
                self.request_data(specific_uid)
                
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            self.client.disconnect()

if __name__ == "__main__":
    retriever = UserAppVisitsRetriever()
    retriever.run()