import time
import threading
import paho.mqtt.client as mqtt
import requests
import shared_state

BROKER = "localhost"
PORT = 1883
TOPIC = "test"
FLASK_POST_URL = "http://127.0.0.1:5000/mqtt-message"

class MQTTWorker:
    def __init__(self):
        self.client = mqtt.Client()
        self.connected = False

        # Attach event handlers
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        print("MQTT connected with result code", rc)
        self.connected = True
        client.subscribe(TOPIC)

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        # Call Flask server
        # print("Received MQTT msg: " + payload)

        if payload == "inc" and shared_state.flag == 1:                
            try:
                requests.post("http://127.0.0.1:5000/mqtt-message", json={"msg": payload})
            except Exception as e:
                print("Failed to call Flask:", e)
        elif payload == "START":
            print("test")
            shared_state.flag = 1
            shared_state.cur_exercise_flag = 1
            requests.post("http://127.0.0.1:5000/stop_rest", json={"msg": payload})
        elif payload == "END":
            print("end test")
            shared_state.flag = 0
            shared_state.cur_exercise_flag = 0
            try:
                requests.post("http://127.0.0.1:5000/reset")
            except Exception as e:
                print("Failed to reset:", e)

            # Start rest timer
            try:
                requests.post("http://127.0.0.1:5000/start_rest", json={"seconds": 30})
            except Exception as e:
                print("Failed to start rest timer:", e)
        else:
            print("Invalid message: " + payload)
        
    def process_message(self, data):
        print("Processing:", data)
        # Add your processing logic here

    def main(self):
        print("Starting MQTT worker")
        self.client.connect(BROKER, PORT, keepalive=60)

        # Wait until connected
        while not self.connected:
            print("Waiting for MQTT connection...")
            try:
                self.client.loop(timeout=1.0)
            except:
                pass
            time.sleep(1)

        # Poll for messages continuously
        print("MQTT connected, entering message loop")
        while True:
            try:
                self.client.loop(timeout=1.0)
            except Exception as e:
                print("MQTT loop error:", e)
            time.sleep(0.1)
