from flask import *
from queue import Queue
import threading
from mqtt_handler import MQTTWorker
import time
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import cv2 # -- ADDED --: Needed for encoding the frame
import mediapipe as mp
import os
import numpy as np
import requests
import tensorflow as tf
from opencv import opencv_run
import shared_state

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
# Global vars
counter = 0
BROKER_HOST = "10.74.242.184"  
BROKER_PORT = 1883
MQTT_TOPIC_ENCOURAGEMENT = "encouragement"
mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER_HOST, BROKER_PORT)
mqtt_client.loop_start() 


def start_background_threads():
    # Create threads
    mqtt_worker = MQTTWorker()
    t1 = threading.Thread(target=mqtt_worker.main, daemon=True)
    t2 = threading.Thread(target=opencv_run, args=(socketio,), daemon=True)
    # Start them
    t1.start()
    t2.start()

def generate_frames():
    while True:
        if shared_state.latest_jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   shared_state.latest_jpeg +
                   b"\r\n")
        time.sleep(0.01)
    




@app.route("/")
def index():
    return render_template("index.html", number=counter)

@app.route("/test")
def test():
    return render_template("test.html")

@app.route("/counter")
def get_counter():
    return jsonify(counter=counter)

# Route that mqtt thread calls to update counter
@app.route("/mqtt-message", methods=["POST"])
def mqtt_message():
    global counter
    counter += 1
    socketio.emit("counter", counter)
    return {"status": "ok"}

# publish mqtt message 
@app.route("/encourage", methods=["POST"])
def publish_message():
    msg = "encouragement"
    mqtt_client.publish(MQTT_TOPIC_ENCOURAGEMENT, msg)

    return jsonify({
        "status": "published",
        "topic": MQTT_TOPIC_ENCOURAGEMENT,
        "message": msg
    })


@app.route("/opencv", methods=["POST"])
def open_cv():
    payload = request.get_json()  # parses JSON automatically
    text = payload.get("type", "Unknown")
    print(text)
    socketio.emit("type", text)  # broadcast to all clients
    return {"status": "type_change"}

    
@app.route("/reset", methods=["POST"])
def reset():
    global counter
    counter = 0
    socketio.emit("counter", 0)
    return {"status": "reset"}

@app.route("/start_rest", methods=["POST"])
def start_rest():
    seconds = int(request.args.get("seconds", 60))
    socketio.emit("start_rest_timer_client", {"duration": seconds})
    return jsonify({"status": "rest_timer_initialized", "duration": seconds})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    start_background_threads()
    socketio.run(app, debug=False, use_reloader=False)