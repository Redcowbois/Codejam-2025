from flask import *
from queue import Queue
import threading
from mqtt_handler import MQTTWorker
from opencv import main as opencv_main
import time
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt

app = Flask(__name__)
socketio = SocketIO(app)

# Global vars
counter = 0
BROKER_HOST = "localhost"     
BROKER_PORT = 1883
MQTT_TOPIC_ENCOURAGEMENT = "encouragement"

mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER_HOST, BROKER_PORT)
mqtt_client.loop_start() 

def start_background_threads():
    # Create threads
    mqtt_worker = MQTTWorker()
    t1 = threading.Thread(target=mqtt_worker.main, daemon=True)
    t2 = threading.Thread(target=opencv_main, args=("Thread-2",), daemon=True)

    # Start them
    t1.start()
    t2.start()

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
@app.route("/publish", methods=["POST"])
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
    data = request.json
    

if __name__ == "__main__":
    start_background_threads()
    socketio.run(app, debug=True, use_reloader=False)
