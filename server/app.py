from flask import *
from queue import Queue
import threading
from mqtt_handler import MQTTWorker
from opencv_thread import main as opencv_thread_main
import time
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import cv2 # -- ADDED --: Needed for encoding the frame

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

camera = cv2.VideoCapture(0)

def start_background_threads():
    # Create threads
    mqtt_worker = MQTTWorker()
    t1 = threading.Thread(target=mqtt_worker.main, daemon=True)
    t2 = threading.Thread(target=opencv_thread_main, args=(camera,), daemon=True)

    # Start them
    t1.start()
    t2.start()

# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             # Encode frame as JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             # Yield frame in byte format
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), 
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

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


if __name__ == "__main__":
    start_background_threads()
    socketio.run(app, debug=True, use_reloader=False)