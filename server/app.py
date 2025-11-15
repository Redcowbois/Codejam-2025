from flask import *
from queue import Queue
import threading
from mqtt_handler import MQTTWorker
from opencv import main as opencv_main
import time
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Global vars
counter = 0
event_queue = Queue()

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


if __name__ == "__main__":
    start_background_threads()
    socketio.run(app, debug=True, use_reloader=False)
