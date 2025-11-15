from flask import Flask, render_template
import threading
from mqtt_handler import main as mqtt_main
from opencv import main as opencv_main

app = Flask(__name__)

def start_background_threads():
    # Create threads
    t1 = threading.Thread(target=mqtt_main, args=("Thread-1",), daemon=True)
    t2 = threading.Thread(target=opencv_main, args=("Thread-2",), daemon=True)

    # Start them
    t1.start()
    t2.start()
start_background_threads() #runs it as soon as imported


@app.route('/login', methods=['GET'])
def login():
    return render_template('main.html')

if __name__ == "__main__":
    # Start threads even when using `python app.py`
    start_background_threads()
    app.run(debug=True)
