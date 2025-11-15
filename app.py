from flask import *

app = Flask(__name__)

@app.route('/login', methods=['GET'])
def login():
    return render_template('main.html')