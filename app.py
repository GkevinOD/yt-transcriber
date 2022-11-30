from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask.json import jsonify
import os, json, time, subprocess, psutil, threading

import server.ichi_moe as process_text
import server.config as config

app = Flask(__name__)
app.config.update(
    TESTING=True
)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods = ['POST'])
def process():
    text = request.form['text']
    dict = process_text.get_ichi_moe(text)
    return jsonify(dict)

@app.route('/update', methods = ['POST'])
def update():
    last_time = request.form['text']
    while True:
        try:
            with open(config.paths['current'], 'r') as json_file:
                data = json.load(json_file)

            if last_time == "" or data.get('time', None) != last_time:
                return jsonify(data)
        except:
            pass

        time.sleep(0.1)

@app.route('/audio/<path:filename>')
def get_audio(filename):
    return send_from_directory(config.dirs['audio'], filename)

@app.route('/transcribe', methods = ['POST'])
def transcribe_route():
    text = request.form['text']
    return transcribe(text)

def transcribe(url):
    data = {}
    if os.path.exists(config.paths['session']):
        with open(config.paths['session'], 'r') as json_file:
            data = json.load(json_file)
        if data.get('pid', None) == None or psutil.pid_exists(data.get('pid')) == False:
            os.remove(config.paths['session'])
            return transcribe()

        data['url'] = url
        with open(config.paths['session'], 'w') as outfile:
            json.dump(data, outfile)

        return ('', 204)

    process_args = ['python', config.paths['process']]
    p = subprocess.Popen(process_args)
    pid = p.pid

    print("Started new session with pid: " + str(pid))
    dict_out = {'pid': pid, 'url': url}
    with open(config.paths['session'], 'w') as outfile:
        json.dump(dict_out, outfile)

    return ('', 204)

def update_thread():
    last_modified = None
    while getattr(threading.currentThread(), "do_run", True):
        try:
            # Check if file has been updated
            modified = os.path.getmtime(config.paths['current'])
            if last_modified == None or modified != last_modified:
                last_modified = modified
                with open(config.paths['current'], 'r') as json_file:
                    data = json.load(json_file)
                socketio.emit('update', data)
        except:
            pass
        time.sleep(0.1)

if __name__ == '__main__':
    try:
        # Prepare session and audio directories
        if not os.path.exists(config.dirs['audio']):
            os.makedirs(config.dirs['audio'])

        if not os.path.exists(config.dirs['session']):
            os.makedirs(config.dirs['session'])

        # Start update thread as daemon
        update_thread = threading.Thread(target=update_thread)
        update_thread.daemon = True
        update_thread.start()

        socketio.run(app, debug=True)
    finally:
        # Delete session
        if os.path.exists(config.dirs['session']):
            from shutil import rmtree
            rmtree(config.dirs['session'])
