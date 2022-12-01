from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from pyngrok import ngrok
from flask.json import jsonify
import os, json, time, subprocess, psutil, threading

import server.ichi_moe as process_text
import server.config as config

app = Flask(__name__)
socketio = SocketIO(app)

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods = ['POST'])
def process():
    text = request.form['text']
    dict = process_text.get_ichi_moe(text)
    return jsonify(dict)

@app.route('/audio/<path:filename>')
def get_audio(filename):
    return send_from_directory(config.dirs['audio'], filename)

@app.route('/transcribe', methods = ['POST'])
def transcribe_route():
    url = request.form['text']
    with open(config.paths['session'], 'w') as outfile:
        json.dump({'url': url}, outfile)
    return ('', 204)

if __name__ == '__main__':
    try:
        # Prepare session and audio directories
        if not os.path.exists(config.dirs['audio']):
            os.makedirs(config.dirs['audio'])

        if not os.path.exists(config.dirs['session']):
            os.makedirs(config.dirs['session'])

        # Start whisper and silero
        from server.process import main
        p_thread = threading.Thread(target=main, args=(socketio, ))
        p_thread.daemon = True
        p_thread.start()

        socketio.run(app)
    finally:
        # Delete session
        if os.path.exists(config.dirs['session']):
            from shutil import rmtree
            rmtree(config.dirs['session'])
