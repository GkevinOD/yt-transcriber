from flask import Flask, render_template, request, send_from_directory
from flask.json import jsonify
import os, json, time, subprocess, psutil

import server.ichi_moe as process_text
import server.config as config

import atexit
def exit_handler():
    if os.path.exists(config.paths['current']):
        os.remove(config.paths['current'])

    if os.path.exists(config.paths['session']):
        with open(config.paths['session'], 'r') as json_file:
            data = json.load(json_file)
        is_running = psutil.pid_exists(data.get('pid'))
        if is_running:
            p = psutil.Process(data.get('pid'))
            p.terminate()
        os.remove(config.paths['session'])

    if os.path.exists(config.dirs['audio']):
        # delete all files in audio folder
        for file in os.listdir(config.dirs['audio']):
            os.remove(os.path.join(config.dirs['audio'], file))

atexit.register(exit_handler)

app = Flask(__name__)
app.config.update(
    TESTING=True
)

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
def transcribe():
    text = request.form['text']
    data = {}
    if os.path.exists(config.paths['session']):
        with open(config.paths['session'], 'r') as json_file:
            data = json.load(json_file)
        if data.get('pid', None) == None or psutil.pid_exists(data.get('pid')) == False:
            os.remove(config.paths['session'])
            return transcribe()

        data['url'] = text
        with open(config.paths['session'], 'w') as outfile:
            json.dump(data, outfile)

        return ('', 204)

    process_args = ['python', config.paths['process']]
    p = subprocess.Popen(process_args)
    pid = p.pid

    print("Started new session with pid: " + str(pid))
    dict_out = {'pid': pid, 'url': text}
    with open(config.paths['session'], 'w') as outfile:
        json.dump(dict_out, outfile)

    return ('', 204)

if __name__ == '__main__':
    # Create audio_dir and session_dir if they don't exist
    if not os.path.exists(config.dirs['audio']):
        os.makedirs(config.dirs['audio'])

    if not os.path.exists(config.dirs['session']):
        os.makedirs(config.dirs['session'])

    app.run(debug=True)