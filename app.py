from flask import Flask, render_template, request, send_from_directory
from flask.json import jsonify
import os, json, time, subprocess, psutil

import server.ichi_moe as process_text
import server.config as config

import atexit
def exit_handler():
    if os.path.exists(config.CURRENT_PATH):
        os.remove(config.CURRENT_PATH)

    if os.path.exists(config.SESSION_PATH):
        with open(config.SESSION_PATH, 'r') as json_file:
            data = json.load(json_file)
        is_running = psutil.pid_exists(data.get('pid'))
        if is_running:
            p = psutil.Process(data.get('pid'))
            p.terminate()
        os.remove(config.SESSION_PATH)

    if os.path.exists(config.AUDIO_DIR):
        # delete all files in audio folder
        for file in os.listdir(config.AUDIO_DIR):
            os.remove(os.path.join(config.AUDIO_DIR, file))

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
            with open(config.CURRENT_PATH, 'r') as json_file:
                data = json.load(json_file)

            if last_time == "" or data.get('time', None) != last_time:
                return jsonify(data)
        except:
            pass

        time.sleep(0.1)

@app.route('/audio/<path:filename>')
def get_audio(filename):
    return send_from_directory(config.AUDIO_DIR, filename)

@app.route('/transcribe', methods = ['POST'])
def transcribe():
    text = request.form['text']
    data = {}
    if os.path.exists(config.SESSION_PATH):
        with open(config.SESSION_PATH, 'r') as json_file:
            data = json.load(json_file)
        if data.get('pid', None) == None or psutil.pid_exists(data.get('pid')) == False:
            os.remove(config.SESSION_PATH)
            return transcribe()

        data['url'] = text
        with open(config.SESSION_PATH, 'w') as outfile:
            json.dump(data, outfile)

        return ('', 204)

    process_args = ['python', config.PROCESS_PATH,
                    '--model',  'medium',
                    '--task',  'both',
                    '--language', 'Japanese',
                    '--temperature', '0',
                    '--beam_size', '3',
                    '--interval',  '6',
                    '--preferred_quality', 'worst',
                    '--use_vad', 'True']
    p = subprocess.Popen(process_args)
    pid = p.pid

    print("Started new session with pid: " + str(pid))
    dict_out = {'pid': pid, 'url': text}
    with open(config.SESSION_PATH, 'w') as outfile:
        json.dump(dict_out, outfile)

    return ('', 204)

if __name__ == '__main__':
    # Create audio_dir and session_dir if they don't exist
    if not os.path.exists(config.AUDIO_DIR):
        os.makedirs(config.AUDIO_DIR)

    if not os.path.exists(config.SESSION_DIR):
        os.makedirs(config.SESSION_DIR)

    app.run(debug=True)