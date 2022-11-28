from flask import Flask, render_template, request, send_from_directory
from flask.json import jsonify
import os
import json
import time
import subprocess
import psutil
import static.py.process_text as pt
import static.py.constants as const
import atexit

def exit_handler():
    if os.path.exists(const.CURRENT_PATH):
        os.remove(const.CURRENT_PATH)

    if os.path.exists(const.SESSION_PATH):
        with open(const.SESSION_PATH, 'r') as json_file:
            data = json.load(json_file)
        is_running = psutil.pid_exists(data.get('pid'))
        if is_running:
            p = psutil.Process(data.get('pid'))
            p.terminate()
        os.remove(const.SESSION_PATH)

    if os.path.exists(const.AUDIO_DIR):
        # delete all files in audio folder
        for file in os.listdir(const.AUDIO_DIR):
            os.remove(os.path.join(const.AUDIO_DIR, file))

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
    dict = pt.get_ichi_moe(text)
    return jsonify(dict)

@app.route('/update', methods = ['POST'])
def update():
    last_time = request.form['text']
    while True:
        try:
            with open(const.CURRENT_PATH, 'r') as json_file:
                data = json.load(json_file)

            if last_time == "" or data.get('time', None) != last_time:
                return jsonify(data)
        except:
            pass

        time.sleep(0.1)

@app.route('/audio/<path:filename>')
def get_audio(filename):
    return send_from_directory(const.AUDIO_DIR, filename)

@app.route('/transcribe', methods = ['POST'])
def transcribe():
    text = request.form['text']
    data = {}
    if os.path.exists(const.SESSION_PATH):
        with open(const.SESSION_PATH, 'r') as json_file:
            data = json.load(json_file)
        if data.get('pid', None) == None or psutil.pid_exists(data.get('pid')) == False:
            os.remove(const.SESSION_PATH)
            return transcribe()

        data['url'] = text
        with open(const.SESSION_PATH, 'w') as outfile:
            json.dump(data, outfile)

        return ('', 204)

    process_args = ['python', const.WHISPER_PROCESS_PATH,
                    '--model',  'large',
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
    with open(const.SESSION_PATH, 'w') as outfile:
        json.dump(dict_out, outfile)

    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)