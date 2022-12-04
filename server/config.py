import configparser, os, time, threading

paths = {'process': 'server/process.py',
         'silero': 'server/models/silero_vad.jit'}

dirs = {'audio': 'server/session/audio/',
        'session': 'server/session/'}

session = ''

settings = configparser.ConfigParser()
settings.read('server/settings.ini')

# Reloads settings if they are changed. Checks every 5 seconds.
def reload_settings():
    last_modified = os.path.getmtime('server/settings.ini')
    while True:
        time.sleep(1)
        if os.path.getmtime('server/settings.ini') != last_modified:
            settings.read('server/settings.ini')
            last_modified = os.path.getmtime('server/settings.ini')

reload_thread = threading.Thread(target=reload_settings)
reload_thread.daemon = True
reload_thread.start()