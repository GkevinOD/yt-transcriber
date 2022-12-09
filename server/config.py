import configparser, os, time, threading

paths = {'process': 'server/process.py',
		 'silero': 'server/models/silero_vad.jit'}

dirs = {'audio': 'server/session/audio/',
		'session': 'server/session/'}

session = ''
settings_modified = None

settings = configparser.ConfigParser()
settings.read('server/settings.ini')

# Reloads settings if they are changed.
def settings_updater(interval=1):
	global settings_modified
	last_modified = os.path.getmtime('server/settings.ini')
	settings_modified = last_modified
	print("Starting settings updater...")
	while True:
		time.sleep(interval)
		try:
			modified = os.path.getmtime('server/settings.ini')
			if modified == 0:
				raise FileNotFoundError('server/settings.ini not found')

			if modified != last_modified:
				settings.read('server/settings.ini')
				last_modified = modified
				settings_modified = modified
		except Exception as e:
			print(f'Error reading settings.ini: {e}')

thread_settings_updater = threading.Thread(target=settings_updater, args=(1, ), daemon=True)
thread_settings_updater.start()