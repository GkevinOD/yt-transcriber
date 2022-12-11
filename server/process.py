import sys, os, time, glob, gc
from torch.cuda import empty_cache

import ffmpeg, streamlink, subprocess, threading, queue
from scipy.io.wavfile import write as wavwrite

import numpy as np
import re
SAMPLE_RATE = 16000

def is_valid_url(url: str) -> bool:
	regex = re.compile(
		r'^(?:http|ftp)s?://' # http:// or https://
		r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
		r'localhost|' #localhost...
		r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
		r'(?::\d+)?' # optional port
		r'(?:/?|[/?]\S+)$', re.IGNORECASE)

	return re.match(regex, url) is not None

def open_stream(stream, preferred_quality):
	stream = stream.strip()
	if not is_valid_url(stream):
		print('Invalid URL: ', stream)
		return None, None, None

	try:
		stream_options = streamlink.streams(stream)
		if not stream_options:
			print('No playable streams found on this URL:', stream)
			return None, None, None
	except Exception as e:
		print(f'Error getting stream options: {e}')
		return None, None, None

	option = None
	for quality in [preferred_quality, 'audio_only', 'audio_mp4a', 'audio_opus', 'worst']:
		if quality in stream_options:
			option = quality
			break
	if option is None:
		option = next(iter(stream_options.values()))

	args = ['streamlink', stream, option, '-O']
	streamlink_process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	try:
		ffmpeg_process = (
			ffmpeg.input('pipe:', thread_queue_size=1024, loglevel='panic')
			.output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
			.run_async(pipe_stdin=True, pipe_stdout=True)
		)
	except ffmpeg.Error as e:
		print(f'Failed to load audio: {e}')
		terminate_processes(streamlink_process, ffmpeg_process)
		return None, None, None


	def writer(streamlink_process, ffmpeg_process):
		try:
			while streamlink_process.poll() is None and ffmpeg_process.poll() is None:
				if getattr(writer_thread, "do_run", True) is False: break
				chunk = streamlink_process.stdout.read(1024)
				if chunk: ffmpeg_process.stdin.write(chunk)
		except Exception as e:
			print(f'Error writing from streamlink to ffmpeg: {e}')
		finally:
			terminate_processes(streamlink_process, ffmpeg_process)

	writer_thread = threading.Thread(target=writer, args=(streamlink_process, ffmpeg_process))
	writer_thread.start()
	return ffmpeg_process, streamlink_process, writer_thread

def terminate_processes(*processes):
	for process in processes:
		if process is not None and process.poll() is None:
			process.kill()

def read_ffmpeg(ffmpeg_process, q_audio: queue.Queue, params):
	if params.verbose: print('Starting stream thread.')

	thread = threading.currentThread()
	start_size = 0 # First data is start of the timestamp
	total_size = 0
	while ffmpeg_process is not None and ffmpeg_process.poll() is None:
		if getattr(thread, "do_run", True) is False: break
		try:
			chunk = ffmpeg_process.stdout.read(params.chunk_size * 2)
			if chunk:
				audio = np.frombuffer(chunk, np.int16).flatten().astype(np.float32) / 32768.0
				if start_size == 0: start_size = len(audio)
				total_size += len(audio)
				timestamp = (total_size - start_size) / SAMPLE_RATE
				q_audio.put((audio, timestamp))
			else:
				raise(Exception('No audio data received from ffmpeg.'))
		except Exception as e:
			print(f'Error reading ffmpeg: {e}')
			break

	if params.verbose: print('Stream thread ended.')

def set_speech_prob(q_audio: queue.Queue, q_processed: queue.Queue, params):
	if params.verbose: print('Starting silero thread.')
	sys.path.append(os.path.dirname(os.path.abspath(__file__)))
	thread = threading.currentThread()

	import silero

	# If vad is disabled, all prob will be 1.0
	silero_model = silero.VAD(params.silero_jit_path, params.vad_enabled, params.sample_size)
	while getattr(thread, "do_run", True):
		if q_audio.empty():
			time.sleep(0.05)
			continue

		raw_audio, timestamp = q_audio.get()
		audio_prob_list = silero_model.speech_prob(raw_audio)
		last_length = 0
		# Fixed size in silero.py set to 512. Roughly 32ms for each chunk.
		while len(audio_prob_list) > 0:
			audio, prob = audio_prob_list.pop(0)
			timestamp += last_length
			last_length = len(audio) / SAMPLE_RATE
			q_processed.put((audio, prob, timestamp))

	if params.verbose: print('Silero thread ended.')

def filter_speech(q_processed: queue.Queue, q_filtered: queue.Queue, params):
	if params.verbose: print('Starting filter thread.')
	thread = threading.currentThread()

	audio = None
	silent = 0
	speech = 0
	prepend_audio = []
	prepend_audio_size = 0
	start_timestamp = None
	while getattr(thread, "do_run", True):
		if q_processed.empty():
			time.sleep(0.05)
			continue

		chunk, prob, timestamp = q_processed.get()
		chunk_size = len(chunk)

		prepend_audio.append(chunk)
		prepend_audio_size += chunk_size
		while prepend_audio_size > params.prepend_size:
			prepend_audio_size -= len(prepend_audio.pop(0))

		if prob < (params.threshold_low if audio is None else params.threshold_high):
			silent += chunk_size if audio is not None else 0
			if audio is None:
				continue
		else:
			silent = 0
			speech += chunk_size

		if start_timestamp is None:
			start_timestamp = timestamp

		# Append chunk to buffer
		if audio is None and params.prepend_size > 0:
			audio = np.concatenate(prepend_audio)
			prepend_audio_size = sum([len(a) for a in prepend_audio[:-1]])
			start_timestamp -= prepend_audio_size / SAMPLE_RATE
		else:
			audio = np.concatenate([audio, chunk]) if audio is not None else chunk

		if len(audio) >= params.max_length + params.prepend_size or silent >= params.max_silent:
			speech_prob = speech / len(audio)
			if speech_prob >= params.min_speech:
				append_audio_size = 0
				while append_audio_size < params.append_size:
					chunk, prob, timestamp = q_processed.get()
					chunk_size = len(chunk)

					prepend_audio.append(chunk)
					prepend_audio_size += chunk_size
					while prepend_audio_size > params.prepend_size:
						prepend_audio_size -= len(prepend_audio.pop(0))

					audio = np.concatenate([audio, chunk])
					append_audio_size += chunk_size
				q_filtered.put((audio, start_timestamp, speech_prob))

			audio = None
			silent = 0
			speech = 0
			start_timestamp = None

	if params.verbose: print('Filter thread ended.')

def process_audio(q_filtered: queue.Queue, socketio_emit: callable, params):
	if params.verbose:
		print('Starting whisper thread.')

	import whisper
	whisper_model = whisper.load_model(params.model)

	# Used for whisper model prompt context
	prompt_buffer = {
		'transcribe': {'text': [], 'seconds': []},
		'translate': {'text': [], 'seconds': []},
	}

	# Used for fixing prefix overlap
	previous_result = None

	thread = threading.currentThread()
	while getattr(thread, "do_run", True):
		if q_filtered.empty():
			time.sleep(0.05)
			continue

		# Empty the queue to catch up to livestream
		if params.max_buffer > 0 and q_filtered.qsize() > params.max_buffer:
			if params.verbose:
				print(f'Clearing filtered buffer: {q_filtered.qsize()}')
			with q_filtered.mutex:
				q_filtered.queue.clear()

			prompt_buffer = {
				'transcribe': {'text': [], 'seconds': []},
				'translate': {'text': [], 'seconds': []},
			}
			continue

		audio, timestamp, speech_prob = q_filtered.get()

		# Parse the temperature parameter
		temperature = [float(x) for x in params.temperature.split(',')]

		# Parse the tasks parameter
		if params.task == 'both':
			tasks = ['transcribe', 'translate']
		else:
			tasks = [params.task]

		# Parse the language parameter
		language = params.language
		if language.lower() == 'auto' or language == '':
			language = None

		# Format the time: HH:MM:SS,SSS
		time_formatted = time.strftime('%H:%M:%S', time.gmtime(timestamp)) + ',' + str(int(timestamp * 1000) % 1000).zfill(3)

		# Create a results dictionary
		results = {
			'time': time_formatted,
			'timestamp': round(timestamp, 4),
			'length': len(audio) / SAMPLE_RATE,
			'speech': round(speech_prob, 4),
			'whisper': time.time(),
			'buffer': 0
		}

		# Calculate the time gap between the current and previous results
		time_gap = timestamp - previous_result['timestamp'] - previous_result['length'] if previous_result is not None else 0

		for task in tasks:
			# Clear the prompt buffer if the time gap is too large
			if params.prompt_history > 0 and time_gap >= params.prompt_max_gap:
				prompt_buffer[task] = {'text': [], 'seconds': []}

			attempts = 3
			while attempts > 0:
				# Transcribe the audio using the whisper model
				result = whisper_model.transcribe(
					audio,
					prompt=prompt_buffer[task],
					language=language,
					task=task,
					temperature=temperature,
					beam_size=params.beam_size,
					best_of=params.best_of,
					without_timestamps=True
				)

				# Concatenate the result text segments
				result_text = ' '.join(clean_text(segment['text']) for segment in result['segments'])
				result_text = result_text.replace('  ', ' ')
				result_text = result_text.strip()

				# Remove the first 0.05 seconds of audio to try to get possible input
				if not result_text or result_text.isspace():
					attempts -= 1
					audio = audio[int(SAMPLE_RATE * 0.05):]
				else:
					break

			# Update the results dictionary with the transcribed text
			results[task] = result_text

			# Update the prompt buffer
			prompt_buffer[task]['text'].append(result_text)
			prompt_buffer[task]['seconds'].append(len(audio) / SAMPLE_RATE)
			while sum(prompt_buffer[task]['seconds']) > params.prompt_history:
				prompt_buffer[task]['text'].pop(0)
				prompt_buffer[task]['seconds'].pop(0)

		results['whisper'] = round(time.time() - results['whisper'], 4)
		results['buffer'] = q_filtered.qsize()

		if results.get('translate', '') != '' or results.get('transcribe', '') != '':
			previous_result = results
			if params.verbose: print(results)
			socketio_emit('update', results)

	if params.verbose: print('Whisper thread ended.')

def clean_text(text: str):
	if text is None:
		return ''

	# Remove consecutive characters at the end of the string
	for length in range(1, int(len(text) / 3)):
		substr = text[-length:]
		if text.endswith(substr * 3):
			while text.endswith(substr * 2):
				text = text[:-length]
	text = ' '.join(text.split())

	return text.strip()

def clean_threads(*threads: threading.Thread):
	for thread in threads:
		if thread is not None and thread.is_alive():
			thread.do_run = False
			thread.join()

def main(config, socketio_emit: callable, verbose: bool = False):
	q_audio = queue.Queue()
	q_processed = queue.Queue()
	q_filtered = queue.Queue()
	class Params:
		def __init__(self):
			self.append_ms = config.settings.getint('process', 'append_ms', fallback=250)
			self.beam_size = config.settings.getint('whisper', 'beam_size', fallback=1)
			self.best_of = config.settings.getint('whisper', 'best_of', fallback=1)
			self.chunk_size_ms = config.settings.getint('process', 'chunk_size_ms', fallback=250)
			self.interval = config.settings.getint('whisper', 'interval', fallback=7)
			self.language = config.settings.get('whisper', 'language', fallback='auto')
			self.max_buffer = config.settings.getint('whisper', 'max_buffer', fallback=5)
			self.max_silent_ms = config.settings.getint('process', 'max_silent_ms', fallback=750)
			self.min_speech = config.settings.getfloat('process', 'min_speech', fallback=0.5)
			self.model = config.settings.get('whisper', 'model', fallback='medium')
			self.preferred_quality = config.settings.get('whisper', 'preferred_quality', fallback='worst')
			self.prepend_ms = config.settings.getint('process', 'prepend_ms', fallback=500)
			self.prompt_history = config.settings.getint('whisper', 'prompt_history', fallback=3)
			self.prompt_max_gap = config.settings.getint('whisper', 'prompt_max_gap', fallback=5)
			self.sample_size_ms = config.settings.getint('silero', 'sample_size_ms', fallback=32)
			self.task = config.settings.get('whisper', 'task', fallback='transcribe')
			self.temperature = config.settings.get('whisper', 'temperature', fallback='0')
			self.threshold = config.settings.getfloat('silero', 'threshold_low', fallback=0.5)
			self.threshold_high = config.settings.getfloat('silero', 'threshold_high', fallback=0.8)
			self.vad_enabled = config.settings.getboolean('process', 'use_vad', fallback=True)

			self.append_size = int(self.append_ms / 1000 * SAMPLE_RATE)
			self.chunk_size = int(self.chunk_size_ms / 1000 * SAMPLE_RATE)
			self.max_length = self.interval * SAMPLE_RATE
			self.max_silent = int(self.max_silent_ms / 1000 * SAMPLE_RATE)
			self.prepend_size = int(self.prepend_ms / 1000 * SAMPLE_RATE)
			self.sample_size = int(self.sample_size_ms / 1000 * SAMPLE_RATE)
			self.silero_jit_path = os.path.dirname(os.path.abspath(__file__)) + '/models/silero_vad.jit'
			self.threshold_low = self.threshold
			self.verbose = verbose
	params = Params()

	last_session = ''
	last_settings_modified = config.settings_modified

	ffmpeg_process = None
	streamlink_process = None
	writer_thread = None

	thread_read_ffmpeg = None
	thread_set_speech_prob = None
	thread_filter_speech = None
	thread_process_audio = None

	change_stream = True
	change_model = True
	change_silero = True
	change_filter = True

	while getattr(threading.currentThread(), "do_run", True):
		time.sleep(1)
		if config.session == '':
			continue

		if config.session != last_session:
			if verbose: print(f'Session url has changed: {config.session}')

			last_session = config.session
			change_stream = True

		if last_settings_modified != config.settings_modified:
			if verbose: print('Change detected in settings.')
			# Keep track of settings that require some sort of restart.
			temp_model = params.model
			temp_preferred_quality = params.preferred_quality
			temp_use_vad = params.vad_enabled
			temp_sample_size_ms = params.sample_size_ms

			params.__init__()

			if temp_model != params.model: change_model = True
			if temp_preferred_quality != params.preferred_quality: change_stream = True
			if temp_use_vad != params.vad_enabled: change_stream = True
			if temp_sample_size_ms != params.sample_size_ms: change_silero = True

			last_settings_modified = config.settings_modified

		if change_stream:
			terminate_processes(ffmpeg_process, streamlink_process)
			clean_threads(thread_read_ffmpeg, writer_thread)

			ffmpeg_process, streamlink_process, writer_thread = open_stream(config.session, config.settings['whisper']['preferred_quality'])
			thread_read_ffmpeg = threading.Thread(target=read_ffmpeg, args=(ffmpeg_process, q_audio, params))
			thread_read_ffmpeg.start()

			change_stream = False

		if change_filter:
			clean_threads(thread_filter_speech)

			thread_filter_speech = threading.Thread(target=filter_speech, args=(q_processed, q_filtered, params))
			thread_filter_speech.start()

			change_filter = False

		if change_silero:
			clean_threads(thread_set_speech_prob)

			thread_set_speech_prob = threading.Thread(target=set_speech_prob, args=(q_audio, q_processed, params))
			thread_set_speech_prob.start()

			change_silero = False

		if change_model:
			clean_threads(thread_process_audio)

			gc.collect()
			empty_cache()

			thread_process_audio = threading.Thread(target=process_audio, args=(q_filtered, socketio_emit, params))
			thread_process_audio.start()

			change_model = False

		if not thread_read_ffmpeg.is_alive() or not thread_set_speech_prob.is_alive() or not thread_filter_speech.is_alive() or not thread_process_audio.is_alive():
			if verbose: print('One of the threads has terminated. Stopping session...')
			terminate_processes(ffmpeg_process, streamlink_process)
			clean_threads(thread_read_ffmpeg, writer_thread, thread_set_speech_prob, thread_filter_speech, thread_process_audio)
			config.session = ''
			change_model = True
			change_silero = True
			change_filter = True
			change_stream = True

	terminate_processes(ffmpeg_process, streamlink_process)
	clean_threads(thread_read_ffmpeg, thread_set_speech_prob, thread_filter_speech, thread_process_audio, writer_thread)

if __name__ == '__main__':
	main(None, None, False)