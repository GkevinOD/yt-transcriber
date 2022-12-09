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

def set_speech_prob(q_audio: queue.Queue, q_processed: queue.Queue, params):
	sys.path.append(os.path.dirname(os.path.abspath(__file__)))
	thread = threading.currentThread()

	if params.verbose: print('Loading silero model...')
	import silero
	path = params.silero_jit_path

	# If vad is disabled, all prob will be 1.0
	silero_model = silero.VAD(path, params.vad_enabled)
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

def filter_speech(q_processed: queue.Queue, q_filtered: queue.Queue, params):
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

def process_audio(q_filtered: queue.Queue, socketio_emit: callable, params):
	sys.path.append(os.path.dirname(os.path.abspath(__file__)))
	thread = threading.currentThread()

	if params.verbose: print('Loading whisper model...')
	import whisper
	whisper_model = whisper.load_model(params.model)

	results_history = {}
	previous_result = None # Used for prefix
	while getattr(thread, "do_run", True):
		if q_filtered.empty():
			time.sleep(0.05)
			continue

		if params.max_buffer > 0 and q_filtered.qsize() > params.max_buffer:
			# Empty queue
			if params.verbose: print('Buffer is full, emptying queue to catch up...')
			with q_filtered.mutex:
				q_filtered.queue.clear()
			continue

		audio, timestamp, speech_prob = q_filtered.get()

		# Parse parameters
		temperature = [float(x) for x in params.temperature.split(',')]
		tasks = [params.task]
		if params.task == 'both':
			tasks = ['transcribe', 'translate']
		language = params.language
		if language == 'auto' or language == '':
			language = None

		# Time in this format: 00.00.00,000
		time_formatted = time.strftime('%H:%M:%S', time.gmtime(timestamp)) + ',' + str(int(timestamp * 1000) % 1000).zfill(3)
		results = {'time': time_formatted, 'timestamp': round(timestamp, 4), 'length': len(audio) / SAMPLE_RATE, 'prob': round(speech_prob, 4), 'process': time.time(), 'buffer': 0}

		# Check for prefix for time overlaps of 80% the prepend_ms value
		prefix_result = None
		if previous_result is not None:
			time_diff = timestamp - previous_result['timestamp']
			if (previous_result['length']) - time_diff > (0.8 * params.prepend_ms / 1000):
				prefix_result = previous_result

		for task in tasks:
			prompt = results_history.get(task, None)
			if prompt is not None:
				prompt = ' '.join(prompt[-params.prompt_history:])
				prompt.strip()

			attempts = 3
			while attempts > 0:
				result = whisper_model.transcribe(audio,
												  prompt=prompt,
												  language=language,
												  task=task,
												  temperature=temperature,
												  beam_size=params.beam_size,
												  best_of=params.best_of,
												  suppress_tokens="-1",
												  without_timestamps=True)
				result_text = ''
				for segment in result['segments']:
					result_text += clean_text(segment['text']) + ' '

				result_text = result_text.replace('  ', ' ')
				result_text = result_text.strip()

				# Check for prefix due to time overlap
				if prefix_result is not None and prefix_result.get(task, '') != '':
					first = prefix_result.get(task).lower()
					second = result_text.lower()

					if first == second:
						result_text = ''
						break
					else:
						# Remove intersection at end and beginning of string, with 1 character degree of freedom
						for i in range(2):
							a = first[:len(first) - i]
							b = second[i:]
							found = False
							for j in range(0, len(a)):
								if b.startswith(a[-j:]):
									found = True
									if params.verbose: print(f'Found intersect: {result_text} -> {(result_text[i:])[j:]}')
									if len(result_text) - len((result_text[i:])[j:]) > 2: # If changes are too short then it is probably wrong.
										result_text = (result_text[i:])[j:]
							if found: break

					# Remove if all that is left is punctuation
					result_text = result_text.replace('  ', ' ')
					result_text = result_text.strip()
					if len(result_text) <= 2: # Remove string that are too short because they are probably wrong.
						result_text = ''
						break

				if result_text == '':
					attempts -= 1
					audio = audio[int(SAMPLE_RATE * 0.05):]
				else:
					break

			results[task] = result_text
			if params.prompt_history > 0:
				if results_history.get(task) is not None:
					results_history[task].append(result_text)
				else:
					results_history[task] = [result_text]

				if len(results_history[task]) > params.prompt_history:
					results_history[task].pop(0)
		results['process'] = round(time.time() - results['process'], 4)
		results['buffer'] = q_filtered.qsize()

		if results.get('translate', '') != '' or results.get('transcribe', '') != '':
			previous_result = results
			if params.verbose: print(results)
			socketio_emit('update', results)

def clean_text(text: str):
	if text is None:
		return ''

	# remove consecutive characters at the end of the string
	for length in range(1, int(len(text) / 3)):
		substr = text[-length:]
		if text.endswith(substr * 3):
			while text.endswith(substr * 2):
				text = text[:-length]
	text = ' '.join(text.split())

	# remove spaces at the beginning and end
	text = text.strip()

	return text

def clean_threads(*threads):
	for thread in threads:
		if thread is not None and thread.is_alive():
			thread.do_run = False
			thread.join()

def main(config, socketio_emit: callable, verbose: bool = False):
	last_session = ''
	last_settings_modified = config.settings_modified
	ffmpeg_process = streamlink_process = writer_thread = None
	thread_read_ffmpeg = thread_set_speech_prob = thread_filter_speech = thread_process_audio = None
	while getattr(threading.currentThread(), "do_run", True):
		if (config.session != last_session and config.session != '') and (ffmpeg_process is None or ffmpeg_process.poll() is not None):
			last_session = config.session
			if verbose: print(f'Starting session: {config.session}')
			ffmpeg_process, streamlink_process, writer_thread = open_stream(config.session, config.settings['whisper']['preferred_quality'])
			if ffmpeg_process == None or ffmpeg_process.poll() is not None:
				print('Failed to open stream')
				config.session = ''
				continue
		else:
			time.sleep(1)
			continue

		if verbose: print('Starting to process stream...')
		try:
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
					self.silero_jit_path = os.path.dirname(os.path.abspath(__file__)) + '/models/silero_vad.jit'
					self.threshold_low = self.threshold
					self.verbose = verbose
			params = Params()
			if verbose: print(params.__dict__)
			thread_read_ffmpeg = threading.Thread(target=read_ffmpeg, args=(ffmpeg_process, q_audio, params))
			thread_set_speech_prob = threading.Thread(target=set_speech_prob, args=(q_audio, q_processed, params))
			thread_filter_speech = threading.Thread(target=filter_speech, args=(q_processed, q_filtered, params))
			thread_process_audio = threading.Thread(target=process_audio, args=(q_filtered, socketio_emit, params))

			thread_read_ffmpeg.start()
			thread_set_speech_prob.start()
			thread_filter_speech.start()
			thread_process_audio.start()

			# Loop for changes in settings
			while getattr(threading.currentThread(), "do_run", True):
				restart_session = False
				restart_model = False
				if last_settings_modified != config.settings_modified:
					if verbose: print('Change detected in settings.')
					temp_model = params.model # Require model restart
					temp_preferred_quality = params.preferred_quality # Require session restart
					temp_use_vad = params.vad_enabled # Require session restart

					params.__init__()

					if temp_model != params.model: restart_model = True
					if temp_preferred_quality != params.preferred_quality: restart_session = True
					if temp_use_vad != params.vad_enabled: restart_session = True

					last_settings_modified = config.settings_modified

				if config.session != last_session or (restart_session or restart_model):
					if verbose: print(f'Changing session to: {config.session}...')
					q_audio.queue.clear()
					q_processed.queue.clear()
					q_filtered.queue.clear()

					terminate_processes(ffmpeg_process, streamlink_process)
					if writer_thread is not None and writer_thread.is_alive():
						writer_thread.join()

					clean_threads(thread_read_ffmpeg, thread_set_speech_prob, thread_filter_speech)

					ffmpeg_process, streamlink_process, writer_thread = open_stream(config.session, config.settings['whisper']['preferred_quality'])
					if verbose: print('Starting to process stream...')
					thread_read_ffmpeg = threading.Thread(target=read_ffmpeg, args=(ffmpeg_process, q_audio, params), daemon=True)
					thread_set_speech_prob = threading.Thread(target=set_speech_prob, args=(q_audio, q_processed, params), daemon=True)
					thread_filter_speech = threading.Thread(target=filter_speech, args=(q_processed, q_filtered, params), daemon=True)

					thread_read_ffmpeg.start()
					thread_set_speech_prob.start()
					thread_filter_speech.start()

					last_session = config.session

				if restart_model:
					if verbose: print(f'Restarting whisper model: {params.model}...')
					thread_process_audio.do_run = False
					thread_process_audio.join()

					gc.collect()
					empty_cache()

					thread_process_audio = threading.Thread(target=process_audio, args=(q_filtered, socketio_emit, params), daemon=True)
					thread_process_audio.start()

				# Break if any thread is dead
				if not thread_read_ffmpeg.is_alive() or not thread_set_speech_prob.is_alive() or not thread_filter_speech.is_alive() or not thread_process_audio.is_alive():
					if verbose: print('One of the threads has terminated. Clearing session...')
					config.session = ''
					break
				time.sleep(1)
		except Exception as e:
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
		finally:
			if verbose: print('Closing all threads and subprocesses...')
			terminate_processes(ffmpeg_process, streamlink_process)
			clean_threads(thread_read_ffmpeg, thread_set_speech_prob, thread_filter_speech, thread_process_audio, writer_thread)
			if verbose: print(f'Finished processing stream.')

	print('Cleaning up processes and threads...')
	terminate_processes(ffmpeg_process, streamlink_process)
	clean_threads(thread_read_ffmpeg, thread_set_speech_prob, thread_filter_speech, thread_process_audio, writer_thread)

if __name__ == '__main__':
	main(None, None, False)