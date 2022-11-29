import argparse, sys, os, json, time, glob
from datetime import datetime

import ffmpeg, streamlink, subprocess, threading
from scipy.io.wavfile import write as wavwrite

import numpy as np
import config

import whisper
from whisper.audio import SAMPLE_RATE
from silero import VAD

def open_stream(stream, preferred_quality):
    stream_options = streamlink.streams(stream)
    if not stream_options:
        print('No playable streams found on this URL:', stream)
        return None, None

    option = None
    for quality in [preferred_quality, 'audio_only', 'audio_mp4a', 'audio_opus', 'worst']:
        if quality in stream_options:
            option = quality
            break
    if option is None:
        # Fallback
        option = next(iter(stream_options.values()))

    def writer(streamlink_proc, ffmpeg_proc):
        while (not streamlink_proc.poll()) and (not ffmpeg_proc.poll()):
            try:
                chunk = streamlink_proc.stdout.read(1024)
                ffmpeg_proc.stdin.write(chunk)
            except (BrokenPipeError, OSError):
                pass

    cmd = ['streamlink', stream, option, '-O']
    streamlink_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        ffmpeg_process = (
            ffmpeg.input('pipe:', loglevel='panic')
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f'Failed to load audio: {e.stderr.decode()}') from e

    thread = threading.Thread(target=writer, args=(streamlink_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, streamlink_process

def process_audio(audio, model, task, q_result, id):
    result_dict = q_result[id]
    temperature = [float(x) for x in config.settings['whisper']['temperature'].split(',')]
    result = model.transcribe(audio,
                              language=config.settings['whisper']['language'],
                              task=task,
                              temperature=temperature,
                              without_timestamps=True)
    result_text = ''
    for segment in result['segments']:
        result_text += clean_text(segment['text'])

    result_dict['audio'] = audio
    result_dict[task] = result_text

RETURN_CONTINUE = -1
RETURN_END = 0
RETURN_MODEL_CHANGE = 1
RETURN_SESSION_CHANGE = 2
def process_stream(url, process1, process2, models_whisper, model_silero):
    return_code = RETURN_END

    try:
        silent_ms = 0
        chunk_history = []
        last_model = config.settings['whisper']['model']
        last_num_models = config.settings['whisper']['num_models']

        # Create thread for every whisper model
        threads = [None for _ in range(len(models_whisper))]
        q_result = {} # Saves results
        q_next = 0

        def send_q_result(q_result):
            t = threading.currentThread()
            while getattr(t, "do_run", True):
                if len(q_result) > 0:
                    key, value = next(iter(q_result.items()))
                    if all(x is not None for x in value.values()):
                        print(f'Sending out #{key}...')

                        send(value, value.pop('audio'))
                        q_result.pop(key)
                        time.sleep(0.5)
                time.sleep(0.1)

        thread_send = threading.Thread(target=send_q_result, args=(q_result,))
        thread_send.start()
        while not session_change(url):
            if last_model != config.settings['whisper']['model'] or last_num_models != config.settings['whisper']['num_models']:
                print('Model changed, restarting...')
                for thread in threads:
                    if thread is not None:
                        thread.do_run = False
                        thread.join()
                thread_send.do_run = False
                thread_send.join()
                return_code = RETURN_MODEL_CHANGE
                break

            CHUNK_SIZE = int((config.settings.getint('process', 'chunk_size_ms') / 1000) * SAMPLE_RATE)
            PREPEND_SIZE = int((config.settings.getint('process', 'prepend_ms') / 1000) * SAMPLE_RATE)

            MIN_AUDIO_LENGTH = int((config.settings.getint('process', 'min_speech_ms') / 1000) * SAMPLE_RATE)
            MAX_AUDIO_LENGTH = config.settings.getint('whisper', 'interval') * SAMPLE_RATE

            audio = None # Stores audio data up to interval length
            while process1.poll() is None:
                in_bytes = process1.stdout.read(CHUNK_SIZE * 2) # Factor of 2 comes from reading int16 as bytes
                if not in_bytes: break

                chunk = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
                chunk_history.append(chunk)
                if len(chunk_history) > (config.settings.getint('process', 'max_silent_ms') / config.settings.getint('process', 'chunk_size_ms')):
                    chunk_history.pop(0)

                # Process audio with vad
                threshold_high = config.settings.getfloat('silero', 'threshold_high')
                threshold_low = config.settings.getfloat('silero', 'threshold_low')
                if model_silero is not None and model_silero.is_silent(chunk, threshold=threshold_high if audio is None else threshold_low):
                    if audio is None:
                        silent_ms = 0
                        continue
                    else:
                        silent_ms += config.settings.getint('process', 'chunk_size_ms')
                else:
                    silent_ms = 0

                # Append chunk to buffer
                if audio is None and PREPEND_SIZE != 0:
                    audio = np.concatenate(chunk_history[-(PREPEND_SIZE // CHUNK_SIZE):])
                else:
                    audio = np.concatenate([audio, chunk]) if audio is not None else chunk

                if len(audio) >= MAX_AUDIO_LENGTH or (silent_ms >= config.settings.getint('process', 'max_silent_ms') and len(audio) >= MIN_AUDIO_LENGTH): break

            task_list = [config.settings['whisper']['task']] if config.settings['whisper']['task'] != 'both' else ['transcribe', 'translate']
            result_dict = {}
            for task in task_list:
                result_dict[task] = None
            q_result[q_next] = result_dict

            while len(task_list) > 0:
                for i, thread in enumerate(threads):
                    if thread is None or not thread.is_alive():
                        print(f'Starting {task_list[0]} on thread {i} for #{q_next}')
                        threads[i] = threading.Thread(target=process_audio, args=(audio, models_whisper[i], task_list.pop(0), q_result, q_next))
                        threads[i].start()
                        break
                time.sleep(0.05)
            q_next += 1
    except Exception as e:
        # print line number and error
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    finally:
        process1.kill()
        if process2:
            process2.kill()

    if session_change(url):
        return_code = RETURN_SESSION_CHANGE

    for thread in threads:
        if thread is not None:
            thread.do_run = False
            thread.join()
    thread_send.do_run = False
    thread_send.join()

    return return_code

def clean_text(text):
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

def session_change(url):
    with open(config.paths['session'], 'r') as f:
        session = json.load(f)
        if session.get('url', None) != url:
            return True
    return False

def send(dict, audio = None, max_files = 200):
    if dict.get('transcribe') is None and dict.get('translate') is None: return
    if dict.get('transcribe') == '' and dict.get('translate') == '': return

    if dict.get('time') is None:
        dict['time'] = datetime.now().strftime('%H:%M:%S')

    if audio is not None:
        wavwrite(config.dirs['audio'] + dict['time'].replace(':', '') + '.wav', SAMPLE_RATE, audio)

        # limit number of files in audio directory
        files = glob.glob(config.dirs['audio'] + '*.wav')
        if len(files) > max_files:
            oldest = min(files, key=os.path.getctime)
            os.remove(oldest)

    with open(config.paths['current'], 'w') as outfile:
        json.dump(dict, outfile)


def load_whisper_models(model_size='medium', num_models=1):
    models = []
    for i in range(num_models):
        try:
            print(f'Loading whisper model {i + 1} of {num_models}...')
            model = whisper.load_model(model_size)
            models.append(model)
        except Exception as e:
            print(f'Error loading model {i + 1}: {e}')

    return models

def main():
    print('Loading silero model...')
    model_silero = VAD() if config.settings.getboolean('whisper', 'use_vad') else None
    models_whisper = None
    return_code = RETURN_CONTINUE
    while return_code != RETURN_END:
        return_code = RETURN_CONTINUE
        if models_whisper is None:
            models_whisper = load_whisper_models(model_size=config.settings['whisper']['model'], num_models=config.settings.getint('whisper', 'num_models'))

        if len(models_whisper) == config.settings.getint('whisper', 'num_models'):
            if os.path.exists(config.paths['session']):
                with open(config.paths['session'], 'r') as f:
                    session = json.load(f)
                    current_url = session.get('url', None)

                try:
                    process1, process2 = open_stream(current_url, config.settings['whisper']['preferred_quality'])
                    if process1 is None or process2 is None: continue

                    return_code = process_stream(current_url, process1, process2, models_whisper, model_silero)
                except:
                    pass
        else:
            config.settings.set('whisper', 'num_models', str(len(models_whisper)))
            return_code = RETURN_MODEL_CHANGE

        if return_code == RETURN_MODEL_CHANGE:
            for model in models_whisper:
                del model
            models_whisper = None

            time.sleep(5)

            from gc import collect
            collect()

            from torch.cuda import empty_cache
            empty_cache()

        time.sleep(1)

if __name__ == '__main__':
    main()
