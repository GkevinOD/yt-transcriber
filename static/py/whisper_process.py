import argparse
import sys
import os
import json
import time
from datetime import datetime

import ffmpeg
import numpy as np
import constants as const
import whisper
from whisper.audio import SAMPLE_RATE

def open_stream(stream, preferred_quality):
    import streamlink
    import subprocess
    import threading
    stream_options = streamlink.streams(stream)
    if not stream_options:
        print("No playable streams found on this URL:", stream)
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

    cmd = ['streamlink', stream, option, "-O"]
    streamlink_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        ffmpeg_process = (
            ffmpeg.input("pipe:", loglevel="panic")
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    thread = threading.Thread(target=writer, args=(streamlink_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, streamlink_process

def process_stream(url, process1, process2, n_bytes, model, language, vad, task_list, **decode_options):
    chunk_size_ms = 250
    max_silent_ms = 500 # Duration of silence before forcing a transcription
    min_speech_ms = 2000 # Minimum duration of speech before checking for max silence
    prepend_ms = chunk_size_ms * 2 # Duration of silence to prepend to the next transcription

    try:
        silent_ms = 0

        CHUNK_SIZE = int((chunk_size_ms / 1000) * (SAMPLE_RATE))
        PREPEND_SIZE = int((prepend_ms / 1000) * (SAMPLE_RATE))
        MIN_SPEECH = int((min_speech_ms / 1000) * (SAMPLE_RATE))

        chunk_history = []
        while not session_change(url):
            audio = None # Stores audio data up to interval length
            while process1.poll() is None:
                in_bytes = process1.stdout.read(CHUNK_SIZE * 2) # Factor of 2 comes from reading int16 as bytes
                if not in_bytes: break

                chunk = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
                chunk_history.append(chunk)
                if len(chunk_history) > (max_silent_ms / chunk_size_ms):
                    chunk_history.pop(0)

                # Process audio with vad
                if vad is not None and vad.is_silent(chunk, threshold=0.5 if audio is None else 0.25):
                    if audio is None:
                        silent_ms = 0
                        continue
                    else:
                        silent_ms += chunk_size_ms
                else:
                    silent_ms = 0

                # Append chunk to buffer
                if audio is None and PREPEND_SIZE != 0:
                    audio = np.concatenate(chunk_history[-(PREPEND_SIZE // CHUNK_SIZE):])
                else:
                    audio = np.concatenate([audio, chunk]) if audio is not None else chunk

                if len(audio) >= n_bytes or (silent_ms >= max_silent_ms and len(audio) >= MIN_SPEECH): break

            # Transcribe and/or translate add to buffers
            result_dict = {}
            for task in task_list:
                result = model.transcribe(audio,
                                          language=language,
                                          without_timestamps=True,
                                          task=task,
                                          **decode_options)

                result_text = ""
                for segment in result["segments"]:
                    result_text += clean_text(segment["text"])

                result_dict[task] = result_text

            send(result_dict, audio)
    except Exception as e:
        # print line number and error
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    finally:
        process1.kill()
        if process2:
            process2.kill()

def clean_text(text):
    if text is None:
        return ""

    # remove consecutive characters at the end of the string
    for length in range(1, int(len(text) / 3)):
        substr = text[-length:]
        if text.endswith(substr * 3):
            while text.endswith(substr * 2):
                text = text[:-length]
    text = " ".join(text.split())

    # remove spaces at the beginning and end
    text = text.strip()

    return text

def session_change(url):
    with open(const.SESSION_PATH, 'r') as f:
        session = json.load(f)
        if session.get('url', None) != url:
            return True
    return False

def send(dict, audio = None, max_files = 200):
    if dict.get('transcribe') is None and dict.get('translate') is None: return
    if dict.get('transcribe') == "" and dict.get('translate') == "": return

    if dict.get('time') is None:
        dict['time'] = datetime.now().strftime("%H:%M:%S")

    if audio is not None:
        from scipy.io.wavfile import write as wavwrite
        wavwrite(const.AUDIO_DIR + dict['time'].replace(':', '') + '.wav', SAMPLE_RATE, audio)

        # limit number of files in audio directory
        import glob
        files = glob.glob(const.AUDIO_DIR + '*.wav')
        if len(files) > max_files:
            oldest = min(files, key=os.path.getctime)
            os.remove(oldest)

    with open(const.CURRENT_PATH, 'w') as outfile:
        json.dump(dict, outfile)


def main(model='medium', language=None, interval=6, preferred_quality="worst",
         use_vad=True, task=['translate'], **decode_options):

    n_bytes = interval * SAMPLE_RATE

    print("Loading model...")
    model = whisper.load_model(model)

    from vad import VAD
    vad = VAD() if use_vad else None

    while True:
        if os.path.exists(const.SESSION_PATH):
            with open(const.SESSION_PATH, 'r') as f:
                session = json.load(f)
                current_url = session.get('url', None)

            try:
                process1, process2 = open_stream(current_url, preferred_quality)
                if process1 is None or process2 is None: continue

                process_stream(current_url, process1, process2, n_bytes, model, language, vad, task, **decode_options)
            except:
                pass

        time.sleep(1)

def cli():
    parser = argparse.ArgumentParser(description="Parameters for translator.py")
    parser.add_argument('--model', type=str,
                        choices=['tiny', 'tiny.en', 'small', 'small.en', 'medium', 'medium.en', 'large'],
                        default='small',
                        help='Model to be used for generating audio transcription. Smaller models are faster and use '
                             'less VRAM, but are also less accurate. .en models are more accurate but only work on '
                             'English audio.')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate', 'both'], default='translate',
                        help='Whether to transcribe the audio (keep original language) or translate to English.')
    parser.add_argument('--language', type=str, default='auto',
                        help='Language spoken in the stream. Default option is to auto detect the spoken language. '
                             'See https://github.com/openai/whisper for available languages.')
    parser.add_argument('--interval', type=int, default=5,
                        help='Interval between calls to the language model in seconds.')
    parser.add_argument('--temperature', type=str, default='0,0.25,0.5',
                        help='Temperature to use for the language model. Higher values lead to more creative. Separate by comma for multiple values.')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Number of beams in beam search. Set to 0 to use greedy algorithm instead.')
    parser.add_argument('--best_of', type=int, default=5,
                        help='Number of candidates when sampling with non-zero temperature.')
    parser.add_argument('--preferred_quality', type=str, default='audio_only',
                        help='Preferred stream quality option. "best" and "worst" should always be available. Type '
                             '"streamlink URL" in the console to see quality options for your URL.')
    parser.add_argument('--use_vad', type=bool, default=True,
                        help='Enable voice activity detection with Silero VAD.')

    args = parser.parse_args().__dict__

    if args['model'].endswith('.en'):
        if args['model'] == 'large.en':
            print("English model does not have large model, please choose from {tiny.en, small.en, medium.en}")
            sys.exit(0)
        if args['language'] != 'English' and args['language'] != 'en':
            if args['language'] == 'auto':
                print("Using .en model, setting language from auto to English")
                args['language'] = 'en'
            else:
                print("English model cannot be used to detect non english language, please choose a non .en model")
                sys.exit(0)

    if args['language'] == 'English':
        args['task'] = 'transcribe'
        if args['model'] != 'large' and not args['model'].endswith('.en'):
            print("English language detected, changing model to english model.")
            args['model'] += '.en'

    if args['task'] == 'both':
        args['task'] = ['transcribe', 'translate']
    else:
        args['task'] = [args['task']]

    try:
        args['temperature'] = [float(t) for t in args['temperature'].split(',')]
    except ValueError:
        print("Temperature must be a comma separated list of floats")
        sys.exit(0)

    if args['language'] == 'auto':
        args['language'] = None

    if args['beam_size'] == 0:
        args['beam_size'] = None

    main(**args)

if __name__ == '__main__':
    cli()
