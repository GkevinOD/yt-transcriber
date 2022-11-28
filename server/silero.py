import torch
import warnings
import config

warnings.filterwarnings("ignore")

class VAD:
    def __init__(self):
        self.model = init_jit_model(config.SILERO_JIT_PATH)

    def is_silent(self, audio, threshold=0.5):
        return _is_silent(torch.Tensor(audio), self.model, threshold=threshold)

def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def _is_silent(  audio: torch.Tensor,
                model,
                threshold: float = 0.5,
                min_speech_duration_ms: int = 75,
                min_silence_duration_ms: int = 100,
                sampling_rate: int = 16000,
                window_size_samples: int = 1536):

    model.reset_states()
    min_speech_duration = sampling_rate * min_speech_duration_ms / 1000
    min_silence_duration = sampling_rate * min_silence_duration_ms / 1000

    audio_length_samples = len(audio)

    speech_start = -1
    speech_end = -1

    silence_start = -1
    silence_end = -1
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = model(chunk, sampling_rate).item()

        if speech_start == -1:
            if speech_prob >= threshold:
                silence_start = -1
                speech_start = current_start_sample
        else:
            speech_end = current_start_sample
            if speech_end - speech_start >= min_speech_duration:
                return False

        if silence_start == -1:
            if speech_prob < threshold:
                silence_start = current_start_sample
        else:
            silence_end = current_start_sample
            if silence_end - silence_start >= min_silence_duration:
                speech_start = -1
                speech_end = -1
    return True
