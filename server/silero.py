import torch
import warnings

warnings.filterwarnings("ignore")

class VAD:
    def __init__(self, path):
        self.model = init_jit_model(path)

    def is_silent(self, audio, threshold=0.5):
        prob = speech_prob(torch.Tensor(audio), self.model)
        return prob < threshold

def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def speech_prob(audio: torch.Tensor, model, sampling_rate: int = 16000, window_size_samples: int = 512):
    audio_length_samples = len(audio)

    highest = 0
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))

        prob = model(chunk, sampling_rate).item()
        highest = max(highest, prob)
    return highest