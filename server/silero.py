import torch
import warnings

warnings.filterwarnings("ignore")

class VAD:
	def __init__(self, path, enabled, sample_size=512):
		self.sample_size = sample_size
		self.model = init_jit_model(path)
		self.enabled = enabled

	def speech_prob(self, audio, sampling_rate: int = 16000, window_size_samples = None):
		if window_size_samples is None:
			window_size_samples = self.sample_size if self.sample_size != 0 else 512

		# Returns highest probability of speech in audio
		# If the threshold is reached, the search is stopped and the highest probability is returned
		if not isinstance(audio, torch.Tensor):
			audio = torch.tensor(audio)

		audio_length_samples = len(audio)
		audio_prob_list = []
		for current_start_sample in range(0, audio_length_samples, window_size_samples):
			chunk = audio[current_start_sample: current_start_sample + window_size_samples]
			if len(chunk) < window_size_samples:
				chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))

			prob = self.model(chunk, sampling_rate).item() if self.enabled else 1.0
			audio_prob_list.append((audio[current_start_sample: current_start_sample + window_size_samples], prob))

		return audio_prob_list

def init_jit_model(model_path: str, device=torch.device('cpu')):
	torch.set_grad_enabled(False)
	model = torch.jit.load(model_path, map_location=device)
	model.eval()
	return model