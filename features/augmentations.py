import torch
import torchaudio

def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def time_shift(waveform, shift_limit=0.2):
    shift_amt = int(torch.randint(int(waveform.size(1) * shift_limit), (1,)))
    return torch.roll(waveform, shifts=shift_amt)
