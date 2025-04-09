import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
from features.extractors import extract_mel_spectrogram
from features.augmentations import add_noise, time_shift

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, transform=True):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".flac"):
                    self.samples.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath = self.samples[idx]
        waveform, sr = torchaudio.load(filepath)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

        if self.transform:
            waveform = add_noise(waveform)
            waveform = time_shift(waveform)

        features = extract_mel_spectrogram(waveform, 16000)
        return features, 1  # Speech label = 1

def get_train_loader(batch_size=16, root_dir="/path/to/librispeech"):
    dataset = LibriSpeechDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)