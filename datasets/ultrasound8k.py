import os
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from features.extractors import extract_mel_spectrogram
import soundata

class UrbanSoundDataset(Dataset):
    def __init__(self, metadata_csv, audio_dir):
        self.meta = pd.read_csv(metadata_csv)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        path = os.path.join(self.audio_dir, f"fold{row['fold']}", row['slice_file_name'])
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        features = extract_mel_spectrogram(waveform, 16000)
        return features, 0 
    
    def download_urbansound8k():
        """
        Download the UrbanSound8K dataset.
        """
        # Initialize the dataset
        dataset = soundata.initialize('urbansound8k')
        
        # Download the dataset
        dataset.download()
        
        # Validate that all the expected files are there
        dataset.validate()
        
        # Choose a random example clip
        example_clip = dataset.choice_clip()
        
        # Print the available data for the example clip
        print(example_clip)