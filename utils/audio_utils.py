import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
import numpy as np
import torch
import tensorflow_hub as hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def get_yamnet_embedding(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform)
        waveform = waveform.mean(dim=0)
        _, embeddings, _ = yamnet_model(waveform.numpy())
        return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"Failed to process {audio_path}: {str(e)}")
        return None




