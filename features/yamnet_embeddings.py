import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import tensorflow_hub as hub

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
def get_yamnet_embedding(audio_path):
    """
    Load a WAV file, resample it to 16kHz if necessary, mix down to mono, and extract an embedding by averaging.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
        # Resample if necessary
        if sr != 16000:
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        waveform_np = waveform.numpy()
        _, embeddings, _ = yamnet_model(waveform_np)

        # Compute various statistics along the time axis (axis=0)
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        min_val = np.min(embeddings, axis=0)
        max_val = np.max(embeddings, axis=0)
        skewness = skew(embeddings, axis=0)
        kurt = kurtosis(embeddings, axis=0)
        return np.concatenate([mean, std, min_val, max_val, skewness, kurt])
    except Exception as e:
        print(f"Failed to process {audio_path}: {e}")
        return None