import os
import pandas as pd
import random
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm

def mix_audio_clips(base_audio_path, urban_paths, output_path):
    y_base, sr = librosa.load(base_audio_path, sr=None)
    
    for path in urban_paths:
        y_noise, sr_noise = librosa.load(path, sr=sr)
        if len(y_noise) < len(y_base):
            # Pad urban sound
            y_noise = np.pad(y_noise, (0, len(y_base) - len(y_noise)))
        else:
            y_noise = y_noise[:len(y_base)]
        y_base = y_base + 0.5 * y_noise  # mix at lower volume

    sf.write(output_path, y_base, sr)