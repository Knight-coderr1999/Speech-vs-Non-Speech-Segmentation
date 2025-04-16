"""
This script provides functionality for processing and balancing audio datasets. 
It includes methods for mixing audio clips with urban noise and preparing 
balanced datasets for training machine learning models.

Functions:
-----------
1. mix_audio_clips(base_audio_path, urban_paths, output_path):
    - Mixes a base audio clip with multiple urban noise clips.
    - Pads or truncates urban noise clips to match the length of the base audio.
    - Saves the mixed audio to the specified output path.

2. get_audio_events():
    - Processes an audio dataset based on metadata from a CSV file.
    - Filters out missing audio files.
    - Balances the dataset by downsampling each class to the smallest class size.
    - Splits the dataset into training and testing sets.
    - Copies the audio files into structured directories for training and testing.
Usage:
------
1. Ensure the required libraries are installed.
2. Provide a CSV file with metadata for the audio dataset (e.g., file names and labels).
3. Place the audio files in the specified directory (AUDIO_DIR).
4. Run the script to process and balance the dataset.
"""


import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from sklearn.utils import resample
import shutil
from sklearn.model_selection import train_test_split


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

def get_audio_events():
        
    # Paths
    CSV_PATH = '/content/train.csv'
    AUDIO_DIR = '/content/ARCA23K.audio'
    OUTPUT_DIR = '/content/balanced_data'

    # Load metadata
    df = pd.read_csv(CSV_PATH)

    # Create full audio file paths
    df['path'] = df['fname'].astype(str) + '.wav'
    df['full_path'] = df['path'].apply(lambda x: os.path.join(AUDIO_DIR, x))

    # Filter out missing files
    df = df[df['full_path'].apply(os.path.exists)]

    print(f"Total available files: {len(df)}")

    # Count examples per label
    min_samples = df['label'].value_counts().min()

    # Downsample each class to the smallest size
    balanced_df = df.groupby('label').apply(lambda x: resample(x, replace=False, n_samples=min(min_samples, len(x))))
    balanced_df = balanced_df.reset_index(drop=True)

    print("Balanced class distribution:")
    print(balanced_df['label'].value_counts())
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)

    def copy_files(df, split):
        for _, row in df.iterrows():
            dest_dir = os.path.join(OUTPUT_DIR, split, row['label'])
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(row['full_path'], os.path.join(dest_dir, row['path']))

    copy_files(train_df, "train")
    copy_files(test_df, "test")


