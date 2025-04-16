# -*- coding: utf-8 -*-
"""
This script is designed to augment audio datasets by mixing speech audio clips 
with urban noise clips. It provides functionality to mix multiple urban noise 
clips with a base audio clip and generate augmented datasets for training machine 
learning models.

Functions:
-----------
1. mix_audio_clips(base_audio_path, urban_paths, output_path):
    - Mixes a base audio clip with multiple urban noise clips.
    - Pads or truncates urban noise clips to match the length of the base audio.
    - Saves the mixed audio to the specified output path.

2. synthesize_urban_clips(urban_clips, output_dir):
    - Synthesizes a dataset by mixing speech audio clips with urban noise clips.
    - Reads metadata from a CSV file to determine the base audio clips.
    - Randomly selects 2 to 4 urban noise clips for each base audio clip.
    - Saves the mixed audio and updates the metadata with paths and labels.
    - Outputs the final augmented dataset as a CSV file.
Usage:
------
1. Ensure the required libraries are installed.
2. Place the urban noise clips in the specified directory (URBAN_PATH).
3. Provide a CSV file with metadata for the base audio clips.
4. Run the script to generate the augmented dataset.

Note:
-----
- The script assumes specific directory structures and file paths. Update 
  the paths (e.g., URBAN_PATH, MIXED_INPUT_DIR, OUTPUT_DIR) as needed.
- Ensure the metadata CSV file contains the required columns (e.g., 'mixed_audio').
"""
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

def synthesize_urban_clips(urban_clips, output_dir):
    # Paths
    URBAN_PATH = '/content/urban5000_clips'
    MIXED_INPUT_DIR = 'multi_speaker_dataset/train'
    OUTPUT_DIR = 'multi_speaker_augmented_dataset/train'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load your original CSV or dataframe
    df = pd.read_csv('/content/train_data_multi_mix.csv')
    urban_clips = glob(os.path.join(URBAN_PATH, '**/*.wav'), recursive=True)

    new_urban_paths = []
    updated_mix_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Randomly choose 2 to 4 urban clips
        random_urbans = random.sample(urban_clips, random.randint(2, 4))
        
        # Save them in the output dir
        chosen_basename = os.path.basename(row['mixed_audio']).replace('.wav', '_urbanmix.wav')
        out_path = os.path.join(OUTPUT_DIR, chosen_basename)
        
        # Mix and save
        mix_audio_clips(row['mixed_audio'], random_urbans, out_path)
        
        # Save info
        updated_mix_paths.append(out_path)
        new_urban_paths.append(random_urbans)
    df['urban_paths'] = new_urban_paths
    df['updated_mix_path'] = updated_mix_paths
    # Reload metadata
    urban_df = pd.read_csv(URBAN_PATH + '/metadata/UrbanSound8K.csv')

    # Map filenames to labels
    label_map = {
        row['slice_file_name']: row['class']
        for _, row in urban_df.iterrows()
    }

    def get_labels_from_paths(paths):
        filenames = [os.path.basename(p) for p in paths]
        return [label_map.get(f, 'unknown') for f in filenames]

    # Add new column with labels
    df['urban_labels'] = df['urban_paths'].apply(get_labels_from_paths)
    df.to_csv(os.path.join("multi_speaker_augmented_dataset", 'final_augmented_dataset_train.csv'), index=False)
    return df
