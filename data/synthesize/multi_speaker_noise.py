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
