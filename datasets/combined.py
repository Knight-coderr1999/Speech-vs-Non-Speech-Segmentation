from torch.utils.data import ConcatDataset
from datasets.librispeech import LibriSpeechDataset
from datasets.urbansound8k import UrbanSoundDataset
from datasets.audioset import AudioSetDataset

def get_combined_loader(batch_size=16, speech_dir="/path/to/librispeech",
                         noise_meta_us8k="/path/to/UrbanSound8K/metadata.csv",
                         noise_audio_us8k="/path/to/UrbanSound8K/audio",
                         noise_meta_audioset="/path/to/audioset/metadata.csv",
                         noise_audio_audioset="/path/to/audioset/audio"):
    speech_ds = LibriSpeechDataset(speech_dir)
    noise_ds_us8k = UrbanSoundDataset(noise_meta_us8k, noise_audio_us8k)
    noise_ds_audioset = AudioSetDataset(noise_meta_audioset, noise_audio_audioset)
    combined_ds = ConcatDataset([speech_ds, noise_ds_us8k, noise_ds_audioset])
    return DataLoader(combined_ds, batch_size=batch_size, shuffle=True)
