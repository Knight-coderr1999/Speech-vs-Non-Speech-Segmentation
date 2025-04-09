import torchaudio
import torch
import torchaudio.transforms as T

def extract_mfcc(waveform, sample_rate, n_mfcc=40):
    mfcc = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)(waveform)
    return mfcc

def extract_mel_spectrogram(waveform, sample_rate):
    mel = T.MelSpectrogram(sample_rate=sample_rate)(waveform)
    return torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
