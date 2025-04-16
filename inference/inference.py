import torchaudio
from features.extractors import extract_mel_spectrogram
import torch
import numpy as np
import torchaudio.transforms as T
import joblib
from scipy.stats import skew, kurtosis
import tensorflow_hub as hub
import os
# Load classifier and label encoder
# update the path to the model and label encoder as per your directory structure
# Ensure the model is in the checkpoints directory

clf = joblib.load(os.getenv("checkpoint_path") + "noise_classifier.pkl")
label_encoder = joblib.load(os.getenv("checkpoint_path") + "label_encoder.pkl")

# Load YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def predict(model, waveform, sample_rate):
    features = extract_mel_spectrogram(waveform, sample_rate).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(features.cuda())
    return pred.item() > 0.5

def get_yamnet_embedding(audio_path):
    """
    Extract YAMNet embeddings with statistical pooling from a WAV file.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        waveform_np = waveform.numpy()
        _, embeddings, _ = yamnet_model(waveform_np)

        # Statistical features
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

def classify_noise(audio_path, threshold=0.6):
    """
    Classify noise with rejection threshold for 'Unknown' label.
    """
    feature = get_yamnet_embedding(audio_path)
    if feature is None:
        return [("Unknown", 0.0)]

    feature = feature.reshape(1, -1)
    probs = clf.predict_proba(feature)[0]
    
    top_idx = np.argmax(probs)
    top_prob = probs[top_idx]
    
    if top_prob < threshold:
        return [("Unknown", top_prob)]

    top_indices = np.argsort(probs)[::-1][:5]
    return [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top_indices]
