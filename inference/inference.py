import torchaudio
from models.deep.cnn_boundary import YOHO_CNN
from features.extractors import extract_mel_spectrogram
import torch
def predict(model, waveform, sample_rate):
    features = extract_mel_spectrogram(waveform, sample_rate).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(features.cuda())
    return pred.item() > 0.5
