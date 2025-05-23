import librosa
import os
import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF

def nonspeech_split(input_file):
    base_path=input_file.replace(".wav","")+"/"
    os.makedirs(base_path, exist_ok=True)
    y, sr = librosa.load(input_file, sr=None, mono=True)

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))

    n_components = 4
    model = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=500)
    W = model.fit_transform(S)
    H = model.components_

    # Separate sources
    sources = []
    for i in range(n_components):
        source_spec = np.outer(W[:, i], H[i])
        # Convert to complex for ISTFT
        phase = np.angle(librosa.stft(y, n_fft=1024, hop_length=512))
        S_complex = source_spec * np.exp(1j * phase)
        y_i = librosa.istft(S_complex, hop_length=512)
        sources.append(y_i)
        sf.write(f"{base_path}separated_nonspeech_{i+1}.wav", y_i, sr)

    print(" Separated sources saved as 'separated_nonspeech_1.wav', ..., etc.")
    return base_path
