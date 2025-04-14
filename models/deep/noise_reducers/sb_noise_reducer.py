import os
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import shutil
from tqdm import tqdm
import wave
import contextlib
import webrtcvad
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pyannote.audio import Pipeline
from speechbrain.lobes.models import SpectralMaskEnhancement
import torch

# Load models once (global for each subprocess)
denoiser = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank", savedir="tmp/denoiser"
)

def denoise_audio(input_path, output_path):
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)
    noisy_audio, sr = torchaudio.load(input_path)
    # Compute lengths tensor (required by speechbrain)
    lengths = torch.tensor([noisy_audio.shape[1]]) / noisy_audio.shape[1]
    denoised = denoiser.enhance_batch(noisy_audio, lengths)
    torchaudio.save(output_path, denoised.cpu(), sr)