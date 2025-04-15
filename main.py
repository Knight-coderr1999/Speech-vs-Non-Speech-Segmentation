# from train.train_deep import train_deep_model
from datasets.librispeech import get_train_loader
# from datasets.combined import get_combined_loader
# from train.train_svm import train_and_evaluate_svm
from models import separate_speech_nonspeech, nonspeech_split
import os
import torch
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    input_path=os.getenv("in_base_path")+"traffic_mix.m4a"
    nonspeech_output=separate_speech_nonspeech(input_path)
    nonspeech_folder=nonspeech_split(nonspeech_output)