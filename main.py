# -*- coding: utf-8 -*-
from models import separate_speech_nonspeech, nonspeech_split
import os
from inference.inference import classify_noise
from dotenv import load_dotenv
import argparse

load_dotenv()

def classify_all_nonspeech(nonspeech_folder):
    for root, _, files in os.walk(nonspeech_folder):
        for file in files:
            if file.endswith(".wav") or file.endswith(".m4a"):
                file_path = os.path.join(root, file)
                print(f"\n File: {file}")
                predictions = classify_noise(file_path)
                print("Top predicted noise classes:")
                for cls, prob in predictions:
                    print(f"{cls}: {prob:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Forensic Noise Classification Pipeline")
    parser.add_argument('--input_path', required=False, help='Path to the input audio file')
    args = parser.parse_args()

    input_path = args.input_path or (os.getenv("in_base_path") + "traffic_mix.m4a")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist!")

    # Step 1 : Separate speech and nonspeech
    print("1. Speech and Nonspeech Separation")
    nonspeech_output = separate_speech_nonspeech(input_path)

    # Step 2 : Split nonspeech into different files
    print("2. Nonspeech Separation")
    nonspeech_folder = nonspeech_split(nonspeech_output)

    # Step 3 : Classify or identify each nonspeech audio
    print("3. Nonspeech Classification")
    classify_all_nonspeech(nonspeech_folder)

if __name__ == "__main__":
    main()
