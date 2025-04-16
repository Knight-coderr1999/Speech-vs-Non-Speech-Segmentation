"""
This script provides functionality to download audio datasets required for 
speech and non-speech segmentation tasks. It includes methods to download 
the UrbanSound8K dataset and the ARCA23K dataset.

Functions:
-----------
1. download_urbansound8k():
    - Downloads the UrbanSound8K dataset using the `soundata` library.
    - Validates the dataset to ensure all files are downloaded correctly.
    - Prints information about a random example clip from the dataset.

2. download_arca23k():
    - Downloads the ARCA23K dataset from Zenodo using `wget`.
    - Handles multiple parts of the dataset (e.g., `.z01`, `.z02`, etc.).
    - Prints the status of each download (success or failure).

Dependencies:
-------------
- soundata: For downloading and managing the UrbanSound8K dataset.
- subprocess: For executing shell commands to download ARCA23K files.

Usage:
------
1. Ensure the required libraries (`soundata`) and tools (`wget`) are installed.
2. Call `download_urbansound8k()` to download and validate the UrbanSound8K dataset.
3. Call `download_arca23k()` to download the ARCA23K dataset.
"""

import soundata
import subprocess

def download_urbansound8k():
    """
    Download the UrbanSound8K dataset.
    """
    # Initialize the dataset
    dataset = soundata.initialize('urbansound8k')
    
    # Download the dataset
    dataset.download()
    
    # Validate that all the expected files are there
    dataset.validate()
    
    # Choose a random example clip
    example_clip = dataset.choice_clip()
    
    # Print the available data for the example clip
    print(example_clip)

def download_arca23k():
    urls = [
        ("ARCA23K.audio.z01", "https://zenodo.org/records/5117901/files/ARCA23K.audio.z01?download=1"),
        ("ARCA23K.audio.z02", "https://zenodo.org/records/5117901/files/ARCA23K.audio.z02?download=1"),
        ("ARCA23K.audio.z03", "https://zenodo.org/records/5117901/files/ARCA23K.audio.z03?download=1"),
        ("ARCA23K.audio.z04", "https://zenodo.org/records/5117901/files/ARCA23K.audio.z04?download=1"),
        ("ARCA23K.audio.zip", "https://zenodo.org/records/5117901/files/ARCA23K.audio.zip?download=1"),
    ]

    for filename, url in urls:
        print(f"Downloading {filename}...")
        result = subprocess.run(["wget", "-O", filename, url])
        if result.returncode != 0:
            print(f"Failed to download {filename}")
        else:
            print(f"Downloaded {filename} successfully.")