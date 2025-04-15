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