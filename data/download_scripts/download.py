import soundata

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