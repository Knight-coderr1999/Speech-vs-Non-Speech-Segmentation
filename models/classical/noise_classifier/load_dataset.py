import os
from tqdm import tqdm
from utils.audio_utils import get_yamnet_embedding

def load_dataset(root_dir):
    X, y = [], []
    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for file in tqdm(os.listdir(class_dir), desc=f"Processing {label}"):
            if file.endswith(('.wav', '.mp3')):
                path = os.path.join(class_dir, file)
                emb = get_yamnet_embedding(path)
                if emb is not None:
                    X.append(emb)
                    y.append(label)
    return np.array(X), np.array(y)

