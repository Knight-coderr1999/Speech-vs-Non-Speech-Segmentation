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
    
    # use this nonspeech_folder for **classification**
    
    # train_loader = get_train_loader(batch_size=16, root_dir="/path/to/librispeech")
    # model = train_deep_model(train_loader)
    # torch.save(model.state_dict(), "cnn_speech_segmentation.pt")

    # print("Training deep model:")
    # train_loader = get_combined_loader(batch_size=16)
    # model = train_deep_model(train_loader)
    # torch.save(model.state_dict(), "cnn_speech_segmentation.pt")

    # print("\nTraining classical SVM:")
    # train_and_evaluate_svm()