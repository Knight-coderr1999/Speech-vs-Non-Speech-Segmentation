from train.train_deep import train_deep_model
from datasets.librispeech import get_train_loader
from datasets.combined import get_combined_loader
from train.train_svm import train_and_evaluate_svm
import torch 

if __name__ == "__main__":
    train_loader = get_train_loader(batch_size=16, root_dir="/path/to/librispeech")
    model = train_deep_model(train_loader)
    torch.save(model.state_dict(), "cnn_speech_segmentation.pt")

    print("Training deep model:")
    train_loader = get_combined_loader(batch_size=16)
    model = train_deep_model(train_loader)
    torch.save(model.state_dict(), "cnn_speech_segmentation.pt")

    print("\nTraining classical SVM:")
    train_and_evaluate_svm()