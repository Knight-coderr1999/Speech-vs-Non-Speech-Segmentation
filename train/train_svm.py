import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from models.classical.svm import train_svm
from datasets.combined import get_combined_loader
import matplotlib.pyplot as plt
import seaborn as sns

def extract_flat_features(dataloader):
    X, y = [], []
    for features, label in dataloader:
        for i in range(features.shape[0]):
            flat_feat = features[i].view(-1).numpy()
            X.append(flat_feat)
            y.append(label[i].item())
    return np.array(X), np.array(y)


def plot_evaluation(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Speech", "Speech"], yticklabels=["Non-Speech", "Speech"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def train_and_evaluate_svm():
    loader = get_combined_loader(batch_size=8)
    X, y = extract_flat_features(loader)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = train_svm(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_evaluation(y_test, y_pred)
