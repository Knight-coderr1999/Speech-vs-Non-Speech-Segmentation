import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.cnn_yoho import CNN_YOHO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


def evaluate_model(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1


def train_deep_model(dataloader, num_epochs=10, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    model = CNN_YOHO()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log_file = open(os.path.join(output_dir, "training_log.txt"), "w")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc, prec, rec, f1 = evaluate_model(model, dataloader)
        log_msg = f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - Acc: {acc:.4f} - Prec: {prec:.4f} - Rec: {rec:.4f} - F1: {f1:.4f}"
        print(log_msg)
        log_file.write(log_msg + "\n")

    log_file.close()
    return model