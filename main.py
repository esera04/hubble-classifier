import torch
import os
import torch.nn
import numpy as np

from network import HubbleClassifier
from dataset import HubbleDataset
from train import train_model
from torch.utils.data import DataLoader

EPOCHS = 500
BATCH_SIZE = 24
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, 'cnn_data')

def main():
    model = HubbleClassifier()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    train_dataset = HubbleDataset(os.path.join(ROOT, 'train.csv'), DATA_DIR)
    test_dataset = HubbleDataset(os.path.join(ROOT, 'test.csv'), DATA_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_model(model, train_loader, test_loader, epochs=EPOCHS, device=device)


if __name__ == '__main__':
    main()

