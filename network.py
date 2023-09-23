#!/usr/bin/env python3.10

import torch

import torch.nn as nn
import torch.nn.functional as F

class HubbleClassifier(nn.Module):
    def __init__(self):
        super(HubbleClassifier, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lay1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.lay2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.lay3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.lay4 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3)
        self.lay5 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3)

        self.fc1 = nn.Linear(24 * 24 * 8, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 3)

        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # convolution net
        x = self.pool(F.relu(self.lay1(x)))
        x = self.pool(F.relu(self.lay2(x)))
        x = F.relu(self.lay3(x))
        x = F.relu(self.lay4(x))
        x = self.pool(F.relu(self.lay5(x))) # output volume 8 * 26 * 26

        # flatten to input vector
        x = torch.flatten(x, start_dim=1)

        # pass through FC layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# if __name__ == '__main__':
#     # sanity check model
#     import os
#     import random

#     from PIL import Image
#     from torchvision import transforms

#     DATA_SOURCE = "cnn_data"

#     img = Image.open(os.path.join(DATA_SOURCE, random.choice(os.listdir(DATA_SOURCE))))

#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     img = transform(img)

#     model = HubbleClassifier()

#     softmax = nn.Softmax()

#     output = softmax(model(img))

#     print(output)
