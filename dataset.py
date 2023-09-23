import os
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HubbleDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mapper = {
            'galaxies': 0,
            'nebulae': 1,
            'solarsystem': 2
        }

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path)
        label = self.mapper[self.img_labels.iloc[idx, 1]]
        
        image = self.transform(img)
        
        return image, label




if __name__ == '__main__':
    import numpy as np

    from network import HubbleClassifier
    from torch.nn import Softmax

    model = HubbleClassifier()
    loader = HubbleDataset("train.csv", "cnn_data")
    
    img, label = loader[689]
    output = model(img)
    softmax = Softmax(dim=-1)

    prediction = softmax(output).detach().numpy()
    # argmax returns index of maximum, max returns the maximum
    print(f"actual label is: {label}, predicted label is {np.argmax(prediction)} with probability {np.max(prediction)}")
