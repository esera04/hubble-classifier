from PIL import Image
import os
import pandas as pd
from torchvision import transforms

from pathlib import Path

Image.MAX_IMAGE_PIXELS = 20000000000000000
new_dir = 'cnn_data'
images = os.listdir('data')
train_df = pd.read_csv('train.csv')
for image in images:
    if image in train_df['filename'].unique():   
        img = Image.open(os.path.join('data', image))
        pipeline = transforms.Compose([
            transforms.Resize(300),
            transforms.GaussianBlur(5),
            transforms.ColorJitter(),
            transforms.RandomPerspective(),
            transforms.RandomVerticalFlip(0.4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.CenterCrop(224)
        ])
        transforms_image = pipeline(img)
        new_filename = f'aug2-{image}'
        transforms_image.save(os.path.join(new_dir, new_filename))
        classification = train_df.loc[train_df['filename'] == image]['classification'].item()
        train_df.loc[len(train_df)] = { 'filename': new_filename, 'classification': classification }
train_df.to_csv("train2.csv", index=False)

# images = os.listdir('aug_data')
# df = pd.read_csv('train2.csv')
# print(df.loc[df['filename'] == 'potw2206a.jpg']['classification'])
# for image in images:
#     classification = df.loc[df['filename'] == image]['classification'].item()
#     print(classification)
#     new_filename = f'aug-{image}'
#     df.loc[len(df)] = { 'filename': new_filename, 'classification': classification }

# df.to_csv('train3.csv', index=False)
# images = os.listdir('aug_data')
# for image in images:
#     Path(os.path.join('aug_data', image)).rename(os.path.join('cnn_data', f'aug-{image}'))

# from sklearn.model_selection import train_test_split

# df = pd.read_csv("train.csv")
# train, test = train_test_split(df, test_size=0.2)
# print(len(train.loc[train["classification"] == "solarsystem"]))
# print(len(train.loc[train["classification"] == "galaxies"]))
# print(len(train.loc[train["classification"] == "nebulae"]))

# print(len(test.loc[test["classification"] == "solarsystem"]))
# print(len(test.loc[test["classification"] == "galaxies"]))
# print(len(test.loc[test["classification"] == "nebulae"]))

# train.to_csv("train.csv", index=False)
# test.to_csv("test.csv", index=False)