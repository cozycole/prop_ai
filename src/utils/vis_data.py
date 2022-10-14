"""
Test for creating new dataset with my own images
"""
import os
import torch
import torch.nn as nn
from time import sleep
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# class HouseDataset(datasets.ImageFolder):
#     def __init__(self, root: str, transform = None, target_transform = None, loader = None, is_valid_file = None):
#         super().__init__(root, transform, target_transform, loader, is_valid_file)

#     def __getitem__(self, index: int):
#         # here is where you can apply CV operations
#         return super().__getitem__(index)


root = "input"
train_dataset = datasets.ImageFolder(
    root=os.path.join(root, "train"),
    transform=transforms.Compose([
        transforms.CenterCrop((800,800)),
        transforms.Resize(600),
        transforms.ToTensor()]
    ),
    target_transform=transforms.Lambda(lambda y: torch.zeros(4, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
valid_dataset = datasets.ImageFolder(
    root=os.path.join(root, "val"),
    transform=transforms.Compose([
        transforms.CenterCrop((800,800)),
        transforms.Resize(600),
        transforms.ToTensor()]
    ),
    target_transform=transforms.Lambda(lambda y: torch.zeros(4, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
test_dataset = datasets.ImageFolder(
    root=os.path.join(root, "test"),
    transform=transforms.Compose([
        transforms.CenterCrop((800,800)),
        transforms.Resize(600),
        transforms.ToTensor()]
    ),
    target_transform=transforms.Lambda(lambda y: torch.zeros(4, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

train_loader = DataLoader(train_dataset ,batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset ,batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset ,batch_size=10, shuffle=True)

img, label = train_dataset[0]
img2, label2 = train_dataset[12]
print(img ,label)
plt.imshow(img.permute(1,2,0))
plt.show()
print(img2, label2)
input("press enter")
# print(img.size())
# plt.axis("off")
# plt.imshow(img.permute(1,2,0))
# print("Img 1 label", label)
# plt.show()
# plt.imshow(img2.permute(1,2,0))
# print("Img 2 label", label2)
# plt.show()

labels_map = {
    0: "distress",
    1: "no_distress",
    2: "slight_distress",
    3: "unknown"
}

figure = plt.figure(figsize=(8,8)) # figure size in inches
cols, rows = 2,2
showed_labels = set()
for i in range(1, cols* rows + 1):
  while True:
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    # sample_idx = random.randint(0, len(training_data)-1) THIS ALSO WORKS
    img, label = train_dataset[sample_idx]
    print(type(img), type(label))
    if label not in showed_labels:

        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        print(img.size())
        plt.imshow(img.permute(1,2,0), cmap="gray")
        showed_labels.add(label)
        break
plt.show()


# input("press enter to continue")


# train_features, train_labels = next(iter(train_loader))
# print("Batch shape: ", train_features.size())
# print("Labels batch shape: ", train_labels.size())