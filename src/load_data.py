import os
import torch
from torchvision import datasets, transforms

class ImageFolderWithPaths(datasets.ImageFolder):
    
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

scene_based_trans = transforms.Compose([
        transforms.CenterCrop((800,1000)),
        transforms.Resize((600, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

patch_based_trans = transforms.Compose([
        transforms.CenterCrop((200,200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def create_dset(img_trans, dset_root, usage):
    dataset = ImageFolderWithPaths(
        root=os.path.join(dset_root, usage),
        transform=img_trans,
        target_transform=transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
    return dataset

# train_loader = DataLoader(train_dataset ,batch_size=64, shuffle=True)
# valid_loader = DataLoader(valid_dataset ,batch_size=10, shuffle=True)
# test_loader = DataLoader(test_dataset ,batch_size=64, shuffle=True)
