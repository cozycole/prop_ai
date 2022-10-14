from math import ceil
import os
import shutil
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import config
# this is where all neighborhood directories are located
# each one having sub-dirs containing the classes of each
img_root = "/media/colet/Images"
dset_root = "/home/colet/programming/projects/ai_model/data"
class_list = ["distress", "no_distress", "unknown"]

class ImageFolderWithPaths(datasets.ImageFolder):
    
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_img_root_distrib(root, class_list):
    # gets distribution from ssd
    distrib_dict = {}
    for cls in class_list:
        distrib_dict[cls] = 0

    for dir in os.listdir(root):
        if '.' not in dir and (dir != "lost+found"):
            path = os.path.join(root, dir)
            # only concerned with class named dirs
            for cls in os.listdir(path):
                if cls in class_list:
                    imgs = filter_valid_imgs(os.listdir(os.path.join(path, cls)))
                    # get class total for images within this class dir
                    distrib_dict[cls] += len(imgs)
    return distrib_dict

def create_data_dirs(root, class_list):
    data_dir = os.path.join(root, "data")
    if os.path.exists(data_dir):
        os.rename(data_dir, os.path.join(root, "data_old"))

    for dir in ["test", "valid", "train"]:
        for cls in class_list:
            path = os.path.join(data_dir, dir, cls)
            os.makedirs(path)

def fill_data_dirs(root, class_list, class_ratios, dataset_ratios):
    # Get all the distressed imgs
    # Based on number of distressed imgs
    dist_dict = get_img_root_distrib(root, class_list)
    print("Total Distrib: ", dist_dict)
    distress_cnt = dist_dict["distress"]
    root_dirs = [dir for dir in os.listdir(root) if '.' not in dir and dir != "lost+found"]
    
    for dir in root_dirs:
        # ex: path = /media/colet/Images/detroit1
        path = os.path.join(root, dir)
        for cls in class_list:
            curr_path = os.path.join(path, cls)
            if cls in ["distress"]:
                curr_path = os.path.join(path, cls)
                files_to_copy = filter_valid_imgs(os.listdir(curr_path))
                copy_to_data_dirs(curr_path, config.DSET_ROOT, cls, files_to_copy ,dataset_ratios)
            else:
                # otherwise get random sample from no_distress or unknown
                cls_ratio = class_ratios[cls]
                total = int(cls_ratio * distress_cnt / len(root_dirs))
                img_dir = filter_valid_imgs(os.listdir(curr_path))
                total = total if total < len(img_dir) else len(img_dir)
                files_to_copy = random.sample(img_dir, total)
                print(curr_path, len(img_dir))
                print(f"Files to copy: {len(files_to_copy)}")
                copy_to_data_dirs(curr_path, config.DSET_ROOT, cls, files_to_copy ,dataset_ratios)

def filter_valid_imgs(in_list):
    return [img for img in in_list if (".jpg" in img and img[:2] != "._")]
            
def get_dir_percentile(path, cls, cls_total):
    return len(filter_valid_imgs(os.listdir(os.path.join(path, cls)))) / cls_total

def get_img_distrib(root, classes):
    # gets distribution from ssd
    distrib_dict = {}
    for cls in class_list:
        distrib_dict[cls] = 0

    for cls in os.listdir(root):
        if cls in classes:
            imgs = filter_valid_imgs(os.listdir(os.path.join(root, cls)))
            # get class total for images within this class dir
            distrib_dict[cls] += len(imgs)
    return distrib_dict

def copy_to_data_dirs(src_root, dest_root, cls_label, file_list, dset_ratio):
    """
    src_root path need to include the class within it
    """
    fcount = len(file_list)
    random.shuffle(file_list)

    test_cnt = int(ceil(fcount*dset_ratio["test"]))
    test_set = file_list[:test_cnt]

    val_cnt = int(ceil(fcount*dset_ratio["valid"]))
    val_set = file_list[test_cnt: (test_cnt + val_cnt)]

    train_set = file_list[(test_cnt + val_cnt):]

    label_set = zip(["train", "valid", "test"], [train_set, val_set, test_set])
    for label, set in label_set:
        dst_path = os.path.join(dest_root, label, cls_label)
        copy_file_list(src_root, dst_path, set)
        # print(f"Copy {len(set)} files from {src_root} to {dst_path}")


def copy_file_list(src_root, dest_root, file_list):
    for file in file_list:
        if ".jpg" in file:
            curr_img = os.path.join(src_root, file)
            shutil.copy2(curr_img, dest_root)

def move_files(root, dest_root):
    dest_dirs = os.listdir(dest_root)
    for dir in os.listdir(root):
        if dir[0] != '.' and dir in dest_dirs:
            dest_dir_path = os.path.join(dest_root, dir)
            root_dir_path = os.path.join(root, dir)
            for img in os.listdir(root_dir_path):
                if img[0] != '.':
                    new_img = os.path.join(dest_dir_path, img)
                    old_img = os.path.join(root_dir_path, img)
                    os.rename(old_img, new_img)

def create_dataset(dest_path):
    dataset_path = os.path.join(dest_path, "data")
    if os.path.exists(dataset_path):
        input("Dataset already exists! Press Enter to create new one")
    create_data_dirs(os.getcwd(), class_list)
    print("got here")
    fill_data_dirs(img_root, class_list, config.CLASS_DISTRIB, config.DSET_RATIO)

img_trans = transforms.Compose([
        transforms.CenterCrop((800,1000)),
        transforms.Resize((600, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

train_dataset = ImageFolderWithPaths(
    root=os.path.join(dset_root, "train"),
    transform=img_trans,
    target_transform=transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

test_dataset = ImageFolderWithPaths(
    root=os.path.join(dset_root, "test"),
    transform=img_trans,
    target_transform=transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )



# valid_dataset = datasets.ImageFolder(
#     root=os.path.join(root, "val"),
#     transform=transforms.Compose([
#         transforms.CenterCrop((800,800)),
#         transforms.ToTensor()]
#     ),
#     target_transform=transforms.Lambda(lambda y: torch.zeros(4, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

train_loader = DataLoader(train_dataset ,batch_size=64, shuffle=True)
# valid_loader = DataLoader(valid_dataset ,batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset ,batch_size=64, shuffle=True)

# print(get_img_root_distrib(img_root, class_list))


if __name__ == '__main__':
    create_dataset("/home/colet/programming/projects/ai_model")