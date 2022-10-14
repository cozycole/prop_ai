"""
Distribution Goal:
    - Get all distressed images from each neighborhood directory (this sets dataset baseline)
    - Get 3x as many slightly_distressed images (randomly selected)
    - Get 5x as many no_distress houses
    - Get 3x unknown
Cleaning Goal:
    - Need to specify a uniform
"""
import os
from PIL import Image

root = "/media/colet/Images"

def get_img_distrib(root):
    img_total = 0
    for dir in os.listdir(root):
        if '.' not in dir:
            dirs = os.listdir(os.path.join(root,dir))
            print(f"{dir} has {len(dirs)} images")
            img_total += len(dirs)
    print(f"Total images {img_total}")

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

root = "/home/colet/programming/projects/ai_model/marked_data"

for img in os.listdir(root):
    if ".jpg" in img:
        path = os.path.join(root, img)
        curr_img = Image.open(path)
        crop_img = curr_img.crop((0,0, 1000,800))
        crop_img.save(path, "JPEG")
# root = os.path.join(root, "south_uni_shots")
# get_img_distrib(root)