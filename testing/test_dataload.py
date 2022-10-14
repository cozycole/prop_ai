import os
import unittest
import random
import torch
import matplotlib.pyplot as plt
from src.load_data import train_loader, train_dataset
class_list = ["distress", "no_distress", "unknown"]
labels_map = {
    0: "distress",
    1: "no_distress",
    2: "slight_distress",
    3: "unknown"
}

class test_unique(unittest.TestCase):
    # ensures the datasets are unique
    root = "/home/colet/programming/projects/ai_model/data"
    def test_no_dupes(self):
        cnt = 0
        no_dupes = True
        for cls in ["test", "valid", "train"]:
            cls_dir = os.path.join(self.root, cls)
            print(f"Checking {cls_dir} for dupes")
            for cmp_cls in ["test", "valid", "train"]:
                if cmp_cls != cls:
                    for img_dir in class_list:
                        imgs = [img for img in os.listdir(os.path.join(cls_dir, img_dir)) if ".jpg" in img]
                        for cmp_dir in class_list:
                            if cmp_dir != img_dir:
                                for img in [imgn for imgn in os.listdir(os.path.join(cls_dir, cmp_dir)) if ".jpg" in imgn]:
                                    cnt += 1
                                    if img in imgs:
                                        no_dupes = False
        print(f"Files checked: {cnt}")
        self.assertTrue(no_dupes)

class test_visdata(unittest.TestCase):
    # This is less of a testcase and more 
    # just to visualize the data

    def test_view_imgs(self):
        
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

# correct, total = 0, 0
# pred_tensor = torch.rand((64,4))
# _ , predicted = torch.max(pred_tensor, 1)
# label_tensor = torch.zeros((64,4))
# for x in label_tensor:
#     idx = random.randint(0,3)
#     x[idx] = 1
# # label_tensor = label_tensor.argmax(1)
# print("Predicted: ", predicted)
# print("Labels: ", label_tensor)
# corr_cnt = 0
# # for x,y in zip(predicted, label_tensor):
# # returns a tensor of the same size where
# # True/False is replaced for every value,
# # sum then gets number of true values
# print((label_tensor.argmax(1) == pred_tensor.argmax(1)).sum())
