
# this is where all neighborhood directories are located
# each one having sub-dirs containing the classes of each
IMAGE_ROOT = "/media/colet/Images"
DSET_ROOT = "/home/colet/programming/projects/ai_model/data"
CLASS_DISTRIB = {
    "distress" : 1,
    "slight_distress": 1,
    "no_distress" : 1,
    "unknown" : 1
}
class_list = ["distress", "no_distress", "unknown"]
DSET_RATIO = {
    "train" : 0.8,
    "valid" : 0.1,
    "test": 0.1
}