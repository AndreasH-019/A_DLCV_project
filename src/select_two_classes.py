from pycocotools.coco import COCO
import json
import os
from tqdm import tqdm
import shutil
import copy
import random

SELECTED_CLASSES = [1, 25]
def prune_coco_images(coco, img_root):
    images = coco.dataset['images']
    new_images = []
    image_paths = os.listdir(img_root)
    print("pruning images")
    for image in tqdm(images):
        annots = coco.loadAnns(coco.getAnnIds(imgIds=[image['id']]))
        present_classes = [annot["category_id"] for annot in annots]
        selected_classes_present = len(set(SELECTED_CLASSES) - set(present_classes)) < len(SELECTED_CLASSES)
        if (image['file_name'] in image_paths) and selected_classes_present:
            new_images.append(image)
    coco.dataset['images'] = new_images

def prune_coco_annots(coco, img_root):
    annots = coco.dataset['annotations']
    new_annots = []
    print("pruning annots")
    for annot in tqdm(annots):
        if annot['category_id'] in SELECTED_CLASSES:
            new_annots.append(annot)
    coco.dataset['annotations'] = new_annots

def prune_image_folder(coco, img_root, new_img_root):
    if os.path.exists(new_img_root):
        shutil.rmtree(new_img_root)
    # os.mkdir(new_img_root, parents=True)
    os.makedirs(new_img_root)
    image_paths = [coco_image['file_name'] for coco_image in coco.dataset['images']]
    print("pruning image folder")
    for image_path in tqdm(image_paths):
        source_file = os.path.join(img_root, image_path)
        destination_file = os.path.join(new_img_root, image_path)
        shutil.copy(source_file, destination_file)

def remove_missing_images_from_annots(coco, img_root, new_json_file, new_img_root):
    prune_coco_annots(coco, img_root)
    prune_coco_images(coco, img_root)
    with open(new_json_file, 'w') as f:
        json.dump(coco.dataset, f)
    prune_image_folder(coco, img_root, new_img_root)

def train_val_split(coco_train, img_root_train, new_train_json, new_val_json, img_root_val):
    new_train = copy.deepcopy(coco_train.dataset)
    new_train['images'] = []
    new_val = copy.deepcopy(coco_train.dataset)
    new_val['images'] = []
    for image in coco_train.dataset['images']:
        if random.random() < 0.85:
            new_train['images'].append(image)
        else:
            new_val['images'].append(image)
            src_path = os.path.join(img_root_train, image['file_name'])
            des_path = os.path.join(img_root_val, image['file_name'])
            shutil.copy(src_path, des_path)
            os.remove(src_path)
    with open(new_val_json, 'w') as f:
        json.dump(new_val, f)
    with open(new_train_json, 'w') as f:
        json.dump(new_train, f)


def rename():
    annot_path = "../../data/coco_minitrain_25k/annotations/instances_test2017_pruned.json"
    if os.path.exists(annot_path):
        os.remove(annot_path)

    os.rename("../../data/coco_minitrain_25k/annotations/instances_val2017_pruned.json",
              annot_path)

    image_folder_path = "../../data/coco_minitrain_25k/images_pruned/test2017"
    if os.path.exists(image_folder_path):
        shutil.rmtree(image_folder_path)
    os.rename("../../data/coco_minitrain_25k/images_pruned/val2017",
              image_folder_path)

    if not os.path.exists("../../data/coco_minitrain_25k/images_pruned/val2017"):
        os.makedirs("../../data/coco_minitrain_25k/images_pruned/val2017")

tasks = ['train', 'val']
for task in tasks:
    coco = COCO(f"../../data/coco_minitrain_25k/annotations/instances_{task}2017.json")
    remove_missing_images_from_annots(coco, f"../../data/coco_minitrain_25k/images/{task}2017",
               f"../../data/coco_minitrain_25k/annotations/instances_{task}2017_pruned.json",
                                      f"../../data/coco_minitrain_25k/images_pruned/{task}2017")

rename()
coco_train = COCO("../../data/coco_minitrain_25k/annotations/instances_train2017_pruned.json")
train_val_split(coco_train,
                "../../data/coco_minitrain_25k/images_pruned/train2017",
                "../../data/coco_minitrain_25k/annotations/instances_train2017_pruned.json",
                "../../data/coco_minitrain_25k/annotations/instances_val2017_pruned.json",
                "../../data/coco_minitrain_25k/images_pruned/val2017")
