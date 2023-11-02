from pycocotools.coco import COCO
import json
import os
from tqdm import tqdm
import shutil

SELECTED_CLASSES = [1, 25]

def prune_coco_images(coco, img_root):
    images = coco.dataset['images']
    new_images = []
    image_paths = os.listdir(img_root)
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
    for annot in tqdm(annots):
        if annot['category_id'] in SELECTED_CLASSES:
            new_annots.append(annot)
    coco.dataset['annotations'] = new_annots

def prune_image_folder(coco, img_root, new_img_root):
    if os.path.exists(new_img_root):
        shutil.rmtree(new_img_root)
    os.mkdir(new_img_root)
    image_paths = [coco_image['file_name'] for coco_image in coco.dataset['images']]
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

tasks = ['train', 'val']
for task in tasks:
    coco = COCO(f"../../data/coco_minitrain_25k/annotations/instances_{task}2017.json")
    remove_missing_images_from_annots(coco, f"../../data/coco_minitrain_25k/images/{task}2017",
               f"../../data/coco_minitrain_25k/annotations/instances_{task}2017_pruned.json",
                                      f"../../data/coco_minitrain_25k/images_pruned/{task}2017")

