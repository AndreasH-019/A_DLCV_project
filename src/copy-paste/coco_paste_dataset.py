import os
import cv2
from torchvision.datasets import CocoDetection
from copy_paste import copy_paste_class
import pickle
import numpy as np
import torch
import albumentations as A
from copy_paste import CopyPaste

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

@copy_paste_class
class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms
    ):
        super(CocoDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)

        self.ids = ids

    def load_example(self, index, pasteImg = False):
        # if we want to paste from diffusion model
        if pasteImg == True:

            # load random image from folder
            pasteRoot = '../data/coco_minitrain_25k/images_pruned/giraffes_elephants/'
            pasteAnnotRoot = '../data/coco_minitrain_25k/annotations'

            chosen_class = np.random.choice(['elephant', 'elephant'])
            if chosen_class == 'elephant':
                with open(pasteAnnotRoot+"/elephant_annotations.pickle", 'rb') as handle:
                    annot_dict = pickle.load(handle)
                    path = np.random.choice(list(annot_dict.keys()))
            else:
                with open(pasteAnnotRoot+"/giraffe_annotations.pickle", 'rb') as handle:
                    annot_dict = pickle.load(handle)
                    path = np.random.choice(list(annot_dict.keys()))

            image = cv2.imread(os.path.join(pasteRoot+chosen_class, path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # load mask which corresponds to image
            mask = [annot_dict[path]['mask'], annot_dict[path]['mask']]

            # mask.append()
            bbox = annot_dict[path]['bbox']

            # load bbox which corresponds to image
            bbox = [[bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], 0],[bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], 1]]

            #pack outputs into a dict
            output = {
                'image': image,
                'masks': mask,
                'bboxes': bbox
            }

        else:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)

            path = self.coco.loadImgs(img_id)[0]['file_name']


            image = cv2.imread(os.path.join(self.root, path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            #convert all of the target segmentations to masks
            #bboxes are expected to be (y1, x1, y2, x2, category_id)
            masks = []
            bboxes = []
            for ix, obj in enumerate(target):
                masks.append(self.coco.annToMask(obj))
                bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

            #pack outputs into a dict
            output = {
                'image': image,
                'masks': masks,
                'bboxes': bboxes
            }
        
        return self.transforms(**output)

class CopyPasteTrain(CocoDetectionCP):
    def __getitem__(self, item):
        img_data = super().__getitem__(item)
        image = (torch.tensor(img_data['image'])/255).permute(2,0,1)
        target = {}
        n = len(img_data['bboxes'])
        target['boxes'] = torch.zeros(size=(n, 4), dtype=torch.float32)
        target['labels'] = torch.zeros(size=(n,), dtype=torch.int64)
        target['masks'] = torch.zeros(size=(n, image.shape[1], image.shape[2]), dtype=torch.bool)

        for i, box in enumerate(img_data['bboxes']):
            target['boxes'][i] = torch.tensor([box[0], box[1], box[0]+box[2], box[1]+box[3]]).to(torch.float32)
            target['labels'][i] = torch.tensor(box[4])
            target['masks'][i] = torch.tensor(img_data['masks'][box[-1]]).to(torch.bool)
        return image, target

def get_paste_transform(task):
    if task == 'train':
        transform = A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
            A.PadIfNeeded(256, 256, border_mode=0),  # pads with image in the center, not the top left like the paper
            A.RandomCrop(256, 256),
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.)  # pct_objects_paste is a guess
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
        )
    elif task in ['val', 'test']:
        transform = A.Compose([], bbox_params=A.BboxParams(format="coco", min_visibility=0.05))
    return transform