import os
import cv2
from torchvision.datasets import CocoDetection
import pickle
import numpy as np
import torch
import albumentations as A
import random

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

class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms,
        paste_transforms,
        load_from_generated_p,
    ):
        super(CocoDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )
        self.paste_transforms = paste_transforms
        self.load_from_generated_p = load_from_generated_p

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)

        self.ids = ids

        self.gen_classes = ['elephant', 'giraffe']
        self.pasteRoot = '../data/coco_minitrain_25k/images_pruned/giraffes_elephants'
        self.pasteAnnotRoot = '../data/coco_minitrain_25k/annotations'
        self.init_gen_annot_dicts()
        self._split_transforms()

    def init_gen_annot_dicts(self):
        self.gen_annot_dicts = {}
        for gen_class in self.gen_classes:
            with open(os.path.join(self.pasteAnnotRoot, f"final_{gen_class}_annotations.pickle"), 'rb') as handle:
                self.gen_annot_dicts[gen_class] = pickle.load(handle)

    def load_example(self, index, paste_from_generated = False):
        # if we want to paste from diffusion model
        if paste_from_generated == True:
            chosen_class = np.random.choice(self.gen_classes)
            annot_dict = self.gen_annot_dicts[chosen_class]
            path = np.random.choice(list(annot_dict.keys()))

            image = cv2.imread(os.path.join(self.pasteRoot, chosen_class, path))
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

        return output
        # return self.transforms(**output)

    def _split_transforms(self):
        split_index = None
        for ix, tf in enumerate(list(self.transforms.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index+1:]

            #replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.transforms.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.transforms.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            #recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste_transform = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste_transform = None
            self.post_transforms = None

    def __getitem__(self, idx):
        img_data = self.load_example(idx)
        img_data = self.transforms(**img_data)
        before_img = img_data['image']
        if self.copy_paste_transform is not None:
            paste_idx = random.randint(0, self.__len__() - 1)
            paste_img_data = self.load_example(paste_idx, paste_from_generated=self.load_from_generated_p > random.random())
            paste_img_data = self.paste_transforms(**paste_img_data)
            paste_img_data = {f'paste_{key}': value for key, value in paste_img_data.items()}
            img_data = self.copy_paste_transform(**img_data, **paste_img_data)
            img_data = self.post_transforms(**img_data)
            img_data['index'] = idx # ADDED
            img_data['paste_index'] = paste_idx
            img_data['paste_image'] = paste_img_data['paste_image']
            img_data['before_image'] = before_img

        return img_data

class CopyPasteTrain(CocoDetectionCP):
    def __getitem__(self, item):
        img_data = super().__getitem__(item)
        image = (torch.tensor(img_data['image'])/255).permute(2, 0, 1)
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