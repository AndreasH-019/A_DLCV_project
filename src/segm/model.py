import lightning as L
import torch
import torchvision
from coco_dataset import CocoDataset, custom_collate_fn, plot_segmentation, get_segmentation_image
import random
from torch.utils.data.dataloader import DataLoader
import torchmetrics
import albumentations as A
import sys
sys.path.append("copy-paste")
from coco_paste_dataset import CopyPasteTrain
from copy_paste import CopyPaste

class LitMaskRCNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        # weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        # self.maskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        self.maskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        self.save_hyperparameters(logger=False)
        self.meanAveragePrecision = torchmetrics.detection.mean_ap.MeanAveragePrecision(iou_type='segm')
        self.debug = False

    def set_debug(self, new_debug):
        self.debug = new_debug

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.maskRCNN(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("loss", loss.item())
        self.log("loss_mask", loss_dict['loss_mask'].item())
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.maskRCNN(images)
        for output in outputs:
            output["masks"] = output["masks"] > 0.5
            output["masks"] = output["masks"].squeeze(1)
        metric_dict = self.meanAveragePrecision(outputs, targets)
        self.log("mAP", metric_dict['map'].item(), batch_size=len(images))

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.maskRCNN(images)
        for output in outputs:
            output["masks"] = output["masks"] > 0.5
            output["masks"] = output["masks"].squeeze(1)
        metric_dict = self.meanAveragePrecision(outputs, targets)
        self.log("mAP", metric_dict['map'].item(), batch_size=len(images))
        if self.should_log_image(batch_idx):
            plot_img = get_segmentation_image(images[0], outputs[0]['masks'], outputs[0]['labels'],
                                              outputs[0]['scores'])
            self.logger.experiment.add_image(f'segm_image_{batch_idx}', plot_img, 0)

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')

    def get_dataset(self, task):
        transform, paste_transform = get_paste_transform(task)
        dataset = CopyPasteTrain(
            f'../data/coco_minitrain_25k/images_pruned/{task}2017',
            f'../data/coco_minitrain_25k/annotations/instances_{task}2017_pruned.json',
            transform,
            paste_transform
        )
        return dataset

    def get_dataloader(self, task):
        dataset = self.get_dataset(task)
        shuffle_options = {'train': True, 'val': False, 'test': False}
        if self.debug:
            dataset.ids = random.sample(dataset.ids, 2)
            dataloader = DataLoader(dataset=dataset, batch_size=1,
                                    shuffle=shuffle_options[task], num_workers=0, collate_fn=custom_collate_fn)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=8,
                                    shuffle=shuffle_options[task], num_workers=4, collate_fn=custom_collate_fn)
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

    def should_log_image(self, batch_idx):
        if self.debug and (batch_idx % 1 == 0):
            return True
        if (not self.debug) and (batch_idx % 10 == 0):
            return True
        return False


def get_model():
    pl_model = LitMaskRCNN()
    return pl_model

def get_paste_transform(task):
    paste_transforms = A.Compose([
        A.RandomScale(scale_limit=(-0.9, -0.6), p=1),
        A.PadIfNeeded(256, 256, border_mode=0),
        A.RandomCrop(256, 256, p=0.5),
        A.Affine(translate_px={'x': (-150, 150), 'y': (-150, 150)}, p=1.0, rotate=[-180, 180]),
        A.Resize(256, 256)
    ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
    )
    if task == 'train':
        transform = A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
            A.PadIfNeeded(256, 256, border_mode=0),  # pads with image in the center, not the top left like the paper
            A.RandomCrop(256, 256, p=0.5),
            A.Resize(256, 256),
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.0),  # pct_objects_paste is a guess
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
        )
    elif task in ['val', 'test']:
        transform = A.Compose([
            A.Resize(256, 256)
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05))
    return transform, paste_transforms