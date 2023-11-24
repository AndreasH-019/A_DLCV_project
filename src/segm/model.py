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
from dice import dice

class LitMaskRCNN(L.LightningModule):
    def __init__(self, paste_mode, debug):
        super().__init__()
        # weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        # self.maskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        self.maskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        assert paste_mode in ['none', 'gen', 'real']
        self.paste_mode = paste_mode
        self.debug = debug
        self.save_hyperparameters()
        self.meanAveragePrecision = torchmetrics.detection.mean_ap.MeanAveragePrecision(iou_type='segm')

    def set_debug(self, new_debug):
        self.debug = new_debug
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.maskRCNN(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("loss", loss.item())
        self.log("loss_mask", loss_dict['loss_mask'].item())
        # plot_segmentation(images[0], targets[0]['masks'], targets[0]['labels'])
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
        dice_score = dice(outputs, targets, threshold=0.9)
        self.log("mAP", metric_dict['map'].item(), batch_size=len(images))
        self.log("dice", dice_score, batch_size=len(images))
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
        load_from_generated_options = {'none': True, 'gen': True, 'real': False}
        transform, paste_transform = get_paste_transform(task, self.paste_mode)
        dataset = CopyPasteTrain(
            f'../data/coco_minitrain_25k/images_pruned/{task}2017',
            f'../data/coco_minitrain_25k/annotations/instances_{task}2017_pruned.json',
            transform,
            paste_transform,
            load_from_generated=load_from_generated_options[self.paste_mode]
        )
        return dataset

    def get_dataloader(self, task):
        dataset = self.get_dataset(task)
        shuffle_options = {'train': True, 'val': False, 'test': False}
        if self.debug:
            dataset.ids = random.sample(dataset.ids, 1)
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
        if (not self.debug) and (batch_idx % 1 == 0):
            return True
        return False

    def forward(self, *args, **kwargs):
        return self.maskRCNN(*args, **kwargs)


def get_model():
    pl_model = LitMaskRCNN()
    return pl_model

def get_paste_transform(task, paste_mode):
    copy_paste_p_options = {'none': 0.0, 'gen': 0.5, 'real': 0.5}
    paste_transforms = A.Compose([
        A.ShiftScaleRotate(shift_limit=(-0.9, 0.9), rotate_limit=(-0, 0),
                           scale_limit=(-0.9, 0.1), border_mode=0, p=0.8),

        A.Resize(256, 256)
    ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
    )
    if task == 'train':
        transform = A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1),
            A.Resize(256, 256),
            CopyPaste(blend=True, sigma=0.1, pct_objects_paste=1.0, p=copy_paste_p_options[paste_mode]),
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
        )
    elif task in ['val', 'test']:
        transform = A.Compose([
            A.Resize(256, 256)
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05))
    return transform, paste_transforms