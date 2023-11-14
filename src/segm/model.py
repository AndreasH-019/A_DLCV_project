import lightning as L
import torch
import torchvision
from coco_dataset import CocoDataset, custom_collate_fn, plot_segmentation
import random
from torch.utils.data.dataloader import DataLoader
import torchmetrics

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
        self.log_metric(batch)

    def test_step(self, batch, batch_idx):
        self.log_metric(batch)
    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')

    def log_metric(self, batch):
        images, targets = batch
        outputs = self.maskRCNN(images)
        for output in outputs:
            output["masks"] = output["masks"] > 0.5
            output["masks"] = output["masks"].squeeze(1)
        metric_dict = self.meanAveragePrecision(outputs, targets)
        self.log("mAP", metric_dict['map'].item(), batch_size=len(images))
        # plot_segmentation(images[0], outputs[0]['masks'], outputs[0]['scores'])

    def get_dataset(self, task):
        image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = CocoDataset(root=f"../../data/coco_minitrain_25k/images_pruned/{task}2017",
                              annFile=f"../../data/coco_minitrain_25k/annotations/instances_{task}2017_pruned.json",
                              transform=image_transform)
        return dataset

    def get_dataloader(self, task):
        dataset = self.get_dataset(task)
        shuffle_options = {'train': True, 'val': False, 'test': False}
        if self.debug:
            dataset.ids = random.sample(dataset.ids, 1)
            dataloader = DataLoader(dataset=dataset, batch_size=1,
                                    shuffle=shuffle_options[task], num_workers=0, collate_fn=custom_collate_fn)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=4,
                                    shuffle=shuffle_options[task], num_workers=4, collate_fn=custom_collate_fn)
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

def get_model():
    pl_model = LitMaskRCNN()
    return pl_model