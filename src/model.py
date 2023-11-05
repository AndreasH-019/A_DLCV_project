import lightning as L
import torch
import torchvision
from coco_dataset import CocoDataset, custom_collate_fn
import random
from torch.utils.data.dataloader import DataLoader, Dataset

class LitMaskRCNN(L.LightningModule):

    def __init__(self, børge):
        super().__init__()
        self.maskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        self.børge = børge
        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.maskRCNN(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def train_dataloader(self):
        image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize([256, 256]),
                                                          torchvision.transforms.ToTensor()])
        dataset = CocoDataset(root="../../data/coco_minitrain_25k/images_pruned/train2017",
                                 annFile="../../data/coco_minitrain_25k/annotations/instances_train2017_pruned.json",
                                 transform=image_transform)
        dataset.ids = random.sample(dataset.ids, 2)
        dataloader = DataLoader(dataset=dataset, batch_size=2,
                   shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
        return dataloader

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.maskRCNN(images)

        # Få udgange for hver nøgle
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        masks = outputs[0]['masks']

        # Beregn mAP
        detection_threshold = 0.5  # Justér dette efter behov
        iou_threshold = 0.5  # Justér dette efter behov

        # Anvend ikke-maximum suppression (NMS) for at fjerne overlappende bokse
        keep = torchvision.ops.nms(boxes, scores, detection_threshold)
        filtered_boxes = boxes[keep]
        filtered_labels = labels[keep]
        filtered_scores = scores[keep]
        filtered_masks = masks[keep]

        # Sammenlign forudsigede bokse med de rigtige bokse ved hjælp af IoU
        iou = torchvision.ops.box_iou(filtered_boxes, targets[0]['boxes'])

        # Beregn mAP for bokse ved at sammenligne forudsigelser og de rigtige bokse
        num_detections = len(filtered_boxes)
        num_gt = len(targets[0]['boxes'])

        tp = torch.zeros(num_detections)
        fp = torch.zeros(num_detections)

        for i in range(num_gt):
            for j in range(num_detections):
                if iou[j, i] >= iou_threshold and filtered_labels[j] == targets[0]['labels'][i]:
                    tp[j] = 1
                else:
                    fp[j] = 1

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Beregn mAP for bokse ved at integrere precision-recall-kurven
        ap = torch.trapz(precision, recall)
        mAP = ap.item()

        # Evaluer masken ved hjælp af beregning af pixelpræcision for hver maske
        mask_iou_threshold = 0.5  # Justér dette efter behov

        gt_masks = targets[0]['masks']
        detected_masks = filtered_masks

        num_detected_masks = len(detected_masks)
        num_gt_masks = len(gt_masks)

        iou_matrix = torch.zeros((num_detected_masks, num_gt_masks))

        for i in range(num_detected_masks):
            for j in range(num_gt_masks):
                iou = torchvision.ops.mask_iou(detected_masks[i], gt_masks[j])
                iou_matrix[i, j] = iou

        pixel_precision = torch.zeros(num_detected_masks)

        for i in range(num_detected_masks):
            max_iou = iou_matrix[i].max()
            if max_iou >= mask_iou_threshold:
                pixel_precision[i] = 1

        mask_mAP = pixel_precision.mean().item()

        self.log("box_mAP", mAP)
        # return {'box_mAP': mAP, 'mask_mAP': mask_mAP}
        # return {'box_mAP': mAP}
    def test_dataloader(self):
        image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize([256, 256]),
                                                          torchvision.transforms.ToTensor()])
        dataset = CocoDataset(root="../../data/coco_minitrain_25k/images_pruned/val2017",
                              annFile="../../data/coco_minitrain_25k/annotations/instances_val2017_pruned.json",
                              transform=image_transform)
        dataset.ids = random.sample(dataset.ids, 10)
        dataloader = DataLoader(dataset=dataset, batch_size=2,
                                shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def get_model():
    pl_model = LitMaskRCNN(børge=5)
    return pl_model
