import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data.dataloader import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
import torchvision
import torch
from tqdm import tqdm

COCO_CLASSES = ["background",
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "street sign",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
                "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
                "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
                "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"
                ]

class CocoDataset(CocoDetection):

    def __init__(self, root, annFile, transform=None,
                 target_transform=None, transforms=None, size=[256, 256]):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.size = size

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        loaded_target = self._load_target(id)
        N = len(loaded_target)
        formatted_target = {"boxes": torch.zeros((N, 4), dtype=torch.float32),
                            "labels": torch.zeros((N), dtype=torch.int64),
                            "masks": torch.zeros((N, self.size[0], self.size[1]), dtype=torch.uint8)}
        if self.transforms is not None:
            image, loaded_target = self.transforms(image, loaded_target)

        for i, detection in enumerate(loaded_target):
            mask = torch.tensor(self.coco.annToMask(detection))
            formatted_target['masks'][i, :, :] = F.resize(mask.unsqueeze(0), self.size, antialias=True).squeeze(0)
            formatted_target['boxes'][i, :] = torch.tensor(detection['bbox'])
            formatted_target['labels'][i] = torch.tensor(detection['category_id'])

        formatted_target["boxes"] = torch.cat((formatted_target["boxes"][:, :2], formatted_target["boxes"][:, :2] + formatted_target["boxes"][:, 2:]), dim=1)
        formatted_target["boxes"] = formatted_target["boxes"]/torch.tensor([self.size[0], self.size[1], self.size[0], self.size[1]])
        return image, formatted_target


def custom_collate_fn(batch):
    imgs = []
    targets = []
    for sample in batch:
        img, target = sample
        imgs.append(img)
        targets.append(target)
    imgs = torch.stack(imgs)
    return imgs, targets

def plot_segmentation(image, segmentations, labels):
    image = (image/image.max()*255).to(torch.uint8)
    segmentations = segmentations.to(torch.bool)
    plot_img = draw_segmentation_masks(image, segmentations, alpha=0.5)
    plt.imshow(F.to_pil_image(plot_img))
    plt.show()


if __name__ == "__main__":

    image_transform = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ToTensor()])
    target_transform = transforms.Compose([
        transforms.Resize([256, 256])
    ])
    # dataset_train = CocoDataset(
    #     root="../../data/coco_minitrain_25k/images/val2017",
    #     annFile="../../data/coco_minitrain_25k/annotations/instances_val2017_pruned.json",
    #     transform=image_transform,
    # )
    dataset_train = CocoDetection(
        root="../../data/coco_minitrain_25k/images/train2017",
        annFile="../../data/coco_minitrain_25k/annotations/instances_train2017_pruned.json",
        transform=image_transform,
    )


    data_loader = DataLoader(dataset=dataset_train, batch_size=2,
                             shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    for images, targets in tqdm(data_loader):
        # image = images[0]  # Assuming batch size is 1
        # target = targets[0]
        #
        # labels = target['labels']
        # masks = target['masks']
        # plot_segmentation(image, masks, labels)
        # out = model(images, targets)
        # print(out)
        pass
