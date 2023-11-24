import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as F
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

class CocoDataset(torchvision.datasets.CocoDetection):

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
                            "masks": torch.zeros((N, image.size[1], image.size[0]), dtype=torch.uint8)}
        if self.transforms is not None:
            image, loaded_target = self.transforms(image, loaded_target)

        for i, detection in enumerate(loaded_target):
            mask = torch.tensor(self.coco.annToMask(detection))
            formatted_target['masks'][i, :, :] = mask
            formatted_target['boxes'][i, :] = torch.tensor(detection['bbox'])
            formatted_target['labels'][i] = torch.tensor(detection['category_id'])

        formatted_target["boxes"] = torch.cat((formatted_target["boxes"][:, :2], formatted_target["boxes"][:, :2] + formatted_target["boxes"][:, 2:]), dim=1)
        formatted_target['masks'] = formatted_target["masks"].to(torch.bool)
        return image, formatted_target


def custom_collate_fn(batch):
    imgs = []
    targets = []
    for sample in batch:
        img, target = sample
        imgs.append(img)
        targets.append(target)
    return imgs, targets

def plot_segmentation(image, segmentations, labels, scores=None):
    plot_img = get_segmentation_image(image, segmentations, labels, scores)
    plt.imshow(F.to_pil_image(plot_img))
    plt.show()

def get_segmentation_image(image, segmentations, labels, scores):
    if scores != None:
        keep = scores > 0.90
        segmentations = segmentations[keep]
    segmentations = segmentations.to(torch.bool)
    image = (image / image.max() * 255).to(torch.uint8)

    colors = 22*['black'] + ['blue'] + ['black']*2 + ['red']
    color_map = [colors[label] for label in labels]

    plot_img = torchvision.utils.draw_segmentation_masks(image, segmentations, alpha=0.5,
                                                         colors=color_map)
    return plot_img

if __name__ == "__main__":
    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset_train = CocoDataset(
        root="../../data/coco_minitrain_25k/images_pruned/val2017",
        annFile="../../data/coco_minitrain_25k/annotations/instances_val2017_pruned.json",
        transform=image_transform,
    )
    data_loader = DataLoader(dataset=dataset_train, batch_size=2,
                             shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    for images, targets in tqdm(data_loader):
        image = images[0]  # Assuming batch size is 1
        target = targets[0]
        labels = target['labels']
        masks = target['masks']
        plot_segmentation(image, masks)

