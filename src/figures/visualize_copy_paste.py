# from src.segm.model import get_paste_transform
import sys
sys.path.append("copy-paste")
sys.path.append("segm")
from src.segm.model import get_paste_transform
from coco_paste_dataset import CocoDetectionCP, CopyPasteTrain
import matplotlib.pyplot as plt

task = 'train'
transform, paste_transforms = get_paste_transform(task, 'gen')
dataset = CocoDetectionCP(
            f'../data/coco_minitrain_25k/images_pruned/{task}2017',
            f'../data/coco_minitrain_25k/annotations/instances_{task}2017_pruned.json',
            transform,
            paste_transforms,
            load_from_generated=True
        )

img_data = dataset[0]
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax[0].imshow(img_data['paste_image_before_transforms'])
ax[1].imshow(img_data['paste_image'])
ax[2].imshow(img_data['image'])
for i in range(len(ax)):
    ax[i].axis('off')

plt.tight_layout()
plt.savefig("figures/paste_transforms.jpg")