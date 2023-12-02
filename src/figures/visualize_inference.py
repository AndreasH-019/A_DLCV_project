import sys
sys.path.append('copy-paste')
sys.path.append("segm")
from src.segm.model import LitMaskRCNN
from src.segm.coco_dataset import get_segmentation_image, get_bounding_box_and_segmentation_image
from tqdm import tqdm

# from torch.utils.data.dataloader import DataLoader
# from model import get_model, LitMaskRCNN
import lightning as L
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


checkpoint_paths = ["lightning_logs/no copy paste/checkpoints/best.ckpt",
                    "lightning_logs/real copy paste/checkpoints/best.ckpt",
                    "lightning_logs/gen copy paste/checkpoints/best.ckpt",
                    "lightning_logs/genreal copy paste/checkpoints/best.ckpt"]
titles = ["Ground truth", "No copy paste", "Paste from real", "Paste from generated", "Paste from both"]
assert len(titles) == len(checkpoint_paths)+1
models = [LitMaskRCNN.load_from_checkpoint(checkpoint_path=checkpoint_path).eval() for checkpoint_path in checkpoint_paths]
dataloader = models[0].test_dataloader()
dataset = dataloader.dataset
segmentation_images = []
# plot_idxs = [18, 65, 20]
# plot_idxs = [53, 175, 128]
# plot_idxs = [175, 65, 20]
plot_idxs = [175, 65, 71]
assert len(plot_idxs) == 3
for i in tqdm(plot_idxs):
    image, target = dataset[i]
    images, targets = [image], [target]
    # segmentation_image = get_segmentation_image(images[0], targets[0]['masks'], targets[0]['labels'])
    segmentation_image = get_bounding_box_and_segmentation_image(images[0], targets[0]['boxes'],
                                                                 targets[0]['masks'], targets[0]['labels'])
    segmentation_images.append(segmentation_image)
    # segmentation_images.append(segmentation_image)
    # segmentation_images.append(segmentation_image)
    # segmentation_images.append(segmentation_image)
    for model in models:
        outputs = model(images)
        for output in outputs:
            output["masks"] = output["masks"] > 0.5
            output["masks"] = output["masks"].squeeze(1)
        # segmentation_image = get_segmentation_image(images[0], outputs[0]['masks'],
        #                                             outputs[0]['labels'], outputs[0]['scores'])
        segmentation_image = get_bounding_box_and_segmentation_image(images[0], outputs[0]['boxes'],
                                                                     outputs[0]['masks'], outputs[0]['labels'],
                                                                     outputs[0]['scores'])
        segmentation_images.append(segmentation_image)

fig, ax = plt.subplots(3, 5, figsize=(11, 8))
ax = ax.flatten()
for i in range(len(ax)):
    F.to_pil_image(segmentation_images[i])
    ax[i].imshow(F.to_pil_image(segmentation_images[i]))
    ax[i].axis('off')
for i in range(len(titles)):
    ax[i].set_title(titles[i])
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.subplots_adjust(hspace=0)
plt.tight_layout()
# plt.subplots_adjust(wspace=0)
plt.savefig("figures/inference.jpg")
# for i in range(3):
#     for j in range(3):
#         ax[i, j].imshow(segmentation_image)

