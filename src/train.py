import lightning as L
from coco_dataset import CocoDataset, custom_collate_fn
import torchvision
from torch.utils.data.dataloader import DataLoader, Dataset
from model import get_model
import random


image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize([256, 256]),
                                    torchvision.transforms.ToTensor()])
datasets = {'train': CocoDataset(root="../../data/coco_minitrain_25k/images_pruned/train2017",
                                 annFile="../../data/coco_minitrain_25k/annotations/instances_train2017_pruned.json",
                                 transform=image_transform)
            }
datasets['train'].ids = random.sample(datasets['train'].ids, 2)
dataloaders = {'train': DataLoader(dataset=datasets['train'], batch_size=2,
                                   shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
               }
pl_model = get_model()
trainer = L.Trainer(max_epochs=1)
# trainer.fit(model=pl_model, train_dataloaders=dataloaders['train'])
trainer.fit(model=pl_model)