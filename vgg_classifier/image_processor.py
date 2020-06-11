import os

import torch
from torchvision import datasets, transforms


class DataTransform():
    def __init__(self, data_dir, _batch_size=4, _shuffle=True, _num_workers=4):
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                          for x in ['train', 'valid', 'test']}

        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=_batch_size,
                                                           shuffle=_shuffle, num_workers=_num_workers)
                            for x in ['train', 'valid', 'test']}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        self.class_names = image_datasets['train'].classes
