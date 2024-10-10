import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Subset, Dataset

class CIFARCDataset(Dataset):
    def __init__(self, data_type, corruption_type, severity, transform=None):
        """
        Prepare CIFAR10-C or CIFAR100-C data
        """
        start_idx = 10000*(severity-1)
        end_idx = 10000*severity
        self.images = np.load(f"datasets/{data_type}-C/{corruption_type}.npy")[start_idx:end_idx]
        self.labels = np.load(f"datasets/{data_type}-C/labels.npy")[start_idx:end_idx]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the sample and label at the given index.
        image = self.images[idx]
        label = self.labels[idx]

        # Apply the transformation, if provided.
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label




class CIFAR10CDataset(CIFARCDataset):
    def __init__(self, corruption_type, severity, transform=None):
        """
        Prepare CIFAR10-C data (with the highest severity level=5)
        """
        super().__init__("CIFAR-10", corruption_type, severity, transform)


class CIFAR100CDataset(CIFARCDataset):
    def __init__(self, corruption_type, severity, transform=None):
        """
        Prepare CIFAR10-C data (with the highest severity level=5)
        """
        super().__init__("CIFAR-100", corruption_type, severity, transform)
