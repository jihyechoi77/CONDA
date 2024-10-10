import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io

class TinyImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, val_dir, val_annotations, class_to_idx, transform=None):
        self.val_dir = val_dir
        self.val_annotations = val_annotations
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.images = list(val_annotations.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.val_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[self.val_annotations[img_name]]
        if self.transform:
            image = self.transform(image)
        return image, label



class CorruptedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            corruption_type (str): Type of corruption (e.g., 'glass_blur')
            severity (int): Severity level (1-5)
            root_dir (str): Root directory of Tiny ImageNet-C dataset
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        image_paths = []
        labels = []
        class_folders = os.listdir(self.root_dir)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        for cls_name in class_folders:
            cls_folder = os.path.join(self.root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    image_paths.append(img_path)
                    labels.append(class_to_idx[cls_name])

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


class TinyImageNetCDataset(CorruptedDataset):
    def __init__(self, corruption_type, severity, root_dir, transform=None):
        """
        Args:
            corruption_type (str): Type of corruption (e.g., 'glass_blur')
            severity (int): Severity level (1-5)
            root_dir (str): Root directory of Tiny ImageNet-C dataset
            transform (callable, optional): Optional transform to be applied on a sample
        """
        corrupted_dir = os.path.join(root_dir, corruption_type, str(severity))
        super().__init__(root_dir=corrupted_dir, transform=transform)


class ImageNetCDataset(CorruptedDataset):
    def __init__(self, corruption_type, severity, root_dir, transform=None):
        """
        Args:
            corruption_type (str): Type of corruption (e.g., 'motion_blur')
            severity (int): Severity level (1-5)
            root_dir (str): Root directory of ImageNet-C dataset
            transform (callable, optional): Optional transform to be applied on a sample
        """
        corrupted_dir = os.path.join(root_dir, corruption_type, str(severity))
        super().__init__(root_dir=corrupted_dir, transform=transform)


def get_imagenet_classnames(classes,
                            meta_file_path):
    # Load meta.mat to get WNID and human-readable names
    meta = scipy.io.loadmat(meta_file_path)

    # Extract the synsets
    synsets = meta['synsets']

    # Create mapping from WNID to human-readable class names
    wnid_to_name = {}

    # Loop over the synsets to build the mapping
    for synset in synsets:
        wnid = synset[0][1][0]  # WNID, e.g., 'n01440764'
        class_name = synset[0][2][0]  # Human-readable class name, e.g., 'tench, Tinca tinca'
        wnid_to_name[wnid] = class_name

    # Map the given WNIDs (classes) to their human-readable names
    names = [wnid_to_name[wnid] for wnid in classes]
    return names



