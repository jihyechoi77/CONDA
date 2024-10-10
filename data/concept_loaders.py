import os
import pickle
import torch

import pandas as pd
import numpy as np
import json
from PIL import Image
from torch.utils.data import DataLoader

class ListDataset:
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        return image


def broden_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed):
    from .constants import BRODEN_CONCEPTS
    concept_loaders = {}
    concepts = [c for c in os.listdir(BRODEN_CONCEPTS) if os.path.isdir(os.path.join(BRODEN_CONCEPTS, c))]
    for concept_name in concepts:
        pos_dir = os.path.join(BRODEN_CONCEPTS, concept_name, "positives")
        pos_images = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]
        if (len(pos_images) < 2*n_samples):
            print(f"\t Not enough positive samples for {concept_name}: {len(pos_images)}! Sampling with replacement")
            pos_images = np.random.choice(pos_images, 2*n_samples, replace=True)
        else:
            pos_images = np.random.choice(pos_images, 2*n_samples, replace=False)
        neg_dir = os.path.join(BRODEN_CONCEPTS, concept_name, "negatives")
        neg_images = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]
        if (len(neg_images) < 2*n_samples):
            print(f"\t Not enough negative samples for {concept_name}: {len(neg_images)}! Sampling with replacement")
            neg_images = np.random.choice(neg_images, 2*n_samples, replace=True)
        else:
            neg_images = np.random.choice(neg_images, 2*n_samples, replace=False)

        pos_ds = ListDataset(pos_images, transform=preprocess)
        neg_ds = ListDataset(neg_images, transform=preprocess)
        pos_loader = DataLoader(pos_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

        neg_loader = DataLoader(neg_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        concept_loaders[concept_name] = {
            "pos": pos_loader,
            "neg": neg_loader
        }
    return concept_loaders


def synthetic_concept_loaders(preprocess, n_samples, batch_size, num_workers, concept_categories):
    from .constants import SYNTHETIC_CONCEPTS
    concept_loaders = {}
    # concepts = [c for c in os.listdir(SYNTHETIC_CONCEPTS) if os.path.isdir(os.path.join(SYNTHETIC_CONCEPTS, c))]
    metadata = json.load(open('./concepts/concept_metadata.json'))
    concepts = [item for category in concept_categories for item in metadata[category]]

    for concept_name in concepts:
        # Load positive images
        pos_dir = os.path.join(SYNTHETIC_CONCEPTS, concept_name, "positives")
        pos_images = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]

        if (len(pos_images) < n_samples):
            print(f"\t Not enough positive samples for {concept_name}: {len(pos_images)}! Sampling with replacement")
            pos_images = np.random.choice(pos_images, n_samples, replace=True)
        else:
            pos_images = np.random.choice(pos_images, n_samples, replace=False)

        # Construct negative images by sampling
        M = int(max(len(pos_images) // len(concepts), 0)) + 1
        neg_images = []
        for neg_concept in concepts:
            if neg_concept == concept_name: continue
            indices = np.random.choice(len(pos_images), M)
            neg_dir = os.path.join(SYNTHETIC_CONCEPTS, neg_concept, "positives")
            _neg_images = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]
            neg_samples = [_neg_images[index] for index in indices]
            neg_images.extend(neg_samples)

        pos_ds = ListDataset(pos_images, transform=preprocess)
        neg_ds = ListDataset(neg_images, transform=preprocess)
        pos_loader = DataLoader(pos_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

        neg_loader = DataLoader(neg_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        concept_loaders[concept_name] = {
            "pos": pos_loader,
            "neg": neg_loader
        }

    return concept_loaders


def get_concept_loaders(dataset_name, preprocess, n_samples=50, batch_size=100, num_workers=4, seed=1):
    if dataset_name == "waterbirds":
       return synthetic_concept_loaders(preprocess, n_samples, batch_size, num_workers,
                                        concept_categories=['nature', 'color', 'texture'])

    elif dataset_name in ['camelyon17', 'chexpert']:
        return synthetic_concept_loaders(preprocess, n_samples, batch_size, num_workers,
                                        concept_categories=['color', 'texture'])

    elif dataset_name == 'metashift':
        return synthetic_concept_loaders(preprocess, n_samples, batch_size, num_workers,
                                        concept_categories=['nature', 'color', 'texture', 'city', 'household', 'others'])
    
    elif dataset_name == "cifar10":
        return broden_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed)

    elif dataset_name == "broden":
        return broden_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    