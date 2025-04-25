from torchvision import datasets
from data.cifar10c import CIFAR10CDataset, CIFAR100CDataset

import torch
import os
import random
import pandas as pd
import numpy as np
from .data_utils import FilepathDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def get_dataset(args, dataset, preprocess=None, seed=42, shuffle_test=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # g = torch.Generator()
    # g.manual_seed(args.seed)


    if dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR10(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

    elif "cifar10-c" in dataset:
        # reference: https://github.com/DequanWang/tent/blob/master/cifar10c.py#L44C13-L46C62

        tags = dataset.split("-")
        corruption_list = tags[2]
        # corruption_list = [t for t in tags[2:-1]]
        severity = int(tags[-1])
        testset = CIFAR10CDataset(corruption_list, severity, preprocess)

        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,)
        train_loader = idx_to_class = classes = None


    elif dataset == "cifar100":
        trainset = datasets.CIFAR100(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR100(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)


    elif "cifar100-c" in dataset:
        print(dataset)
        tags = dataset.split("-")
        corruption_list = tags[2]
        # corruption_list = [t for t in tags[2:-1]]
        severity = int(tags[-1])
        testset = CIFAR100CDataset(corruption_list, severity, preprocess)

        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,)
        train_loader = idx_to_class = classes = None


    elif dataset == 'camelyon17':
        from wilds import get_dataset
        from wilds.common.data_loaders import get_eval_loader
        dataset = get_dataset(dataset='camelyon17', download=True, root_dir="./datasets")
        train_data = dataset.get_subset('train', transform=preprocess)
        val_data = dataset.get_subset('val', transform=preprocess)

        # DataLoaders for train and test subsets
        train_loader = get_eval_loader("standard", train_data, batch_size=args.batch_size)
        test_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size)

        # classes = train_data._dataset.classes
        classes = ["normal tissue", "tumor tissue"]
        class_to_idx = {c: i for (i, c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}


    elif dataset == 'camelyon17-shift':
        from wilds import get_dataset
        from wilds.common.data_loaders import get_eval_loader

        dataset = get_dataset(dataset='camelyon17', download=True, root_dir="./datasets")
        test_data = dataset.get_subset('test', transform=preprocess)
        test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)

        train_loader = None
        classes = None
        idx_to_class = None


    elif dataset == 'metashift':
        from .constants import METASHIFT_DATA_DIR
        trainset = datasets.ImageFolder(root=os.path.join(METASHIFT_DATA_DIR,'train'),
                                        transform=preprocess)
        testset = datasets.ImageFolder(root=os.path.join(METASHIFT_DATA_DIR, 'test'),
                                       transform=preprocess)
        classes = testset.classes
        class_to_idx = testset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,)

    elif dataset == 'metashift_shift':
        from .constants import METASHIFT_DATA_DIR
        testset = datasets.ImageFolder(root=os.path.join(METASHIFT_DATA_DIR, 'test_shift'),
                                       transform=preprocess)
        classes = testset.classes
        class_to_idx = testset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,)




    elif 'waterbirds' in dataset:
        from .constants import WATERBIRDS_DATA_DIR
        metadata_df = pd.read_csv(os.path.join(WATERBIRDS_DATA_DIR, 'metadata.csv'))

        # Get necessary information from metadata
        labels = metadata_df['y'].values
        label_dict = {'waterbird': 1, 'landbird': 0}
        confounders = metadata_df['place'].values
        confounder_dict = {'water': 1, 'land': 0}
        filenames = np.array([os.path.join(WATERBIRDS_DATA_DIR, f) for f in metadata_df['img_filename'].values])
        split_array = metadata_df['split'].values
        split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        is_train = split_array == split_dict['train']
        is_test = split_array == split_dict['test']
        is_land = confounders == confounder_dict['land']
        is_landbird = labels == label_dict['landbird']

        # Get the train and test splits
        idx_train = is_train * ((is_landbird * is_land) + (~is_landbird * ~is_land))
        if dataset == 'waterbirds_shift':
            idx_test = is_test * ((is_landbird * ~is_land) + (~is_landbird * is_land))
        else: # 'waterbirds' or 'waterbirds-c'
            idx_test = is_test * ((is_landbird * is_land) + (~is_landbird * ~is_land))

        trainset = FilepathDataset(filenames[idx_train], labels[idx_train], transform=preprocess)
        testset = FilepathDataset(filenames[idx_test], labels[idx_test], transform=preprocess)


        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.num_workers,
                                                   worker_init_fn=seed_worker,)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  worker_init_fn=seed_worker,
                                                  # generator=g,
                                                  )
        classes = ["landbird", "waterbird"]
        class_to_idx = {c: i for (i, c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}


    else:
        raise ValueError(dataset)



    return train_loader, test_loader, idx_to_class, classes

