import torchvision.transforms as transforms

"""
Transformation functions that preprocess datasets
"""


def get_transform_cub(model_type, train, augment_data):
    scale = 256.0 / 224.0
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    target_resolution = model_attributes[model_type]['target_resolution']
    assert target_resolution is not None
    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale),
                               int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    return transform


def get_transform_ISIC(model_type, train, augment_data):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    target_resolution = model_attributes[model_type]['target_resolution']
    if train and augment_data:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(target_resolution[0], scale=(0.75, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(hue=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ColorJitter(hue=0.2),
            transforms.ToTensor(),
            normalize
        ])
    return transform


def get_transform_CIFAR10(model_type, train, augment_data):
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])
    if train and augment_data:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
    return transform


model_attributes = {
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'densenet121':{
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    }
}


transform_dict = {
    'CUB': get_transform_cub,
    'MetaDatasetCatDog': get_transform_cub,
    'FMoW': get_transform_cub,
    'ISIC': get_transform_ISIC,
    'CIFAR10': get_transform_CIFAR10
}

