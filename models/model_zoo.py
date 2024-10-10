import torch
import torch.nn as nn
import numpy as np

from torchvision import models, transforms

# downloaded models are saved in this directory
torch.hub.set_dir('./models')

class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(args, backbone_name="resnet18_cub", full_model=False):


    if "clip:" in backbone_name:
        import clip
        tmp = args.backbone_name.split('-')
        backbone_name = '-'.join(tmp[:-1])+'/'+tmp[-1]

        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = backbone_name.split(":")[1]
        backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.out_dir)
        backbone = backbone.eval()
        model = None
    elif backbone_name == 'robustclip':
        # adversarially robust CLIP: https://github.com/chs20/RobustVLM?tab=readme-ov-file
        import open_clip
        backbone, _, preprocess = open_clip.create_model_and_transforms('hf-hub:chs20/fare2-clip')
        backbone = backbone.eval()
        model = None

    elif backbone_name == "altclip":
        from transformers import AltCLIPModel
        backbone = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        backbone = backbone.eval()
        model = None
    elif backbone_name == "align":
        from transformers import AlignProcessor, AlignModel
        preprocess = AlignProcessor.from_pretrained("kakaobrain/align-base")
        backbone = AlignModel.from_pretrained("kakaobrain/align-base")
        backbone = backbone.eval()
        model = None

    elif backbone_name == "medclip":
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
        backbone = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        backbone.from_pretrained()
        # preprocess = MedCLIPProcessor()
        preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        backbone = backbone.eval().to(args.device)
        model = None


    elif backbone_name == "resnet50":
        # for Metashift, Waterbirds datasets
        # output emb dim = 2048

        model = models.resnet50(weights="IMAGENET1K_V2")
        # d = model.fc.in_features
        # model.fc = nn.Linear(d, n_classes)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)

        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        preprocess = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale),
                               int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    elif backbone_name == "densenet121":
        # for FMoW dataset
        model = models.densenet121(pretrained=True)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        # d = model.classifier.in_features
        # model.classifier = nn.Linear(d, n_classes)
        print(backbone)

        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        preprocess = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale),
                               int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    elif backbone_name == "resnet50_isic":
        model = models.resnet50(weights="IMAGENET1K_V2")
        backbone, model_top = ResNetBottom(model), ResNetTop(model)

        target_resolution = (224, 224)
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        # TODO: parameterize these two
        train = False
        augment_data = False
        if train and augment_data:
            preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(target_resolution[0], scale=(0.75, 1.0)),
                transforms.RandomRotation(45),
                transforms.ColorJitter(hue=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(target_resolution),
                transforms.ColorJitter(hue=0.2),
                transforms.ToTensor(),
                normalize
            ])


    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess


