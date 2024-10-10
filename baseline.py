import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import clip

from data import get_dataset
from models import BaseLinear, get_model
from training_tools import AverageMeter
from medclip import MedCLIPProcessor


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results/waterbirds", help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="waterbirds", type=str)
    parser.add_argument("--dataset-shift", default="waterbirds_shift", type=str)
    parser.add_argument("--backbone-name", default="clip:ViT-B/32", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-epochs", default=20, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=2e-4, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--l2-penalty", default=0.001, type=float)

    return parser.parse_args()


@torch.no_grad()
def get_projections(args, backbone, loader):
    all_embs, all_lbls = None, None

    for data in tqdm(loader):
        if args.dataset == 'camelyon17':
            batch_X, batch_Y, metadata = data
        else:
            batch_X, batch_Y = data
        batch_X = batch_X.to(args.device)
        if "clip:" in args.backbone_name or args.backbone_name in ["medclip", "robustclip", "altclip"]:
            embeddings = backbone.encode_image(batch_X).detach().cpu().float()
        # elif args.backbone_name == 'altclip':
        #     from transformers import AltCLIPProcessor
        #     processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")
        #     inputs = processor(images=batch_X, return_tensors="pt")
        #     outputs = backbone(**inputs)
        #     embeddings = outputs.image_embeds
        # elif args.backbone_name == 'medclip':
        #     processor = MedCLIPProcessor()
        #     inputs = processor(text=["The given region of tissue is normal", "The given region of tissue contain atumor tissue"],
        #                        images=batch_X, return_tensors="pt", padding=True)
        #     outputs = backbone(**inputs)
        #     embeddings = outputs['img_embds']
        else:
            embeddings = backbone(batch_X).detach()
            embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
    return all_embs, all_lbls

@torch.no_grad()
def zeroshot_eval(args, backbone, processor, loader, dataset_name, classes):
    if args.dataset in ['cifar10', 'cifar100']:
        candidate_labels = [f"a photo of {c}" for c in classes]
    elif args.dataset == 'imagenet':
        candidate_labels = [f"a photo of {c.split(',')[0]}" for c in classes]
    elif 'camelyon' in args.dataset:
        # candidate_labels = ["The given region of tissue is normal.", "The histopathology slide shows a tumor tissue."] #[f"The histopathology slide shows {c}" for c in classes]
        candidate_labels = [f"a photo of {c}" for c in classes]
    else:
        candidate_labels = [f"an image of {c}" for c in classes]
    print("ZS prompts: ", candidate_labels[:5])
    # candidate_labels = [f"{c}" for c in classes]
    # candidate_labels = [f"an image of {c} skin lesion" for c in classes]
    texts = clip.tokenize(candidate_labels).cuda()


    all_preds, all_lbls = None, None
    for data in tqdm(loader):
        if args.dataset == 'camelyon17':
            batch_X, batch_Y, metadata = data
        else:
            batch_X, batch_Y = data
        batch_X = batch_X.to(args.device)

        if args.backbone_name == 'altclip':
            from transformers import AltCLIPProcessor
            processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            pil_images = [to_pil(image) for image in batch_X]
            inputs = processor(text=candidate_labels, images=pil_images, return_tensors="pt", padding=True)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}

            outputs = backbone(**inputs)

            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            preds = logits_per_image.softmax(dim=1).cpu().numpy()

        elif args.backbone_name == 'robustclip':
            image_features = backbone.encode_image(batch_X)
            text_features = backbone.encode_text(texts)

            # Normalize the embeddings before computing cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity between image and text features
            logits_per_image = (image_features @ text_features.T)

            # Get the zero-shot predictions using softmax
            preds = logits_per_image.softmax(dim=1).cpu().numpy()

        elif args.backbone_name == 'medclip':
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            pil_images = [to_pil(image) for image in batch_X]

            processor = MedCLIPProcessor()
            inputs = processor(text=candidate_labels, images=pil_images, return_tensors="pt", padding=True)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}

            outputs = backbone(**inputs)
            logits_per_image = outputs['logits'] # Shape: [batch_size, num_prompts]
            preds = logits_per_image.softmax(dim=1).cpu().numpy()  # Convert to probabilities

        else:
            # image-tex similarity score
            logits_per_image, logits_per_text = backbone(batch_X, texts)
            preds = logits_per_image.softmax(dim=1).cpu().numpy()

        if all_preds is None:
            all_preds = preds
            all_lbls = batch_Y.numpy()
        else:
            all_preds = np.concatenate([all_preds, preds], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)

    np.save(os.path.join(args.out_dir, f"{dataset_name}-test__preds_zs__{args.backbone_name}.npy"), all_preds)

    if args.dataset in ["isic", "camelyon17"]:
        auroc = roc_auc_score(all_lbls, all_preds.max(1))
        print(f"[Zero Shot] AUROC: {auroc}")
        # return all_preds, auroc

    cm = confusion_matrix(all_lbls, all_preds.argmax(1))
    diagonal = cm.diagonal()
    per_class_acc = diagonal / np.sum(cm, axis=1)
    print(f"[Zero Shot] Avg acc: {per_class_acc.mean()}, Worst acc: {per_class_acc.min()}")
    # return all_preds, per_class_acc.mean(), per_class_acc.min()


def train(args, train_loader, model, optimizer, save_path):
    """
    Args:
        loaders (dict of torch.utils.data.DataLoader): Training/test data of (embedding, label) pairs\
        posthoc_layer (models.BaseLinear, models.PosthocLinearCBM, or models.PosthocHybridCBM): layer following the backbone
        optimizer (torch.optim.Optimizer): Optimizer
        num_classes (int): Number of classes in the dataset
        type (str): Type of training. Can be "baseline", "posthoc" or "hybrid"
    """

    for epoch in range(1, args.num_epochs + 1):
        # print(f"Epoch: {epoch}")

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            optimizer.zero_grad()

            out = model(batch_X)

            loss = model.loss(out, batch_Y)

            loss.backward()
            optimizer.step()

    # torch.save(model, save_path)
    # print(f"Model saved to : {save_path}")


def get_group_acc(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds.argmax(1))
    diagonal = cm.diagonal()
    per_class_acc = diagonal / np.sum(cm, axis=1)

    return per_class_acc.mean(), per_class_acc.min()

@torch.no_grad()
def eval(args, test_loader, model):

    all_preds = []
    all_labels = []

    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
        out = model(batch_X)
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_labels, all_preds


def get_loader(X, Y, batch_size, shuffle=False):
    loader = DataLoader(TensorDataset(torch.tensor(X).float(), torch.tensor(Y).long()),
                                  batch_size=batch_size, shuffle=shuffle)
    return loader


def compute_or_load(args, backbone, loader, dataset, file_type, extension='npy'):
    filepath_emb = os.path.join(args.out_dir, f"{dataset}-{file_type}__embs__{args.backbone_name}.{extension}")
    filepath_lbl = os.path.join(args.out_dir, f"{dataset}-{file_type}__lbls__{args.backbone_name}.{extension}")

    if os.path.exists(filepath_emb) and os.path.exists(filepath_lbl):
        print("already exists... loading")
        embs = np.load(filepath_emb)
        lbls = np.load(filepath_lbl)
    else:
        print("computing embeddings...")
        embs, lbls = get_projections(args, backbone, loader)
        np.save(filepath_emb, embs)
        np.save(filepath_lbl, lbls)

    return embs, lbls



def main(args, backbone, preprocess):

    train_loader, test_loader, idx_to_class, classes = get_dataset(args, args.dataset, preprocess)
    _, test_shift_loader, _, _ = get_dataset(args, args.dataset_shift, preprocess)

    num_classes = len(classes)

    os.makedirs(args.out_dir, exist_ok=True)

    train_embs, train_lbls = compute_or_load(args, backbone, train_loader, args.dataset, file_type='train', extension='npy')
    print(np.sum(train_lbls==0), np.sum(train_lbls==1))
    test_embs, test_lbls = compute_or_load(args, backbone, test_loader, args.dataset, file_type='test', extension='npy')
    test_shift_embs, test_shift_lbls = compute_or_load(args, backbone, test_shift_loader, args.dataset_shift, file_type='test', extension='npy')

    zeroshot_eval(args, backbone, preprocess, test_loader, args.dataset, classes)
    zeroshot_eval(args, backbone, preprocess, test_shift_loader, args.dataset_shift, classes)

    dim_emb = train_embs.shape[1]

    if args.dataset == 'camelyon17':
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_indices, test_indices = next(split.split(train_lbls, train_lbls))
        test_embs, test_lbls = train_embs[test_indices], train_lbls[test_indices]
        train_embs, train_lbls = train_embs[train_indices], train_lbls[train_indices]

    num_experiments = 1

    #######################
    # run linear probe with embeddings (Baseline)

    avg_all, worst_all = [], []
    avg_shift_all, worst_shift_all = [], []
    auroc_all, auroc_shift_all = [], []
    for seed in tqdm(range(num_experiments)):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


        train_loader = get_loader(train_embs, train_lbls, args.batch_size, shuffle=True)
        test_loader = get_loader(test_embs, test_lbls, args.batch_size, shuffle=False)
        test_shift_loader = get_loader(test_shift_embs, test_shift_lbls, args.batch_size, shuffle=False)


        base_path = os.path.join(args.out_dir, f"baseline_{args.dataset}__{args.backbone_name}__seed{seed}.ckpt")

        base_probe = BaseLinear(dim_emb, num_classes)
        base_probe = base_probe.to(args.device)

        # base_optimizer = torch.optim.Adam(base_probe.classifier.parameters(), lr=args.lr)
        base_optimizer = torch.optim.SGD(base_probe.trainable_params(), lr=args.lr)
        base_probe.classifier = base_probe.classifier.float()
        train(args, train_loader, base_probe, base_optimizer, save_path=base_path)


        if args.dataset == "camelyon17":
            test_labels, test_preds_lp = eval(args, test_loader, base_probe)
            auroc = roc_auc_score(test_labels, test_preds_lp.max(1))
            test_shift_labels, test_shift_preds_lp = eval(args, test_shift_loader, base_probe)
            auroc_shift = roc_auc_score(test_shift_labels, test_shift_preds_lp.max(1))

            auroc_all.append(auroc)
            auroc_shift_all.append(auroc_shift)

        _, test_preds_lp = eval(args, test_loader, base_probe)
        avg, worst = get_group_acc(test_lbls, test_preds_lp)
        _, test_shift_preds_lp = eval(args, test_shift_loader, base_probe)
        avg_shift, worst_shift = get_group_acc(test_shift_lbls, test_shift_preds_lp)

        avg_all.append(avg)
        worst_all.append(worst)
        avg_shift_all.append(avg_shift)
        worst_shift_all.append(worst_shift)

        np.save(os.path.join(args.out_dir, f"{args.dataset_shift}-test__preds_lp__{args.backbone_name}_seed{seed}.npy"), test_shift_preds_lp)

    print(f"[Baseline] Over {num_experiments} trials...")
    print("[Source test] AUROC: {:.5f} +/- {:.5f}".format(np.mean(auroc_all),
                                                                     np.std(auroc_all, ddof=1) / np.sqrt(len(auroc_all))))
    print("[Target test] AUROC: {:.5f} +/- {:.5f}".format(np.mean(auroc_shift_all),
                                                                     np.std(auroc_shift_all, ddof=1) / np.sqrt(
                                                                         len(auroc_shift_all))))

    print("[Source test] Average Accuracy: {:.5f} +/- {:.5f}".format(np.mean(avg_all), np.std(avg_all, ddof=1)/np.sqrt(len(avg_all))))
    print("[Source test] Worst Accuracy: {:.5f} +/- {:.5f}".format(np.mean(worst_all), np.std(worst_all, ddof=1)/np.sqrt(len(worst_all))))
    print("[Target test] Average Accuracy: {:.5f} +/- {:.5f}".format(np.mean(avg_shift_all), np.std(avg_shift_all, ddof=1)/np.sqrt(len(avg_shift_all))))
    print("[Target test] Worst Accuracy: {:.5f} +/- {:.5f}".format(np.mean(worst_shift_all), np.std(worst_shift_all, ddof=1)/np.sqrt(len(worst_shift_all))))


if __name__ == "__main__":
    args = config()

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, backbone, preprocess)
