import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from data import get_dataset
from models import PosthocLinearCBM, get_model
from concepts import ConceptBank



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results/waterbirds", help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="waterbirds", type=str)
    parser.add_argument("--dataset-shift", default="waterbirds_shift", type=str)
    parser.add_argument("--backbone-name", default="clip:ViT-B/32", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    # parser.add_argument("--seed", default=42, type=int, help="Random seed")
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
    for batch_X, batch_Y in tqdm(loader):
        batch_X = batch_X.to(args.device)
        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(batch_X).detach().float()
        else:
            embeddings = backbone(batch_X).detach()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
    return all_embs, all_lbls


def train(args, train_loader, model, optimizer, save_path):
    """
    Args:
        loaders (dict of torch.utils.data.DataLoader): Training/test data of (embedding, label) pairs\
        posthoc_layer (models.BaseLinear, models.PosthocLinearCBM, or models.PosthocHybridCBM): layer following the backbone
        optimizer (torch.optim.Optimizer): Optimizer
        num_classes (int): Number of classes in the dataset
        type (str): Type of training. Can be "baseline", "posthoc" or "hybrid"
    """

    for epoch in tqdm(range(1, args.num_epochs + 1)):
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

@torch.no_grad()
def eval(args, test_loader, model):

    all_preds = []
    all_labels = []
    all_projs = []

    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
        out, projs = model(batch_X, return_concept=True)
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())
        all_projs.append(projs.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_projs = np.concatenate(all_projs, axis=0)

    cm = confusion_matrix(all_labels, all_preds.argmax(1))
    diagonal = cm.diagonal()
    per_class_acc = diagonal / np.sum(cm, axis=1)

    print(all_preds[:10])

    return per_class_acc.mean(), per_class_acc.min(), all_projs


def get_group_acc(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds.argmax(1))
    diagonal = cm.diagonal()
    per_class_acc = diagonal / np.sum(cm, axis=1)

    return per_class_acc.mean(), per_class_acc.min()


def run_inference(args, loader, model, num_epochs=1):

    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        all_projs = []
        all_groundtruths = []

        for batch_X, batch_Y in loader:
            if len(batch_Y.shape) > 1:
                all_groundtruths += list(batch_Y[:,0].numpy())
                batch_Y = batch_Y[:,1:]

            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            out, projs = model(batch_X, return_concept=True)

            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(batch_Y.detach().cpu().numpy())
            all_projs.append(projs.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_projs = np.concatenate(all_projs, axis=0)

    return all_groundtruths, all_labels, all_preds, all_projs


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
    test_embs, test_lbls = compute_or_load(args, backbone, test_loader, args.dataset, file_type='test', extension='npy')
    test_shift_embs, test_shift_lbls = compute_or_load(args, backbone, test_shift_loader, args.dataset_shift, file_type='test', extension='npy')

    dim_emb = train_embs.shape[1]

    num_experiments = 1

    #######################
    # run CBM

    avg_all, worst_all = [], []
    avg_shift_all, worst_shift_all = [], []
    for seed in tqdm(range(num_experiments)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        train_loader = get_loader(train_embs, train_lbls, args.batch_size, shuffle=True)
        test_loader = get_loader(test_embs, test_lbls, args.batch_size, shuffle=False)
        test_shift_loader = get_loader(test_shift_embs, test_shift_lbls, args.batch_size, shuffle=False)

        cbm_path = os.path.join(args.out_dir, f"pcbm_{args.dataset}__{args.backbone_name}__seed{seed}.ckpt")

        # load concept bank
        all_concepts = pickle.load(open(os.path.join('datasets/concepts',f'{args.dataset}_{args.backbone_name}_0.1_150_seed{seed}.pkl'), 'rb'))
        all_concept_names = list(all_concepts.keys())
        concept_bank = ConceptBank(all_concepts, args.device)


        # define CBM
        cbm = PosthocLinearCBM(concept_bank,
                               backbone_name=args.backbone_name,
                               idx_to_class=idx_to_class,
                               n_classes=num_classes,
                               alpha=args.alpha, lam=args.lam,
                               dim_emb=dim_emb)


        cbm = cbm.to(args.device)
        print(f"using {cbm.n_concepts} concepts...")

        optimizer = torch.optim.Adam(cbm.trainable_params(), lr=args.lr)
        cbm.classifier = cbm.classifier.float()
        train(args, train_loader, cbm, optimizer, save_path=cbm_path)
        avg, worst, projs = eval(args, test_loader, cbm)
        avg_shift, worst_shift, projs_shift = eval(args, test_shift_loader, cbm)

        # _, _, preds, projs = run_inference(args, test_loader, cbm)
        # avg, worst = get_group_acc(test_lbls, preds)

        avg_all.append(avg)
        worst_all.append(worst)
        avg_shift_all.append(avg_shift)
        worst_shift_all.append(worst_shift)

        # np.save(os.path.join(args.out_dir, f"{args.dataset}-test__projs__{args.backbone_name}__pcbm__seed{seed}.npy"),
        #         projs)
        # np.save(
        #     os.path.join(args.out_dir, f"{args.dataset_shift}-test__projs__{args.backbone_name}__pcbm__seed{seed}.npy"),
        #     projs_shift)

    print(f"[CBM] Over {num_experiments} trials...")
    print("[Source test] Average Accuracy: {:.5f} +/- {:.5f}".format(np.mean(avg_all),
                                                                     np.std(avg_all, ddof=1) / np.sqrt(len(avg_all))))
    print("[Source test] Worst Accuracy: {:.5f} +/- {:.5f}".format(np.mean(worst_all),
                                                                   np.std(worst_all, ddof=1) / np.sqrt(len(worst_all))))
    print("[Target test] Average Accuracy: {:.5f} +/- {:.5f}".format(np.mean(avg_shift_all),
                                                                     np.std(avg_shift_all, ddof=1) / np.sqrt(
                                                                         len(avg_shift_all))))
    print("[Target test] Worst Accuracy: {:.5f} +/- {:.5f}".format(np.mean(worst_shift_all),
                                                                   np.std(worst_shift_all, ddof=1) / np.sqrt(
                                                                       len(worst_shift_all))))

    # np.save(os.path.join(args.out_dir, f"{args.dataset}-test.npy"}), projs)
    # np.save(, projs_shift)


if __name__ == "__main__":
    args = config()

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, backbone, preprocess)
