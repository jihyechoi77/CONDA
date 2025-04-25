import argparse
import os
import subprocess
import pickle
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import get_dataset
from models import PosthocLinearCBM, get_model
from concepts import ConceptBank
from adaptation import (Adaptation, sample_estimator)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results/waterbirds", help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="waterbirds", type=str)
    parser.add_argument("--dataset-shift", default="waterbirds_shift", type=str)
    parser.add_argument("--backbone-name", default="clip:ViT-B/32", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    # parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-epochs", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=2e-3, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--l2-penalty", default=0.001, type=float)

    ## concept learning
    parser.add_argument("--concept-method", default="pcbm", type=str, help="concept learning method")
    parser.add_argument("--num-concepts", default=0, type=int)

    ## adaptation
    # parser.add_argument("--adapt-method", default="crossentropy", type=str, help="loss type for test-time adaptation")
    parser.add_argument("--adapt-cavs", action="store_true", help="whether to only adapt the concept bottleneck")
    parser.add_argument("--adapt-classifier", action="store_true", help="whether to only adapt the classifier")
    parser.add_argument("--adapt-steps", default=10, type=int)
    parser.add_argument("--episodic", action="store_true",
                        help="Whether to use episodic reset at the start of every batch")
    parser.add_argument("--label-ensemble", action="store_true", help="whether to use ensemble of multiple reference models for psueo-labeling")
    parser.add_argument("--num-residual-concepts", default=5, type=int, help="number of additional concepts in residual concept bottleneck")
    # parser.add_argument("--rcbm-joint", action="store_true", help="whether to use residual concept bottleneck (joint)")
    parser.add_argument("--rcbm", action="store_true", help="whether to use residual concept bottleneck")

    return parser.parse_args()



def train(args, train_loader, model, optimizer):
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

            if args.concept_method == 'yeh':
                out, projs = model(batch_X, return_concept=True)
                loss = model.loss(out, batch_Y, projs)
            else:
                out = model(batch_X)
                loss = model.loss(out, batch_Y)

            loss.backward()
            optimizer.step()



def get_group_acc(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds.argmax(1))
    diagonal = cm.diagonal()
    per_class_acc = diagonal / np.sum(cm, axis=1)

    return per_class_acc.mean(), per_class_acc.min()



# @torch.no_grad()
def run_inference(args, loader, model, num_epochs=1,
                  adaptation=False,
                  adapt_cavs=False, adapt_classifier=False, adapt_rcb=False):

    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        all_projs = []
        all_projs_res = []
        all_groundtruths = []

        for batch_X, batch_Y in loader:
            if len(batch_Y.shape) > 1:
                all_groundtruths += list(batch_Y[:,0].numpy())
                batch_Y = batch_Y[:,1:]

            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            if not adaptation:
                out, projs = model(batch_X, return_concept=True)
            else:
                out, projs, projs_res = model(batch_X, batch_Y,
                                        adapt_cavs=adapt_cavs, adapt_classifier=adapt_classifier, adapt_rcb=adapt_rcb,
                                        return_concept=True)
            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(batch_Y.detach().cpu().numpy())
            all_projs.append(projs.detach().cpu().numpy())
            if adapt_rcb:
                all_projs_res.append(projs_res.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_projs = np.concatenate(all_projs, axis=0)
        if adapt_rcb:
            all_projs_res = np.concatenate(all_projs_res, axis=0)

    return all_groundtruths, all_labels, all_preds, all_projs, all_projs_res


def prepare_adapt(args, cbm, jointly=False):
    cbm.eval()

    cbm_adapt = Adaptation(cbm,
                           lr=args.lr,
                           steps=args.adapt_steps,
                           episodic=args.episodic,
                           num_res_concepts=args.num_residual_concepts,
                           jointly=jointly)

    cbm_adapt.to(args.device)

    return cbm_adapt


def pseudo_labeling(all_labels, ensemble=True):
    Y, Y_zs, Y_lp, Y_cbm = all_labels
    # reserve Y for evaluation later
    if ensemble:
        # labels = Y
        stacked_maxes = np.stack((Y_zs.max(1), Y_lp.max(1)), axis=-1)
        higher_logit_indices = np.argmax(stacked_maxes, axis=-1)
        # Select the argmax of Y1 or Y2 depending on which had the higher max logit
        labels = np.where(higher_logit_indices == 0, Y_zs.argmax(axis=1), Y_lp.argmax(axis=1))

        # stacked_maxes = np.stack((Y_zs.max(axis=1), Y_lp.max(axis=1), Y_cbm.max(axis=1)), axis=-1)
        # # Find the indices of the maximum logits across the three models
        # higher_logit_indices = np.argmax(stacked_maxes, axis=-1)
        # # Choose the labels corresponding to the model with the highest logit for each sample
        # labels = np.where(higher_logit_indices == 0, Y_zs.argmax(axis=1),
        #                   np.where(higher_logit_indices == 1, Y_lp.argmax(axis=1), Y_cbm.argmax(axis=1)))

        return np.c_[Y, labels]

    else:
        return np.c_[Y, Y_zs.argmax(1)]


def get_loader(X, Y, batch_size, shuffle=False):
    loader = DataLoader(TensorDataset(torch.tensor(X).float(), torch.tensor(Y).long()),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        )
    return loader


def load(args, dataset, file_type, extension='npy'):
    filepath_emb = os.path.join(args.out_dir, f"{dataset}-{file_type}__embs__{args.backbone_name}.{extension}")
    filepath_lbl = os.path.join(args.out_dir, f"{dataset}-{file_type}__lbls__{args.backbone_name}.{extension}")

    print("already exists... loading")
    embs = np.load(filepath_emb)
    lbls = np.load(filepath_lbl)

    return embs, lbls


def prepare_cbm(args, seed, idx_to_class, num_classes, dim_emb):
    cbm_path = os.path.join(args.out_dir, f"{args.concept_method}_{args.dataset}__{args.backbone_name}__seed{seed}.ckpt")
    if args.concept_method == "pcbm" or args.concept_method == "pcbm-h":
        # load concept bank
        all_concepts = pickle.load(
            open(os.path.join('datasets/concepts', f'{args.dataset}_{args.backbone_name}_0.1_150_seed{seed}.pkl'), 'rb'))
        all_concept_names = list(all_concepts.keys())
        concept_bank = ConceptBank(all_concepts, args.device)

        # define CBM
        cbm = PosthocLinearCBM(concept_bank,
                               backbone_name=args.backbone_name,
                               idx_to_class=idx_to_class,
                               n_classes=num_classes,
                               alpha=args.alpha, lam=args.lam,
                               dim_emb=dim_emb)

    elif args.concept_method == "yeh":
        # define CBM
        cbm = UnsupervisedCBM(backbone_name=args.backbone_name,
                              idx_to_class=idx_to_class,
                              n_classes=num_classes,
                              dim_emb=dim_emb,
                              n_concepts=args.num_concepts,
                              method=args.concept_method)
    elif args.concept_method == "label-free":
        # define CBM
        cbm = LabelFreeCBM(backbone_name=args.backbone_name,
                              idx_to_class=idx_to_class,
                              n_classes=num_classes,
                              dim_emb=dim_emb,
                              n_concepts=args.num_concepts,
                              method=args.concept_method)

    cbm = cbm.to(args.device)
    print(f"using {cbm.n_concepts} concepts...")

    return cbm, cbm_path


def main(args, backbone, preprocess):

    train_loader, test_loader, idx_to_class, classes = get_dataset(args, args.dataset, preprocess)
    _, test_shift_loader, _, _ = get_dataset(args, args.dataset_shift, preprocess)
    num_classes = len(classes)

    os.makedirs(args.out_dir, exist_ok=True)

    train_embs, train_lbls = load(args, args.dataset, file_type='train', extension='npy')
    test_embs, test_lbls = load(args, args.dataset, file_type='test', extension='npy')
    test_shift_embs, test_shift_lbls = load(args, args.dataset_shift, file_type='test', extension='npy')
    test_shift_lbls_zs = np.load(os.path.join(args.out_dir, f'{args.dataset_shift}-test__preds_zs__{args.backbone_name}.npy'))

    dim_emb = train_embs.shape[1]

    if args.dataset == 'camelyon17':
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_indices, test_indices = next(split.split(train_lbls, train_lbls))
        test_embs, test_lbls = train_embs[test_indices], train_lbls[test_indices]
        train_embs, train_lbls = train_embs[train_indices], train_lbls[train_indices]

    num_experiments = 10

    #######################
    # run CBM

    avg_all, worst_all = [], []
    avg_shift_all, worst_shift_all = [], []
    avg_shift_adapt_all, worst_shift_adapt_all = [], []
    auroc_all, auroc_shift_all, auroc_shift_adapt_all = [], [], []
    for seed in tqdm(range(num_experiments)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        test_shift_lbls_lp = np.load(
            os.path.join(args.out_dir, f'{args.dataset_shift}-test__preds_lp__{args.backbone_name}_seed{seed}.npy'))
        # note: these are logits before softmax applied
        test_shift_lbls_lp = F.softmax(torch.tensor(test_shift_lbls_lp), dim=1).numpy()

        train_loader = get_loader(train_embs, train_lbls, args.batch_size, shuffle=True)
        test_loader = get_loader(test_embs, test_lbls, args.batch_size, shuffle=False)
        test_shift_loader = get_loader(test_shift_embs, test_shift_lbls, args.batch_size, shuffle=False)

        cbm, cbm_path = prepare_cbm(args, seed, idx_to_class, num_classes, dim_emb)
        if os.path.exists(cbm_path):
            print(f"Model loaded from : {cbm_path}")
            cbm = torch.load(cbm_path)
        else:
            optimizer = torch.optim.Adam(cbm.trainable_params(), lr=1e-3)
            cbm.classifier = cbm.classifier.float()
            train(args, train_loader, cbm, optimizer)
            torch.save(cbm, cbm_path)
            print(f"Model saved to : {cbm_path}")

        cbm.eval()
        _, _, preds, projs, _ = run_inference(args, test_loader, cbm)
        _, _, preds_shift, projs_shift, _ = run_inference(args, test_shift_loader, cbm)
        np.savez(os.path.join(args.out_dir, f'{args.dataset_shift}-test__projs__{args.backbone_name}__{args.concept_method}__seed{seed}.npz'),
                 proj=projs, proj_shift=projs_shift,
                 weight=cbm.classifier.weight.detach().cpu().numpy(),
                 concepts=cbm.names,
                 idx_to_class=cbm.idx_to_class)

        if args.dataset == "isic":
            auroc = roc_auc_score(test_lbls, preds.max(1))
            auroc_shift = roc_auc_score(test_shift_lbls, preds_shift.max(1))
            auroc_all.append(auroc)
            auroc_shift_all.append(auroc_shift)
        else:
            avg, worst = get_group_acc(test_lbls, preds)
            avg_shift, worst_shift = get_group_acc(test_shift_lbls, preds_shift)
            avg_all.append(avg)
            worst_all.append(worst)
            avg_shift_all.append(avg_shift)
            worst_shift_all.append(worst_shift)


        ## compute statistics for mahalanobis-based loss in adaptation
        _, _, _, projs_train, _ = run_inference(args, train_loader, cbm)
        cbm.source_stats = sample_estimator(projs_train, train_lbls, num_classes)
        cbm.source_stats_embs = sample_estimator(train_embs, train_lbls, num_classes)


        ####################### ADAPTATION
        # continue

        cbm_adapt = prepare_adapt(args, cbm, jointly=False)
        # target finetuning
        # _, avg_shift_adapt, worst_shift_adapt, projs_shift_adapt = eval(args, test_shift_loader, cbm_adapt,
        #                                                              adaptation=True)
        # target ZS pseudo label
        test_shift_lbls_adapt = pseudo_labeling((test_shift_lbls, test_shift_lbls_zs, test_shift_lbls_lp, preds_shift), args.label_ensemble)
        test_shift_ps_loader = get_loader(test_shift_embs, test_shift_lbls_adapt, args.batch_size, shuffle=True)

        (lbls_shift_adapt_shuffled, pseudo_lbls_shift_adapt_shuffled, preds_shift_adapt,
         projs_shift_adapt, projs_res_shift_adapt) = run_inference(args, test_shift_ps_loader, cbm_adapt,
                                                                   adapt_cavs=args.adapt_cavs, adapt_classifier=args.adapt_classifier, adapt_rcb=args.rcbm,
                                                                   adaptation=True)

        #     lbls_shift_adapt_shuffled, _, preds_shift_adapt, projs_res_shift_adapt = run_inference(args, test_shift_ps_loader, cbm_adapt_rcb,
        #                                                                                             adaptation=True)


        if args.dataset == "isic":
            auroc_shift_adapt = roc_auc_score(lbls_shift_adapt_shuffled, preds_shift_adapt.max(1))
            auroc_shift_adapt_all.append(auroc_shift_adapt)

        else:
            # avg_shift_pseudo, worst_shift_pseudo = get_group_acc(lbls_shift_adapt_shuffled, pseudo_lbls_shift_adapt_shuffled)
            # print("performance with pseudo-labels... (AVG, WG):", avg_shift_pseudo, worst_shift_pseudo)

            avg_shift_adapt, worst_shift_adapt = get_group_acc(lbls_shift_adapt_shuffled, preds_shift_adapt)
            avg_shift_adapt_all.append(avg_shift_adapt)
            worst_shift_adapt_all.append(worst_shift_adapt)

        # projs_savename = f"{args.dataset_shift}-test__projs-adapted-{args.adapt_method}__{args.backbone_name}__{args.concept_method}__seed{seed}.npy"
        adapt_type = "adapted"
        if args.adapt_cavs:
            adapt_type += '-CSA'
        if args.adapt_classifier:
            adapt_type += '-LPA'
        if not args.rcbm:
            projs_savename = f"{args.dataset_shift}-test__projs-{adapt_type}__{args.backbone_name}__{args.concept_method}__seed{seed}.npz"
            print("outputs save to....", projs_savename)
            np.savez(os.path.join(args.out_dir, projs_savename), proj=projs_shift_adapt,
                     weight=cbm_adapt.model.classifier.parameters)
        else:
            adapt_type += '-RCB'
            # adapt_type += '-RCB-jointly'
            projs_savename = f"{args.dataset_shift}-test__projs-{adapt_type}__{args.backbone_name}__{args.concept_method}__seed{seed}.npz"
            print("outputs save to....", projs_savename)
            np.savez(os.path.join(args.out_dir, projs_savename),
                     proj=projs_shift_adapt, proj_residual=projs_res_shift_adapt,
                     weight=cbm_adapt.model.classifier.weight.detach().cpu().numpy(), weight_residual=cbm_adapt.residual_classifier.weight.detach().cpu().numpy())

        del cbm, cbm_adapt
        torch.cuda.empty_cache()


    print(f"[CBM] Over {num_experiments} trials...")
    print("[Source test] Average Accuracy: {:.5f} +/- {:.5f}".format(np.mean(avg_all), np.std(avg_all, ddof=1)/np.sqrt(len(avg_all))))
    print("[Source test] Worst Accuracy: {:.5f} +/- {:.5f}".format(np.mean(worst_all), np.std(worst_all, ddof=1)/np.sqrt(len(worst_all))))
    print("[Target test] Average Accuracy: {:.5f} +/- {:.5f}".format(np.mean(avg_shift_all), np.std(avg_shift_all, ddof=1)/np.sqrt(len(avg_shift_all))))
    print("[Target test] Worst Accuracy: {:.5f} +/- {:.5f}".format(np.mean(worst_shift_all), np.std(worst_shift_all, ddof=1)/np.sqrt(len(worst_shift_all))))
    print("[Target test (adapted)] Average Accuracy: {:.5f} +/- {:.5f}".format(np.mean(avg_shift_adapt_all), np.std(avg_shift_adapt_all, ddof=1)/np.sqrt(len(avg_shift_adapt_all))))
    print("[Target test (adapted)] Worst Accuracy: {:.5f} +/- {:.5f}".format(np.mean(worst_shift_adapt_all), np.std(worst_shift_adapt_all, ddof=1)/np.sqrt(len(worst_shift_adapt_all))))

    return avg_all, worst_all, avg_shift_all, worst_shift_all, avg_shift_adapt_all, worst_shift_adapt_all


if __name__ == "__main__":
    args = config()

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()

    main(args, backbone, preprocess)
