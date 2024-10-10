import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1]
    elif len(batch) == 2:
        return batch
    else:
        raise ValueError()


@torch.no_grad()
def get_projections(args, backbone, posthoc_layer, loader):
    all_projs, all_embs, all_lbls, all_preds = None, None, None, None
    for batch in tqdm(loader):
        batch_X, batch_Y = unpack_batch(batch)
        batch_X = batch_X.to(args.device)
        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(batch_X).detach().float()
        else:
            embeddings = backbone(batch_X).detach()
        # projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        # preds = (posthoc_layer(embeddings).detach().cpu().numpy())
        preds, projs = posthoc_layer(embeddings, return_concept=True)
        preds, projs = preds.detach().cpu().numpy(), projs.detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y.numpy()
            all_preds = preds
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
            all_preds = np.concatenate([all_preds, preds], axis=0)
    return all_embs, all_projs, all_lbls, all_preds


class EmbDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    def __len__(self):
        return len(self.data)


def get_file_path(args, dataset, file_type, extension='npy'):
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0]
    # filename = f"{dataset}-{file_type}__{args.backbone_name}__{conceptbank_source}"
    filename = f"{dataset}-{file_type}__{args.backbone_name}"
    if 'projs' in file_type or 'preds' in file_type:
        filename += f"__{args.cbm_type}"
    return os.path.join(args.out_dir, f"{filename}.{extension}")

def load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader,
                                dataset_shift=False, skip_train=False):
    """
    dataset_shift: if True, we'll use shifted dataset to compute the projections.
    skip_train: if True, we'll skip computing the projections for the train set.
    """



    dataset = args.dataset if not dataset_shift else args.dataset_shift
    # To make it easier to analyize results/rerun with different params, we'll extract the embeddings and save them

    """
    train_file = f"{dataset}-train-embs__{args.backbone_name}__{conceptbank_source}.npy"
    test_file = f"{dataset}-test-embs__{args.backbone_name}__{conceptbank_source}.npy"
    # train_proj_file = f"{dataset}-train-proj__dim{args.num_concepts}__{args.backbone_name}__{conceptbank_source}.npy"
    test_proj_file = f"{dataset}-test-proj__dim{args.num_concepts}__{args.backbone_name}__{conceptbank_source}.npy"
    train_lbls_file = f"{dataset}-train-lbls__{args.backbone_name}__{conceptbank_source}.npy"
    test_lbls_file = f"{dataset}-test-lbls__{args.backbone_name}__{conceptbank_source}.npy"
    test_preds_file = f"{dataset}-test-preds__{args.backbone_name}__{conceptbank_source}.npy"

    train_file = os.path.join(args.out_dir, train_file)
    test_file = os.path.join(args.out_dir, test_file)
    # train_proj_file = os.path.join(args.out_dir, train_proj_file)
    test_proj_file = os.path.join(args.out_dir, test_proj_file)
    train_lbls_file = os.path.join(args.out_dir, train_lbls_file)
    test_lbls_file = os.path.join(args.out_dir, test_lbls_file)
    test_preds_file = os.path.join(args.out_dir, test_preds_file)
    """
    # Create file paths
    file_types = ['train-embs', 'test-embs', 'train-projs', 'test-projs', 'train-lbls', 'test-lbls', 'test-preds']
    if skip_train:
        file_types.remove('train-embs')
        file_types.remove('train-lbls')
        file_types.remove('train-projs')


    file_paths = {file_type: get_file_path(args, dataset, file_type) for file_type in file_types}

    train_embs, train_lbls = None, None
    if all(os.path.exists(file_path) for file_path in file_paths.values()):
        print(f"Loading embeddings from {file_paths['test-embs']}...")
        test_embs = np.load(file_paths['test-embs'])
        # test_projs = np.load(test_proj_file)
        test_lbls = np.load(file_paths['test-lbls'])

        if not skip_train:
            train_embs = np.load(file_paths['train-embs'])
            # train_projs = np.load(train_proj_file)
            train_lbls = np.load(file_paths['train-lbls'])


        # assert test_embs.shape[-1] == args.dim_emb
        # assert test_projs.shape[-1] == args.num_concepts

    else:
        print("Computing embeddings...")
        # test_embs, test_lbls = get_projections(args, backbone, posthoc_layer, test_loader)
        test_embs, test_projs, test_lbls, test_preds = get_projections(args, backbone, posthoc_layer, test_loader)

        np.save(file_paths['test-embs'], test_embs)
        np.save(file_paths['test-lbls'], test_lbls)
        np.save(file_paths['test-projs'], test_projs)
        np.save(file_paths['test-preds'], test_preds)

        if not skip_train:
            # train_embs, train_projs, train_lbls = get_projections(args, backbone, posthoc_layer, train_loader, compute_proj)
            # train_embs, train_lbls = get_projections(args, backbone, posthoc_layer, train_loader)
            train_embs, train_projs, train_lbls, _ = get_projections(args, backbone, posthoc_layer, train_loader)

            np.save(file_paths['train-embs'], train_embs)
            np.save(file_paths['train-lbls'], train_lbls)
            np.save(file_paths['train-projs'], train_projs)

    return (train_embs, train_lbls), (test_embs, test_lbls)

