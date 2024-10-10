import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", type=str, default="datasets/concepts/cub_resnet18_cub_0.1_100.pkl",
                        help="Path to the concept bank")
    parser.add_argument("--out-dir", type=str, default="results/cub/", help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--dataset-shift", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-epochs", default=20, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=2e-4, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--l2-penalty", default=0.001, type=float)

    # parameters for concept learning
    parser.add_argument("--unsupervised", action="store_true",
                        help="Whether to learn concepts or to use a static concept bank.")
    parser.add_argument("--dim-emb", default=512, type=int, help="Dimensionality of embedding vector.")
    parser.add_argument("--num-concepts", default=150, type=int, help="Number of concepts to learn.")
    parser.add_argument("--cbm-type", default="bank-85concepts", type=str)

    # parameters for OOD detection
    parser.add_argument("--ood-method", default="msp", type=str, help="OOD detection method.")
    return parser.parse_args()
