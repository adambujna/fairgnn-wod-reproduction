#!/usr/bin/env/python

import argparse
import torch

from src.dataset.dataset import GraphDataset
from src.training.train_gcn import train_gcn
from src.training.train_stage1 import train_vgae
from src.training.train_stage2 import train_stage2
from src.training.train_fairkd import train_fairkd


def main():
    parser = argparse.ArgumentParser(description="fairGNN-WOD Orchestrator")
    parser.add_argument('--model', type=str, required=True,
                        choices=['gcn', 'vgae', 'fairgnnwod', 'fairkd'])
    parser.add_argument('--task', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('--dataset', type=str, default='credit',
                        choices=['dblp', 'pokec_z', 'pokec_n', 'credit'])
    parser.add_argument('--dgl_data', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Shared Hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=50)

    # GCN Specifics
    parser.add_argument('--dropout', type=float, default=0.0)

    # FairKD Specific
    parser.add_argument('--kd_temp', type=float, default=4)
    parser.add_argument('--kd_lambda', type=float, default=0.5)

    # VGAE Specific
    parser.add_argument('--hidden_vgae', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--hgr_steps_per_epoch', type=int, default=10)
    parser.add_argument('--lambda_hgr', type=float, default=0.5)
    parser.add_argument('--hidden_hgr', type=int, default=512)

    # FairGNN-WOD Specific
    parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--lambda_dd', type=float, default=0.1)   # Lambda for the discriminator
    parser.add_argument('--alpha', type=float, default=1.0)  # Weight for LI
    parser.add_argument('--beta', type=float, default=1.0)  # Weight for LD
    parser.add_argument('--gamma', type=float, default=5.0)  # Weight for LF
    parser.add_argument('--adv_steps_per_epoch', type=int, default=2)
    parser.add_argument('--mask_warmup', type=float, default=0.2)

    args = parser.parse_args()
    print(vars(args))
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed(args.seed)

    print(f"--- Starting {args.task} {args.model.upper()} on {args.dataset} ---")
    data = GraphDataset(args.dataset, dgl_data=args.dgl_data)

    if args.task == 'train':
        if args.model == 'gcn':
            train_gcn(data, args)
        elif args.model == 'vgae':
            train_vgae(data, args)
        elif args.model == 'fairgnnwod':
            train_stage2(data, args)
        elif args.model == 'fairkd':
            train_fairkd(data, args)
    elif args.task == 'predict':
        raise NotImplementedError("Inference logic not implemented yet.")


if __name__ == "__main__":
    main()
