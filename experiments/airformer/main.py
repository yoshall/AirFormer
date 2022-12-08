import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch.nn as nn
import torch

from src.utils.helper import get_dataloader, check_device, get_num_nodes
from src.utils.metrics import masked_mae
from src.models.airformer import AirFormer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config, str_to_bool


def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='airformer',
                        help='which model to train')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dartboard', type=int, default=0,
                        help='0: 50-200, 1: 50-200-500, 2: 50, 3: 25-100-250')
    parser.add_argument('--stochastic_flag', type=str_to_bool,
                        default=True, help='whether to use stochastic temporal transformer')
    parser.add_argument('--spatial_flag', type=str_to_bool,
                        default=True, help='whether to use spatial transformer')

    parser.add_argument('--base_lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    args = parser.parse_args()
    args.steps = [12000]
    print(args)

    folder_name = '{}-{}-{}-{}-{}-{}-{}'.format(args.n_hidden,
                                                args.num_heads,
                                                args.dropout,
                                                args.stochastic_flag,
                                                args.spatial_flag,
                                                args.dartboard,
                                                args.aug)
    args.log_dir = './logs/{}/{}/{}/'.format(args.dataset,
                                             args.model_name,
                                             folder_name)
    args.num_nodes = get_num_nodes(args.dataset)

    # you can ignore the following code (they are for GNN baselines)
    if args.filter_type == 'scalap':
        args.support_len = 1
    else:
        args.support_len = 2

    args.datapath = os.path.join('./data', args.dataset)
    args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(
        args.dataset.lower())

    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return args


def main():
    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)

    model = AirFormer(dropout=args.dropout,
                      spatial_flag=args.spatial_flag,
                      stochastic_flag=args.stochastic_flag,
                      hidden_channels=args.n_hidden,
                      dartboard=args.dartboard,
                      end_channels=args.n_hidden * 8,
                      num_heads=args.num_heads,
                      name=args.model_name,
                      dataset=args.dataset,
                      device=device,
                      num_nodes=args.num_nodes,
                      seq_len=args.seq_len,
                      horizon=args.horizon,
                      input_dim=args.input_dim,
                      output_dim=args.output_dim)

    data = get_dataloader(args.datapath,
                          args.batch_size,
                          args.output_dim)

    if args.stochastic_flag:
        from src.trainers.airformer_stochastic_trainer import Trainer
    else:
        from src.trainers.airformer_trainer import Trainer

    trainer = Trainer(model=model,
                      adj_mat=adj_mat,
                      filter_type=args.filter_type,
                      data=data,
                      aug=args.aug,
                      base_lr=args.base_lr,
                      steps=[3, 6, 9],
                      lr_decay_ratio=args.lr_decay_ratio,
                      log_dir=args.log_dir,
                      n_exp=args.n_exp,
                      save_iter=args.save_iter,
                      clip_grad_value=args.max_grad_norm,
                      max_epochs=args.max_epochs,
                      patience=args.patience,
                      device=device)

    if args.mode == 'train':
        trainer.train()
        trainer.test(-1, 'test')
    else:
        trainer.test(-1, args.mode)


if __name__ == "__main__":
    main()
