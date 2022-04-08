

import os
import sys
import ast
import argparse
from tqdm.auto import tqdm
from IPython.display import display
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from happyid.data.config import *
from happyid.utils import import_class, setup_parser
from happyid.lit_models.losses import euclidean_dist
from happyid.retrieval import get_closest_ids_df, predict_top5, get_map5_score


def _setup_parser():
    parser = setup_parser()
    _add = parser.add_argument
    _add('--emb_dir', type=str, default='/kaggle/input/happyid-tvet-data')

    _add('--newid_dist_thres', type=float, default=.8,
         help='new_individual distance threshold.')
    _add('--auto_newid_dist_thres', type=ast.literal_eval, default='False')
    _add('--newid_weight', type=float, default=0.1)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    ifold = 0

    print('Loading all samples')
    emb_df = pd.read_csv(f'{args.emb_dir}/fold{ifold}_emb.csv')
    emb = np.load(f'{args.emb_dir}/fold{ifold}_emb.npz')['embed']
    emb = torch.from_numpy(emb)

    print('Loading reference samples')
    dfs_ = [
        pd.read_csv(f'{args.meta_data_path}/{s}_fold{ifold}.csv')
        for s in ('train', 'valid', 'extra', 'test')]
    ref_df = pd.concat(dfs_, axis=0, ignore_index=True)
    ref_idx = (
        ref_df
        .merge(emb_df.reset_index(), on='image', how='inner')['index']
        .to_list()
    )
    ref_emb = emb[ref_idx]  
    ref_emb = ref_emb / ref_emb.norm(p='fro', dim=1, keepdim=True)

    print('Loading test samples')
    test_df = pd.read_csv(
        '/kaggle/input/happy-whale-and-dolphin/sample_submission.csv')
    test_idx = (
        test_df
        .merge(emb_df.reset_index(), on='image', how='inner')['index']
        .to_list()
    )
    test_emb = emb[test_idx]
    test_emb = test_emb / test_emb.norm(p='fro', dim=1, keepdim=True)

    print('Retrieving...')
    print('Computing distance matrix...')
    dist_matrix = euclidean_dist(test_emb, ref_emb)
    print('Searching closest 50...')
    shortest_dist, ref_idx = dist_matrix.topk(k=50, largest=False, dim=1)
    print('Distance df')
    dist_df = get_closest_ids_df(test_df, ref_df, shortest_dist, ref_idx)
    print('Predict closest 5')
    preds = predict_top5(dist_df, newid_dist_thres=args.newid_dist_thres)

    test_df['prediction'] = test_df.image.apply(lambda x: preds[x])
    test_df.to_csv('/kaggle/working/submission.csv', index=False)


if __name__ == '__main__':
    main()
