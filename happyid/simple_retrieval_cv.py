
import os, sys, ast
import argparse
from tqdm.auto import tqdm
from IPython.display import display
import importlib
import numpy as np, pandas as pd
import torch, torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from happyid.data.config import *
from happyid.utils import import_class, setup_parser
from happyid.lit_models.losses import euclidean_dist
from happyid.retrieval import (
    load_embedding, load_ref_test_dfs, load_fliplr_df, load_ss,
    get_emb_subset, include_new_individual,
    retrieve_topk, get_closest_ids_df, simple_predict_top5,
    get_map5_score)



def _setup_parser():
    parser = setup_parser()
    _add = parser.add_argument
    _add('--emb_dir', type=str, default='/kaggle/input/happyid-tvet-data')
    _add('--retrieval_crit', type=str, default='cossim')
    _add('--newid_inclusion_method', type=str, default=None)
    _add('--newid_close_thres', type=float, default=.8,
         help='new_individual distance threshold.')
    _add('--auto_newid_close_thres', type=ast.literal_eval, default='False')
    _add('--newid_weight', type=float, default=0.1)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    folds_score = []

    for ifold in range(NUM_FOLD):
        print(f'Validating fold {ifold + 1}/{NUM_FOLD}')

        emb_df, emb = load_embedding(args.emb_dir, ifold)

        ref_df, test_df = load_ref_test_dfs(
            meta_data_path=args.meta_data_path, ifold=ifold,
            ref_splits=['train', 'valid', 'extra'],
            test_splits=['test'],
            new_individual=True)

        new_df = load_fliplr_df()

        ref_df, ref_emb = get_emb_subset(emb_df, emb, ref_df)
        test_df, test_emb = get_emb_subset(emb_df, emb, test_df)
        new_df, new_emb = get_emb_subset(emb_df, emb, new_df)

        ref_df, ref_emb = include_new_individual(
            ref_df, ref_emb, new_df, new_emb,
            method=args.newid_inclusion_method)

        topked = retrieve_topk(test_emb, ref_emb, k=50, 
                               batch_size=len(test_emb),
                               retrieval_crit=args.retrieval_crit)

        close_df = get_closest_ids_df(test_df, ref_df, topked,
                                      retrieval_crit=args.retrieval_crit)

        preds = simple_predict_top5(close_df)

        score = get_map5_score(test_df, preds, newid_weight=args.newid_weight)
        folds_score.append(score)


    print('MAP@5 of each fold:', *[f'{score:.3f}' for score in folds_score])
    print('Mean MAP@5 over all folds:', f'{np.array(folds_score).mean():.3f}')


if __name__ == '__main__':
    main()
