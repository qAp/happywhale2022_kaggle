

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
    _add('--newid_inclusion_method', type=str, default='all')
    _add('--newid_close_thres', type=float, default=.8,
         help='new_individual distance threshold.')
    _add('--auto_newid_close_thres', type=ast.literal_eval, default='False')
    _add('--newid_weight', type=float, default=0.1)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    ifold = 0

    emb_df, emb = load_embedding(args.emb_dir, ifold)

    ref_df = load_ref_test_dfs(
        meta_data_path=args.meta_data_path, ifold=ifold,
        ref_splits=['train', 'valid', 'extra', 'test'],
        test_splits=None)
    test_df = load_ss()
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

    test_df['predictions'] = test_df.image.apply(lambda x: ' '.join(preds[x]))
    test_df.to_csv('/kaggle/working/submission.csv', index=False)


if __name__ == '__main__':
    main()
