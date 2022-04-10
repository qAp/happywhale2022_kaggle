

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
from happyid.retrieval import (load_embedding, load_ref_test_dfs, 
                               get_emb_subset, retrieve_topk, 
                               get_closest_ids_df, predict_top5)


def _setup_parser():
    parser = setup_parser()
    _add = parser.add_argument
    _add('--emb_dir', type=str, default='/kaggle/input/happyid-tvet-data')
    _add('--retrieval_crit', type=str, default='cossim')
    _add('--newid_close_thres', type=float, default=.8,
         help='new_individual distance threshold.')
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    ifold = 0

    emb_df, emb = load_embedding(args.emb_dir, ifold)

    ref_df = load_ref_test_dfs(
        meta_data_path=args.meta_data_dir, ifold=0,
        ref_splits=['train', 'valid', 'extra', 'test'],
        test_splits=None)
    test_df = pd.read_csv(
        '/kaggle/input/happy-whale-and-dolphin/sample_submission.csv')

    ref_df, ref_emb = get_emb_subset(emb_df, emb, ref_df)
    test_df, test_emb = get_emb_subset(emb_df, emb, test_df)

    topked = retrieve_topk(test_emb, ref_emb, k=50, 
                           batch_size=args.batch_size,
                           retrieval_crit=args.retrieval_crit)

    close_df = get_closest_ids_df(test_df, ref_df, topked, 
                                  retrieval_crit=args.retrieval_crit)

    preds = predict_top5(close_df, 
                         newid_close_thres=args.newid_close_thres, 
                         retrieval_crit=args.retrieval_crit)

    test_df['predictions'] = test_df.image.apply(lambda x: ' '.join(preds[x]))
    test_df.to_csv('/kaggle/working/submission.csv', index=False)


if __name__ == '__main__':
    main()
