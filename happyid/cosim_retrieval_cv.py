
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
    load_embedding, load_ref_test_dfs, get_emb_subset,
    retrieve_topk, get_closest_ids_df, predict_top5, 
    get_map5_score)



def _setup_parser():
    parser = setup_parser()
    _add = parser.add_argument
    _add('--emb_dir', type=str, default='/kaggle/input/happyid-tvet-data')
    _add('--retrieval_crit', type=str, default='cossim')

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
            new_individual=True
            )
        ref_df, ref_emb = get_emb_subset(emb_df, emb, ref_df)
        test_df, test_emb = get_emb_subset(emb_df, emb, test_df)

        topked = retrieve_topk(test_emb, ref_emb, k=50, 
                               batch_size=len(test_emb),
                               retrieval_crit=args.retrieval_crit)

        close_df = get_closest_ids_df(
            test_df, ref_df, topked,
            retrieval_crit=args.retrieval_crit)
        
        print('Close df')                                        
        display(close_df.describe())

        if args.auto_newid_close_thres:
            print('Searching for best newid_close_thres...', end='')
            thres_step = 0.1
            if args.retrieval_crit == 'cossim':
                thres_values = np.arange(-1, 1 + thres_step, thres_step)
            else:
                thres_values = np.arange(2, 0 - thres_step, - thres_step)

            best_score, best_thres = 0., None
            for thres in thres_values:
                preds = predict_top5(close_df, newid_close_thres=thres,
                                     retrieval_crit=args.retrieval_crit)

                score = get_map5_score(test_df, preds, 
                                       newid_weight=args.newid_weight)

                print(f'thres = {thres:.1f}. score = {score:.3f}')
                if score >= best_score:
                    best_score = score
                    best_thres = thres

            print(f'Best newid_dist_thres = {best_thres:.1f}. Score = {best_score:.3f}.')
            folds_score.append(best_score)
        else:
            preds = predict_top5(
                close_df, 
                newid_close_thres=args.newid_close_thres,
                retrieval_crit=args.retrieval_crit)

            score = get_map5_score(test_df, preds, 
                                   newid_weight=args.newid_weight)
            folds_score.append(score)


    print('MAP@5 of each fold:', *[f'{score:.3f}' for score in folds_score])
    print('Mean MAP@5 over all folds:', f'{np.array(folds_score).mean():.3f}')


if __name__ == '__main__':
    main()
