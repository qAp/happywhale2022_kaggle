
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

    folds_score = []

    for ifold in range(NUM_FOLD):
        print(f'Validating fold {ifold + 1}/{NUM_FOLD}')

        emb_df = pd.read_csv(f'{args.emb_dir}/fold{ifold}_emb.csv')

        ref_df_list = [
            pd.read_csv(f'{args.meta_data_path}/{split}_fold{ifold}.csv') 
            for split in ('train', 'valid', 'extra')]
        ref_df = pd.concat(ref_df_list, axis=0, ignore_index=True)

        test_df = pd.read_csv(f'{args.meta_data_path}/test_fold{ifold}.csv')
        is_oldid = test_df.individual_id.isin(ref_df.individual_id.unique())
        test_df.loc[~is_oldid, 'individual_id'] = 'new_individual'

        emb = np.load(f'{args.emb_dir}/fold{ifold}_emb.npz')['embed']
        emb = torch.from_numpy(emb)

        ref_idx = (
            ref_df
            .merge(emb_df.reset_index(), on='image', how='inner')['index']
            .to_list()
        )
        ref_emb = emb[ref_idx]

        test_idx = (
            test_df
            .merge(emb_df.reset_index(), on='image', how='inner')['index']
            .to_list()
        )
        test_emb = emb[test_idx]

        ref_emb = ref_emb / ref_emb.norm(p='fro', dim=1, keepdim=True)
        test_emb = test_emb / test_emb.norm(p='fro', dim=1, keepdim=True)
        dist_matrix = euclidean_dist(test_emb, ref_emb)

        shortest_dist, ref_idx = dist_matrix.topk(k=50, largest=False, dim=1)

        dist_df = get_closest_ids_df(test_df, ref_df, 
                                     shortest_dist, ref_idx)
        
        print('Distance df')                                        
        display(dist_df.describe())

        if args.auto_newid_dist_thres:
            print('Searching for best newid_dist_thres...', end='')
            thres_step = 0.1
            thres_values = np.arange(2, 0 - thres_step, - thres_step)

            best_score, best_thres = 0., None
            for thres in thres_values:
                preds = predict_top5(dist_df, newid_dist_thres=thres)
                score = get_map5_score(test_df, preds, 
                                       newid_weight=args.newid_weight)
                print(f'thres = {thres:.1f}. score = {score:.3f}')
                if score >= best_score:
                    best_score = score
                    best_thres = thres

            print(f'Best newid_dist_thres = {best_thres:.1f}. Score = {best_score:.3f}.')
            folds_score.append(best_score)
        else:
            preds = predict_top5(dist_df, newid_dist_thres=args.newid_dist_thres)
            score = get_map5_score(test_df, preds, 
                                   newid_weight=args.newid_weight)
            folds_score.append(score)


    print('MAP@5 of each fold:', *[f'{score:.3f}' for score in folds_score])
    print('Mean MAP@5 over all folds:', f'{np.array(folds_score).mean():.3f}')


if __name__ == '__main__':
    main()
