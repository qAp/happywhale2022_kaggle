
import os, sys
import argparse
from tqdm.auto import tqdm
import importlib
import numpy as np, pandas as pd
import torch, torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from happyid.data.config import *
from happyid.utils import import_class, setup_parser
from happyid.lit_models.losses import euclidean_dist
from happyid.lit_models.metrics import map_per_set
from happyid.retrieval import get_closest_ids_df, predict_top5




def _setup_parser():
    parser = setup_parser()
    _add = parser.add_argument

    _add('--folds_id_encoder_path', nargs='+', type=str, 
         default=[f'label_encoder_fold{i}' for i in range(NUM_FOLD)])
    _add('--folds_model_class', nargs='+', type=str, 
         default=NUM_FOLD * ['DOLG'])
    _add('--folds_backbone_name', nargs='+', type=str, 
         default=NUM_FOLD * ['resnet18'])
    _add('--folds_checkpoint_path', nargs='+', type=str,
        default=NUM_FOLD * ['best.pth'], 
        help='Model checkpoint paths for the CV folds.')
    _add('--folds_ref_emb_dir', nargs='+', type=str,
         default=NUM_FOLD * ['emb'])

    return parser


def _get_ref_emb(args):
    ifold = args.fold
    knownid_emb_path = args.folds_knownid_emb_path[ifold]
    knownid_emb_meta_path = args.folds_knownid_emb_meta_path[ifold]
    newid_emb_path = args.folds_newid_emb_path[ifold]
    newid_emb_meta_path = args.folds_newid_emb_meta_path[ifold]

    knownid_emb = np.load(knownid_emb_path)['embed']
    knownid_emb_meta = pd.read_csv(knownid_emb_meta_path)

    if newid_emb_path is not None:

        newid_emb = np.load(newid_emb_path)['embed']
        ref_emb = np.concatenate(
            [knownid_emb, newid_emb.mean(axis=0, keepdims=True)],
            axis=0)

        newid_df = pd.DataFrame(columns=knownid_emb_meta.columns)
        newid_df['individual_id'] = ['new_individual']
        ref_emb_meta = knownid_emb_meta.append(newid_df, ignore_index=True)
    else:
        ref_emb = knownid_emb
        ref_emb_meta = knownid_emb_meta

    return ref_emb, ref_emb_meta


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    assert (NUM_FOLD
            == len(args.folds_id_encoder_path)
            == len(args.folds_model_class)
            == len(args.folds_backbone_name)
            == len(args.folds_checkpoint_path)
            == len(args.folds_ref_emb_dir)
            )

    folds_score = []

    for ifold in range(NUM_FOLD):
        print(f'Validating fold {ifold + 1}/{NUM_FOLD}')

        args.fold = ifold
        args.id_encoder_path = args.folds_id_encoder_path[ifold]
        args.model_class = args.folds_model_class[ifold]
        args.backbone_name = args.folds_backbone_name[ifold]
        args.load_from_checkpoint = args.folds_checkpoint_path[ifold]
        ref_emb_dir = args.folds_ref_emb_dir[ifold]

        data_class = import_class(f'happyid.data.{args.data_class}')
        model_class = import_class(f'happyid.models.{args.model_class}')
        lit_model_class = import_class(
            f'happyid.lit_models.{args.lit_model_class}')

        data = data_class(args)
        data.prepare_data()
        data.setup()

        model = model_class(data_config=data.config(), args=args)
        if args.load_from_checkpoint:
            lit_model = lit_model_class.load_from_checkpoint(
                checkpoint_path=args.load_from_checkpoint,
                model=model, args=args)
        else:
            lit_model = lit_model_class(model=model, args=args)

        # trainer = pl.Trainer.from_argparse_args(args)
        # emb = trainer.predict(model=lit_model, dataloaders=data.val_dataloader())
        # emb = torch.cat(emb, dim=0)
        # For debugging, generate random embedding to save time
        emb = torch.randn(len(data.valid_ds), args.embedding_size)

        print('Loading embedding database for reference...', end='')
        ref_emb = np.load(f'{ref_emb_dir}/emb.npz')['embed']
        ref_emb_df = pd.read_csv(f'{ref_emb_dir}/emb.csv')
        ref_emb = torch.from_numpy(ref_emb)
        print('done')

        print(emb.shape, ref_emb.shape)
        emb = emb / emb.norm(p='fro', dim=1, keepdim=True)
        ref_emb = ref_emb / ref_emb.norm(p='fro', dim=1, keepdim=True)

        print('Computing distance matrix...')
        dist_matrix = euclidean_dist(emb, ref_emb)
        print('done')

        print('Get 50 closest database images...', end='')
        shortest_dist, ref_idx = dist_matrix.topk(k=50, largest=False, dim=1)
        print('done')

        dist_df = get_closest_ids_df(data.valid_ds.df, ref_emb_df, 
                                     shortest_dist, ref_idx)

        print('Finalising top 5 predictions...', end='')
        preds = predict_top5(dist_df, newid_dist_thres=args.newid_dist_thres)
        print('done')

        predictions = (data.valid_ds.df.image
                       .apply(lambda x: ' '.join(preds[x])).to_list()
                       )
        labels = data.valid_ds.df['individual_id'].to_list()

        print('Calculating MAP@5...', end='')
        map5 = map_per_set(labels=labels, predictions=predictions)
        print('done\n')

        folds_score.append(map5)


    print('MAP@5 of each fold:', *[f'{score:.3f}' for score in folds_score])
    print('Mean MAP@5 over all folds:', f'{np.array(folds_score).mean():.3f}')


if __name__ == '__main__':
    main()
