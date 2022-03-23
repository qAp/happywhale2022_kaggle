
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
from happyid.retrieval import retrieval_predict




def _setup_parser():
    parser = setup_parser()
    _add = parser.add_argument
    _add('--ref_emb_path', type=str, default='emb.npz',
        help='Embeddings of existing images')
    _add('--ref_emb_meta_path', type=str, default='train.csv',
        help='Embedding metat data.')
    _add('--folds_ref_emb_meta_path', nargs='+', type=str,
        default=5*['train.csv'], help='Embedding metat data for the 5 folds.')
    _add('--folds_model_class', nargs='+', type=str, default=5*['DOLG'])
    _add('--folds_backbone_name', nargs='+', type=str, default=5*['resnet18'])
    _add('--folds_checkpoint_path', nargs='+', type=str,
        default=5*['best.pth'], help='Model checkpoint paths for the 5 folds.')

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    args.gem_p_trainable = True
    args.return_emb = True

    assert (NUM_FOLD
            == len(args.folds_model_class)
            == len(args.folds_backbone_name)
            == len(args.folds_checkpoint_path)
            == len(args.folds_ref_emb_meta_path)
            )

    loaded = np.load(args.ref_emb_path)

    folds_score = []

    for ifold in range(NUM_FOLD):
        print(f'Validating fold {ifold + 1}/{NUM_FOLD}')

        args.ref_emb_meta_path = args.folds_ref_emb_meta_path[ifold]
        args.model_class = args.folds_model_class[ifold]
        args.backbone_name = args.folds_backbone_name[ifold]
        args.load_from_checkpoint = args.folds_checkpoint_path[ifold]

        emb_df = pd.read_csv(args.ref_emb_meta_path)
        ref_emb = torch.from_numpy(loaded['embed'][emb_df['idx_embed']])

        data_class = import_class(f'happyid.data.{args.data_class}')
        model_class = import_class(f'happyid.models.{args.model_class}')
        lit_model_class = import_class(
            f'happyid.lit_models.{args.lit_model_class}')

        data = data_class(args)
        data.prepare_data()
        data.setup()

        model = model_class(args=args)
        if args.load_from_checkpoint:
            lit_model = lit_model_class.load_from_checkpoint(
                checkpoint_path=args.load_from_checkpoint,
                model=model, args=args)
        else:
            lit_model = lit_model_class(model=model, args=args)

        trainer = pl.Trainer.from_argparse_args(args)
        preds = trainer.predict(model=lit_model, dataloaders=data.val_dataloader())
        preds = torch.cat(preds, dim=0)
    #     # For debugging, generate random embedding to save time
    #     preds = torch.randn(len(data.valid_ds), args.embedding_size)

        predictions = retrieval_predict(pred_emb=preds,
                                        ref_emb=ref_emb, ref_emb_df=emb_df,
                                        batch_size=args.batch_size)

        map5 = map_per_set(
            labels=data.valid_ds.df['individual_id'].to_list(),
            predictions=predictions)

        folds_score.append(map5)


    print('MAP@5 of each fold:', *[f'{score:.3f}' for score in folds_score])
    print('Mean MAP@5 over all folds:', f'{np.array(folds_score).mean():.3f}')


if __name__ == '__main__':
    main()