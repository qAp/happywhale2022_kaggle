import os
import argparse
import pathlib
import shutil
import albumentations as albu
import numpy as np, pandas as pd
import torch
import pytorch_lightning as pl
from happyid.data.individual_id import IndividualIDDataset
from happyid.data import IndividualID
from happyid.data.transforms import base_tfms
from happyid.utils import setup_parser, import_class




def main():
    parser = setup_parser()
    args = parser.parse_args()

    data_class = import_class(f'happyid.data.{args.data_class}')
    data = data_class(args)
    data.prepare_data()
    data.setup()
    # fig, axs = data.show_batch(split='train')

    model_class = import_class(f'happyid.models.{args.model_class}')
    lit_model_class = import_class(f'happyid.lit_models.{args.lit_model_class}')
    model = model_class(args=args)
    if args.load_from_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            checkpoint_path=args.load_from_checkpoint, 
            model=model, args=args)
    else:
        lit_model = lit_model_class(model=model, args=args)

    trainer = pl.Trainer.from_argparse_args(args)
    preds = trainer.predict(model=lit_model, 
                            dataloaders=data.train_dataloader())
    preds = torch.cat(preds, dim=0)
    preds = preds.cpu().numpy()

    pathlib.Path(args.dir_out).mkdir(exist_ok=True, parents=True)
    np.savez_compressed(f'{args.dir_out}/emb', 
                        embed=preds)
    shutil.copy(args.emb_meta_path, args.dir_out)


if __name__ == '__main__':
    main()
