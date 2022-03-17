import os
import torch
import albumentations as albu
import pytorch_lightning as pl


BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count()


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpus', None), (int, str))

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--batch_size', type=int, default=BATCH_SIZE)
        add('--num_workers', type=int, default=NUM_WORKERS)

    def config(self):
        epochs = self.args.get('max_epochs')
        len_dl = len(self.train_dataloader())

        overfit_batches = self.args.get('overfit_batches')
        if overfit_batches != 0:
            if overfit_batches < 1:
                len_dl = int(overfit_batches * len_dl)
            else:
                len_dl = int(overfit_batches)

        return dict(total_steps=epochs * len_dl)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu)
