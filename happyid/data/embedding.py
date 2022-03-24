

import numpy as np, pandas as pd
import torch
import albumentations as albu
from happyid.data.individual_id import IndividualIDDataset
from happyid.data import IndividualID
from happyid.data.transforms import base_tfms



class EmbeddedIndividual(IndividualID):
    def __init__(self, args=None):
        super().__init__(args)
        self.args = vars(args) if args is not None else {}

    def setup(self):
        df = pd.read_csv(self.meta_data_path).iloc[:100]

        if self.image_dir is not None:
            df['dir_img'] = self.image_dir
        else:
            assert 'dir_img' in df

        self.train_ds = None
        self.valid_ds = None

        self.test_ds = IndividualIDDataset(
            df, transform=albu.Compose(self.test_tfms))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu)

        
class EmbIndividualID(IndividualID):
    def __init__(self, args=None):
        super().__init__(args)
        self.args = vars(args) if args is not None else {}

        self.emb_meta_path = self.args.get('emb_meta_path')
        self.train_tfms = base_tfms(self.image_size)

    @staticmethod
    def add_argparse_args(parser):
        IndividualID.add_argparse_args(parser)
        parser.add_argument('--emb_meta_path', type=str, default='train.csv')

    def setup(self):
        self.train_ds = IndividualIDDataset(
            df=pd.read_csv(self.emb_meta_path),
            transform=albu.Compose(self.train_tfms))

        self.valid_ds = None
        self.test_ds = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu)


class EmbNewIndividual(EmbIndividualID):
    def __init__(self, args=None):
        super().__init__(args)
        self.args = vars(args) if args is not None else {}

        self.image_dir = self.args.get('image_dir')

    @staticmethod
    def add_argparse_args(parser):
        EmbIndividualID.add_argparse_args(parser)
        parser.add_argument('--image_dir', type=str, default='train_images')

    def setup(self):
        df = pd.read_csv(self.emb_meta_path)
        df['dir_img'] = self.image_dir
        df.drop('individual_id', axis=1, inplace=True)

        self.train_ds = IndividualIDDataset(
            df.iloc[:100],
            transform=albu.Compose(self.train_tfms))

        self.valid_ds = None
        self.test_ds = None
