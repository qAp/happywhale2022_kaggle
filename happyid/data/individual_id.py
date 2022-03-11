import os
import joblib
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as albu
import pytorch_lightning as pl

from happyid.data.config import *
from happyid.data.base_data_module import BaseDataModule
from happyid.data.transforms import base_tfms


ID_ENCODER = joblib.load('/kaggle/input/happyid-label-encoding/label_encoder')

class IndividualIDDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pth = f'{r.dir_img}/{r.image}'

        img = cv2.imread(pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            tfmd = self.transform(image=img)
            img = tfmd['image']

        img = img.astype('float32')
        img = img / 255
        img = (img - MEAN_IMG) / STD_IMG

        label = ID_ENCODER.transform([r['individual_id']])
        return img, label


META_DATA_PATH = '/kaggle/input/happyid-train-meta'
FOLD = 0
IMAGE_SIZE = 64


class IndividualID(BaseDataModule):
    def __init__(self, args=None):
        super().__init__(args)

        self.args = vars(args) if args is not None else {}

        self.meta_data_path = self.args.get('meta_data_path', META_DATA_PATH)
        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)

    @staticmethod
    def add_argparse_args(parser):
        BaseDataModule.add_argparse_args(parser)
        add = parser.add_argument
        add('--meta_data_path', type=str, default=META_DATA_PATH)
        add('--fold', type=int, default=FOLD)
        add('--image_size', type=int, default=IMAGE_SIZE)

    def config(self):
        return dict()

    def setup(self):
        train_df = pd.read_csv(
            f'{self.meta_data_path}/train_fold{self.fold}.csv')

        self.train_ds = IndividualIDDataset(
            train_df,
            transform=albu.Compose(base_tfms(self.image_size))
        )

        id2weight = (1 / train_df['individual_id'].value_counts()).to_dict()
        self.train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_df['individual_id'].map(
                lambda x: id2weight[x]).values,
            num_samples=self.batch_size, replacement=True)

        valid_df = pd.read_csv(
            f'{self.meta_data_path}/valid_fold{self.fold}.csv')

        self.valid_ds = IndividualIDDataset(
            valid_df,
            transform=albu.Compose(base_tfms(self.image_size))
        )

    def prepare_data(self):
        assert os.path.exists(self.meta_data_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu)
