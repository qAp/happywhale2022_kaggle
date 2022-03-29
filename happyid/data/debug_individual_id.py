import os
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import pytorch_lightning as pl

from happyid.data.config import *
from happyid.data.base_data_module import BaseDataModule
from happyid.data.transforms import base_tfms, aug_tfms


class DebugIndividualIDDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, id_encoder=None):
        super().__init__()

        self.df = df
        self.transform = transform
        self.id_encoder = id_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        pth = f'{r.dir_img}/{r.image}'
        # img = cv2.imread(pth)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(Image.open(pth).convert('RGB'))

        if self.transform:
            tfmd = self.transform(image=img)
            img = tfmd['image']


        img = img.astype('float32')
        img = img / 255
        img = (img - MEAN_IMG) / STD_IMG

        if 'individual_id' in r:
            if r['individual_id'] in self.id_encoder.classes_:
                label = self.id_encoder.transform([r['individual_id']])
            else:
                label = np.array([-1])
        else:
            label = np.array([-1])

        return img, label


META_DATA_PATH = '/kaggle/input/happyid-train-meta'
FOLD = 0
IMAGE_SIZE = 64
INFER_SUBSET = None
IMAGE_DIR = None
ID_ENCODER_PATH = '/kaggle/input/happyid-label-encoding/label_encoder'


class DebugIndividualID(BaseDataModule):
    def __init__(self, args=None):
        super().__init__(args)

        self.args = vars(args) if args is not None else {}

        self.meta_data_path = self.args.get('meta_data_path', META_DATA_PATH)
        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)
        self.aug = self.args.get('aug', False)
        self.infer_subset = self.args.get('infer_subset', INFER_SUBSET)
        self.image_dir = self.args.get('image_dir', IMAGE_DIR)
        self.id_encoder_path = self.args.get(
            'id_encoder_path', ID_ENCODER_PATH)

        if self.aug:
            self.train_tfms = aug_tfms(self.image_size)
        else:
            self.train_tfms = base_tfms(self.image_size)
        self.valid_tfms = base_tfms(self.image_size)
        self.test_tfms = base_tfms(self.image_size)

    @staticmethod
    def add_argparse_args(parser):
        BaseDataModule.add_argparse_args(parser)
        add = parser.add_argument
        add('--meta_data_path', type=str, default=META_DATA_PATH)
        add('--fold', type=int, default=FOLD)
        add('--image_size', type=int, default=IMAGE_SIZE)
        add('--aug', action='store_true', default=False)
        add('--infer_subset', type=int, default=INFER_SUBSET,
            help='Infer on subset of submission samples')
        add('--image_dir', type=str, default=IMAGE_DIR)
        add('--id_encoder_path', type=str, default=ID_ENCODER_PATH)

    def config(self):
        config = super().config()
        config.update({
            'num_class': len(self.train_ds.id_encoder.classes_)})
        return config

    def setup(self):
        id_encoder = joblib.load(self.id_encoder_path)

        train_df = pd.read_csv(
            f'{self.meta_data_path}/train_fold{self.fold}.csv')
        if self.image_dir is not None:
            train_df['dir_img'] = self.image_dir
        else:
            assert 'dir_img' in train_df
        self.train_ds = DebugIndividualIDDataset(
            train_df,
            transform=albu.Compose(self.train_tfms),
            id_encoder=id_encoder
        )
        id2weight = (1 / train_df['individual_id'].value_counts()).to_dict()
        self.train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_df['individual_id'].map(
                lambda x: id2weight[x]).values,
            num_samples=len(self.train_ds),
            replacement=True)

        valid_df = pd.read_csv(
            f'{self.meta_data_path}/valid_fold{self.fold}.csv')
        if self.image_dir is not None:
            valid_df['dir_img'] = self.image_dir
        else:
            assert 'dir_img' in valid_df
        self.valid_ds = DebugIndividualIDDataset(
            valid_df,
            transform=albu.Compose(self.valid_tfms),
            id_encoder=id_encoder
        )

        ss_df = pd.read_csv(f'{DIR_BASE}/sample_submission.csv')
        if self.infer_subset is not None:
            assert self.infer_subset <= len(ss_df)
            test_df = ss_df.sample(self.infer_subset, replace=False)
        else:
            test_df = ss_df
        self.test_ds = DebugIndividualIDDataset(
            test_df, transform=albu.Compose(self.test_tfms)
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

    def show_batch(self, split='train'):
        if split == 'valid':
            dl = iter(self.val_dataloader())
        elif split == 'test':
            dl = iter(self.test_dataloader())
        else:
            dl = iter(self.train_dataloader())

        for _ in range(np.random.randint(low=1, high=11)):
            if split in ['train', 'valid']:
                xb, yb = next(dl)
            else:
                xb = next(dl)

        images = STD_IMG * xb.numpy() + MEAN_IMG
        images = (255 * images).astype('uint8')
        if split in ['train', 'valid']:
            labels = self.train_ds.id_encoder.inverse_transform(
                yb.squeeze().numpy())

        ncols = 4
        nrows = (self.batch_size - 1) // ncols + 1
        fig, axs = plt.subplots(figsize=(12, 3 * nrows),
                                nrows=nrows, ncols=ncols)
        axs = axs.flatten()
        for i in range(self.batch_size):
            axs[i].imshow(images[i])
            if split in ['train', 'valid']:
                axs[i].set_title(labels[i])
        plt.tight_layout()
        return fig, axs
