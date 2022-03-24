

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
        df = pd.read_csv(self.meta_data_path)

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

        

