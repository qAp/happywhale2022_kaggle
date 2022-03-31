
'''
Definitions widely used on Discussion.
'''
import math
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from timm.data.transforms_factory import create_transform
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder



INPUT_DIR = Path("..") / "input"
OUTPUT_DIR = Path("/") / "kaggle" / "working"

DATA_ROOT_DIR = INPUT_DIR / "convert-backfintfrecords" / \
    "happy-whale-and-dolphin-backfin"
TRAIN_DIR = DATA_ROOT_DIR / "train_images"
TEST_DIR = DATA_ROOT_DIR / "test_images"
TRAIN_CSV_PATH = DATA_ROOT_DIR / "train.csv"
SAMPLE_SUBMISSION_CSV_PATH = DATA_ROOT_DIR / "sample_submission.csv"
PUBLIC_SUBMISSION_CSV_PATH = INPUT_DIR / \
    "0-720-eff-b5-640-rotate" / "submission.csv"
IDS_WITHOUT_BACKFIN_PATH = INPUT_DIR / \
    "ids-without-backfin" / "ids_without_backfin.npy"

N_SPLITS = 5

ENCODER_CLASSES_PATH = OUTPUT_DIR / "encoder_classes.npy"

CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
SUBMISSION_CSV_PATH = OUTPUT_DIR / "submission.csv"


DEBUG = False


def get_image_path(id: str, dir: Path) -> str:
    return f"{dir / id}"


class HappyWhaleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        self.transform = transform

        self.image_names = self.df["image"].values
        self.image_paths = self.df["image_path"].values
        self.targets = self.df["individual_id"].values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_name = self.image_names[index]

        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": image_name, "image": image, "target": target}

    def __len__(self) -> int:
        return len(self.df)


TRAIN_CSV_ENCODED_FOLDED_PATH = OUTPUT_DIR / "train_encoded_folded.csv"
TEST_CSV_PATH = OUTPUT_DIR / "test.csv"
VAL_FOLD = 0
IMAGE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 2
K = 50

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_encoded_folded: str,
        test_csv: str,
        val_fold: float,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_encoded_folded)
        self.test_df = pd.read_csv(test_csv)

        self.transform = create_transform(
            input_size=(self.hparams.image_size, self.hparams.image_size),
            crop_pct=1.0,
        )

    @staticmethod
    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--train_csv_encoded_folded', type=str,
             default=str(TRAIN_CSV_ENCODED_FOLDED_PATH))
        _add('--test_csv', type=str, default=str(TEST_CSV_PATH))
        _add('--val_fold', type=float, default=VAL_FOLD)
        _add('--image_size', type=int, default=IMAGE_SIZE)
        _add('--batch_size', type=int, default=BATCH_SIZE)
        _add('--num_workers', type=int, default=NUM_WORKERS)

    def prepare_data(self):
        train_df = pd.read_csv(TRAIN_CSV_PATH)

        train_df["image_path"] = train_df["image"].apply(get_image_path, dir=TRAIN_DIR)

        encoder = LabelEncoder()
        train_df["individual_id"] = encoder.fit_transform(train_df["individual_id"])
        np.save(ENCODER_CLASSES_PATH, encoder.classes_)

        skf = StratifiedKFold(n_splits=N_SPLITS)
        for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.individual_id)):
            train_df.loc[val_, "kfold"] = fold

        train_df.to_csv(TRAIN_CSV_ENCODED_FOLDED_PATH, index=False)

        train_df.head()

        # Use sample submission csv as template
        test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
        test_df["image_path"] = test_df["image"].apply(get_image_path, dir=TEST_DIR)

        test_df.drop(columns=["predictions"], inplace=True)

        # Dummy id
        test_df["individual_id"] = 0

        test_df.to_csv(TEST_CSV_PATH, index=False)

        test_df.head()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Split train df using fold
            train_df = self.train_df[self.train_df.kfold !=
                                     self.hparams.val_fold].reset_index(drop=True)
            val_df = self.train_df[self.train_df.kfold ==
                                   self.hparams.val_fold].reset_index(drop=True)

            self.train_dataset = HappyWhaleDataset(
                train_df, transform=self.transform)
            self.val_dataset = HappyWhaleDataset(
                val_df, transform=self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = HappyWhaleDataset(
                self.test_df, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: HappyWhaleDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )


