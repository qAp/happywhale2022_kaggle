import math
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from pathlib import Path
from happyid.models.discussion import ArcMarginProduct

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
from happyid.models.discussion import ArcMarginProduct


class LitModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_rate: float,
        embedding_size: int,
        num_classes: int,
        arc_s: float,
        arc_m: float,
        arc_easy_margin: bool,
        arc_ls_eps: float,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        len_train_dl: int,
        epochs: int
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = timm.create_model(
            model_name, pretrained=pretrained, drop_rate=drop_rate)
        self.embedding = nn.Linear(
            self.model.get_classifier().in_features, embedding_size)
        self.model.reset_classifier(num_classes=0, global_pool="avg")

        self.arc = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )

        self.loss_fn = F.cross_entropy

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        embeddings = self.embedding(features)

        return embeddings

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            steps_per_epoch=self.hparams.len_train_dl,
            epochs=self.hparams.epochs,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        images, targets = batch["image"], batch["target"]

        embeddings = self(images)
        outputs = self.arc(embeddings, targets, self.device)

        loss = self.loss_fn(outputs, targets)

        self.log(f"{step}_loss", loss)

        return loss
