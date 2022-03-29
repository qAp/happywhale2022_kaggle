
import os, sys, ast
import argparse
from happyid.models.dolg import PRETRAINED
import numpy as np, pandas as pd
import torch, torch.nn as nn
import timm


BACKBONE_NAME = 'resnet18'
PRETRAINED = 'True'
EMBEDDING_SIZE = 512
DROP_RATE = 0


class EmbeddingTimm(nn.Module):
    def __init__(self, data_config={}, args=None):
        super().__init__()

        self.data_config = data_config
        self.args = vars(args) if args is not None else {}

        self.n_classes = self.data_config['num_class']
        self.backbone_name = self.args.get('backbone_name', BACKBONE_NAME)
        self.pretrained = self.args.get(
            'pretrained', ast.literal_eval(PRETRAINED))
        self.drop_rate = self.args.get('drop_rate', DROP_RATE)
        self.embedding_size = self.args.get('embedding_size', EMBEDDING_SIZE)

        self.create_model()

    @staticmethod
    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--backbone_name', type=str, default=BACKBONE_NAME)
        _add('--pretrained', type=ast.literal_eval, default=PRETRAINED)
        _add('--drop_rate', type=float, default=DROP_RATE)
        _add('--embedding_size', type=int, default=EMBEDDING_SIZE)

    def create_model(self):
        self.backbone = timm.create_model(
            self.backbone_name, pretrained=self.pretrained, 
            drop_rate=self.drop_rate)

        self.embedding = nn.Linear(
            self.backbone.get_classifier().in_features, self.embedding_size)

        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

    def forward(self, x):
        features = self.backbone(x)
        emb = self.embedding(features)
        return emb

