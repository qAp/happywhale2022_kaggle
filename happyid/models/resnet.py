

import timm
from happyid.data.config import *




class Resnet18:
    def __init__(self):
        self = timm.create_model(
            model_name='resnet18',
            pretrained=True,
            num_classes=NUM_INDIVIDUALS)
