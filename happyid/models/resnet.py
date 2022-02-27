

import timm
from happyid.data.config import *




def Resnet18():
    return timm.create_model(
        model_name='resnet18',
        pretrained=True,
        num_classes=NUM_INDIVIDUALS)
