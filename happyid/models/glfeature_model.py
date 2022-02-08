
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from happyid.data.config import NUM_INDIVIDUALS



class GLFeatureModel(nn.Module):
    '''
    Global and local feature model.
    '''
    def __init__(self, num_classes=NUM_INDIVIDUALS, in_channels=3,
                 backbone='se_resnet50'):
        super().__init__()

        module = importlib.import_module('happyid.models')
        backbone_class = getattr(module, backbone)
        self.backbone = backbone_class(pretrained=None, inchannels=in_channels)
        planes = 2048

        local_planes = 512

        # global feature
        self.bottleneck_g = nn.BatchNorm1d(planes)
        self.bottleneck_g.bias.requires_grad_(False)

        # local feature
        self.local_conv = nn.Conv2d(in_channels=planes,
                                    out_channels=local_planes,
                                    kernel_size=1)
        self.local_bn = nn.BatchNorm2d(num_features=local_planes)
        self.local_bn.bias.requires_grad_(False)
        self.fc = nn.Linear(local_planes, num_classes)

    def forward(self, x):
        feat = self.backbone(x)

        # global feature
        global_feat = F.avg_pool2d(feat, kernel_size=feat.shape[2:])
        global_feat = global_feat.view(feat.shape[0], -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.bottleneck_g(global_feat)
        norm = torch.linalg.vector_norm(
            global_feat, ord=2, dim=1, keepdim=True)
        global_feat = torch.div(global_feat, norm)

        # local feature
        local_feat = feat.mean(dim=3, keepdim=True)
        local_feat = self.local_conv(local_feat)
        local_feat = self.local_bn(local_feat)
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        norm = torch.linalg.vector_norm(
            local_feat, ord=2, dim=-1, keepdim=True)
        local_feat = torch.div(local_feat, norm)

        out = self.fc(local_feat) * 16

        return global_feat, local_feat, out
