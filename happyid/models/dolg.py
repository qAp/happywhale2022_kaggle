
'''
DOLG EfficientNet by Christof Henkel.
https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place/blob/main/models/ch_mdl_dolg_efficientnet.py
'''

import timm
from torch import nn
import torch
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np

from happyid.data.config import *


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)



class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()

        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=3,
                            dilation=dilations[0], padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=3,
                            dilation=dilations[1], padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=3,
                            dilation=dilations[2], padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        # use default setting.
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1)

        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)

        x = att * feature_map_norm
        return x, att_score


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape

        fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))
        fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)
        fg_norm = torch.norm(fg, dim=1)

        fl_proj = (fl_dot_fg / fg_norm[:, None,
                   None, None]) * fg[:, :, None, None]
        fl_orth = fl - fl_proj

        f_fused = torch.cat(
            [fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
        return f_fused


DILATIONS = [3, 6, 9]
PRETRAINED = True
IN_CHANNELS = 3
BACKBONE_NAME = 'resnet18'
STRIDE = (1, 1)
EMBEDDING_SIZE = 512
POOL = 'gem'
GEM_P_TRAINABLE = True
FREEZE = []

class DOLG(nn.Module):
    def __init__(self, data_config, args=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        self.n_classes = NUM_INDIVIDUALS
        self.backbone_name = self.args.get('backbone_name', BACKBONE_NAME)
        self.pretrained = self.args.get('pretrained', PRETRAINED)
        self.in_channels = self.args.get('in_channels', IN_CHANNELS)
        self.stride = self.args.get('stride', STRIDE)
        self.pool = self.args.get('pool', POOL)
        self.gem_p_trainable = self.args.get(
            'gem_p_trainable', GEM_P_TRAINABLE)
        self.embedding_size = self.args.get('embedding_size', EMBEDDING_SIZE)
        self.dilations = self.args.get('dilations', DILATIONS)
        self.freeze = self.args.get('freeze', FREEZE)

        self.create_model()

        if self.freeze:
            self.freeze_weights(freeze=self.freeze)

    @staticmethod
    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--backbone_name', type=str, default=BACKBONE_NAME)
        _add('--pretrained', type=bool, default=PRETRAINED)
        _add('--in_channels', type=int, default=IN_CHANNELS)
        _add('--stride', type=int, nargs='+', default=STRIDE)
        _add('--pool', type=str, default=POOL)
        _add('--gem_p_trainable', type=bool, default=GEM_P_TRAINABLE)
        _add('--embedding_size', type=int, default=EMBEDDING_SIZE)
        _add('--dilations', type=int, nargs='+', default=DILATIONS)
        _add('--freeze', type=str, nargs='+', default=FREEZE)

    def create_model(self):
        self.backbone = timm.create_model(self.backbone_name,
                                          pretrained=self.pretrained,
                                          num_classes=0,
                                          global_pool="",
                                          in_chans=self.in_channels, features_only=True)

        if ("efficientnet" in self.backbone_name) & (self.stride is not None):
            self.backbone.conv_stem.stride = self.stride
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']

        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        if self.pool == "gem":
            self.global_pool = GeM(p_trainable=self.gem_p_trainable)
        elif self.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif self.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = self.embedding_size

        self.neck = nn.Sequential(
            nn.Linear(fusion_out, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU())

        self.head_in_units = self.embedding_size
        self.head = ArcMarginProduct_subcenter(
            self.embedding_size, self.n_classes)

        self.mam = MultiAtrousModule(
            backbone_out_1, feature_dim_l_g, self.dilations)
        self.conv_g = nn.Conv2d(backbone_out, feature_dim_l_g, kernel_size=1)
        self.bn_g = nn.BatchNorm2d(
            feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g = nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()

    def forward(self, x, return_emb=False):

        x = self.backbone(x)

        x_l = x[-2]
        x_g = x[-1]

        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)

        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)

        x_g = self.global_pool(x_g)
        x_g = x_g[:, :, 0, 0]

        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:, :, 0, 0]

        x_emb = self.neck(x_fused)

        if return_emb:
            return x_emb

        logits = self.head(x_emb)
        return logits

    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True
