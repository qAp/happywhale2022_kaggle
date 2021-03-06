import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss



def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def shortest_dist(dist_mat):
    """Parallel version.
    Args:
      dist_mat: pytorch Variable, available shape:
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
      dist: three cases corresponding to `dist_mat`:
        1) scalar
        2) pytorch Variable, with shape [N]
        3) pytorch Variable, with shape [*]
    """
    m, n = dist_mat.size()[:2]
    # Just offering some reference for accessing intermediate distance.
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
          if (i == 0) and (j == 0):
              dist[i][j] = dist_mat[i, j]
          elif (i == 0) and (j > 0):
              dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
          elif (i > 0) and (j == 0):
              dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
          else:
              dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]

    dist = dist[-1][-1]
    return dist


def local_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [M, m, d]
      y: pytorch Variable, with shape [N, n, d]
    Returns:
      dist: pytorch Variable, with shape [M, N]
    """
    M, m, d = x.size()
    N, n, d = y.size()
    x = x.contiguous().view(M * m, d)
    y = y.contiguous().view(N * n, d)
    # shape [M * m, N * n]
    dist_mat = euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
    # shape [M, N]
    dist_mat = shortest_dist(dist_mat)
    return dist_mat


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    new_whale_indexs = (labels == 5004 * 2).nonzero()
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    for i in new_whale_indexs:
        is_pos[i, :] = 0
        is_pos[:, i] = 0
        is_pos[i, i] = 1

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        (dist_mat * is_pos.float()).contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    temp = dist_mat * is_neg.float()
    temp[temp == 0] = 10e5
    dist_an, relative_n_inds = torch.min(
        (temp).contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):
        """
        Args:
          dist_ap: pytorch Variable, distance between anchor and positive sample,
            shape [N]
          dist_an: pytorch Variable, distance between anchor and negative sample,
            shape [N]
        Returns:
          loss: pytorch Variable, with shape [1]
        """
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


def global_loss(tri_loss, global_feat, labels, normalize_feature=False):
    """
    Args:
      tri_loss: a `TripletLoss` object
      global_feat: pytorch Variable, shape [N, C]
      labels: pytorch LongTensor, with shape [N]
      normalize_feature: whether to normalize feature to unit length along the
        Channel dimension
    Returns:
      loss: pytorch Variable, with shape [1]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      =============
      For Debugging
      =============
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      ===================
      For Mutual Learning
      ===================
      dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an = hard_example_mining(
        dist_mat, labels, return_inds=False)
    loss = tri_loss(dist_ap, dist_an)
    return loss, dist_ap, dist_an, dist_mat


def local_loss(
        tri_loss,
        local_feat,
        labels=None,
        p_inds=None,
        n_inds=None,
        normalize_feature=False):
    """
    Args:
      tri_loss: a `TripletLoss` object
      local_feat: pytorch Variable, shape [N, H, c] (NOTE THE SHAPE!)
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      labels: pytorch LongTensor, with shape [N]
      normalize_feature: whether to normalize feature to unit length along the
        Channel dimension
    If hard samples are specified by `p_inds` and `n_inds`, then `labels` is not
    used. Otherwise, local distance finds its own hard samples independent of
    global distance.
    Returns:
      loss: pytorch Variable,with shape [1]
      =============
      For Debugging
      =============
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      ===================
      For Mutual Learning
      ===================
      dist_mat: pytorch Variable, pairwise local distance; shape [N, N]
    """
    if normalize_feature:
        local_feat = normalize(local_feat, axis=-1)
    if p_inds is None or n_inds is None:
        dist_mat = local_dist(local_feat, local_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels, return_inds=False)
        loss = tri_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an, dist_mat
    else:
        dist_ap = batch_local_dist(local_feat, local_feat[p_inds])
        dist_an = batch_local_dist(local_feat, local_feat[n_inds])
        loss = tri_loss(dist_ap, dist_an)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim = n_classes

    def forward(self, logits, labels):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1),
                          phi, cosine - mm.view(-1, 1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine



class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", 
                 weight=None, reduction="mean", class_weights_norm=None):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm

        self.crit = nn.CrossEntropyLoss(reduction="none")

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor(
                [45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):

        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()

            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


# Loss definition in popular competition notebooks 

class ArcMarginProduct1(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + \
                self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcLoss1(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 s: float,
                 m: float,
                 easy_margin: bool,
                 ls_eps: float):
        super().__init__()

        self.arc = ArcMarginProduct1(in_features, out_features, s, m,
                                     easy_margin, ls_eps)

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings, targets):
        outputs = self.arc(embeddings, targets)
        loss = self.cross_entropy(outputs, targets)
        return loss

