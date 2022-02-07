import torch
from torch import nn
import torch.nn.functional as F
import math

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, factor=0.1, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.factor = factor

    def forward(self, input, target):
        # num_classes = input.shape[-1]
        # target = smooth(target.float(), num_classes, self.factor)
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

"""
https://github.com/dongkyuk/DOLG-pytorch/blob/main/model/arcface.py
"""
class ArcFaceLoss(nn.Module):
    def __init__(self, scale_factor=45.0, margin=0.10, criterion=None, weight=None):
        super(ArcFaceLoss, self).__init__()

        if criterion:
            self.criterion = criterion
        else:
            if weight:
                self.criterion = nn.CrossEntropyLoss(weight=weight)
            else:
                self.criterion = nn.CrossEntropyLoss()
        self.margin = margin
        self.scale_factor = scale_factor


        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, logits, label):
        # input is not l2 normalized
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=logits.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        loss = self.criterion(logit, label)

        return loss