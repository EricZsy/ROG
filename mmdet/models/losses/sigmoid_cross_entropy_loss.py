import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .accuracy import accuracy


@LOSSES.register_module()
class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 num_classes=1203,
                 ):
        super(SigmoidCrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.custom_cls_channels = True
        self.custom_activation = True

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)
        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target, reduction='none')
        cls_loss = torch.sum(cls_loss) / self.n_i
        return self.loss_weight * cls_loss

    def get_cls_channels(self, num_classes):
        num_classes = num_classes + 1
        return num_classes

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        acc = dict()
        obj_labels = (labels == self.num_classes).long()  # 0 fg, 1 bg
        acc_objectness = accuracy(torch.cat([1 - cls_score[:, -1:], cls_score[:, -1:]], dim=1), obj_labels)
        acc_classes = accuracy(cls_score[:, :-1][pos_inds], labels[pos_inds])

        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc