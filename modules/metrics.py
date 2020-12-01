#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :metrics.py
# @Time     :2020/11/24 下午3:51
# @Author   :Chang Qing

import torch
from torch.nn import functional as F


def miou(outputs, targets, eps=1e-6):
    """
    compute the miou value between outputs and targets
    :param outputs: (torch.float32)  shape (N, 1, H, W)  after sigmoid function
    :param targets: targets: (torch.float32) shape (N, 1, H, W), value {0,1,...,C-1}
    :param eps: default(1e-6)
    :return: average miou
    """

    outputs[outputs > 0.5] = 1
    outputs[outputs <= 0.5] = 0
    targets[targets > 0.5] = 1
    targets[targets <= 0.5] = 0

    outputs = outputs.type(torch.int64)
    targets = targets.type(torch.int64)

    inter = (outputs & targets).type(torch.float32).sum(dim=(1, 2))
    union = (outputs | targets).type(torch.float32).sum(dim=(1, 2))
    iou = inter / (union + eps)
    return iou.mean()
