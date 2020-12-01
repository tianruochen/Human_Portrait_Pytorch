#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :loss.py.py
# @Time     :2020/11/24 上午11:53
# @Author   :Chang Qing

import torch.nn as nn

bce_loss = nn.BCELoss(size_average=True)


def multi_bce_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    print(d0.shape, labels_v.shape)
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss
