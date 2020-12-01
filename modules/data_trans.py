#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :datatransform.py
# @Time     :2020/11/23 下午7:31
# @Author   :Chang Qing
"""
transforms for cv image
"""
import cv2
import random
import torch
import numpy as np
from torchvision import transforms


class Rescale:
    """
    Rescale for cv image
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_input = sample["image_input"]
        image_label = sample["image_label"]
        ori_h, ori_w = image_input.shape[:2]

        if isinstance(self.output_size, tuple):
            new_h, new_w = self.output_size
        if isinstance(self.output_size, int):
            if ori_h > ori_w:
                new_w, new_h = self.output_size, self.output_size * ori_h / ori_w
            else:
                new_w, new_h = self.output_size * ori_w / ori_h, self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image_input = cv2.resize(image_input, (new_w, new_h))
        image_label = cv2.resize(image_label, (new_w, new_h))
        sample["image_input"] = image_input
        sample["image_label"] = image_label

        return sample



class RandomCrop:
    """
    RandomCrop for cv image
    """
    def __init__(self, crop_size):
        assert isinstance(crop_size, (int, tuple))
        self.crop_size = crop_size

    def __call__(self, sample):
        image_input, image_label = sample["image_input"], sample["image_label"]
        ori_h, ori_w = image_input.shape[:2]
        if isinstance(self.crop_size, int):
            new_h, new_w = self.crop_size, self.crop_size
        elif isinstance(self.crop_size, tuple):
            new_h, new_w = self.crop_size[0], self.crop_size[1]
        assert ori_w > new_w
        assert ori_h > new_h
        start_h = random.randint(0, ori_h - new_h)
        start_w = random.randint(0, ori_w - new_w)
        image_input = image_input[start_h:start_h + new_h, start_w:start_w + new_w]
        image_label = image_label[start_h:start_h + new_h, start_w:start_w + new_w]
        sample["image_input"] = image_input
        sample["image_label"] = image_label
        return sample


class ToTensor:
    """
    totensor for cv image:
        image_input: 0-255 --> 0-1 and normalize and convert to tensor
        image_label: 0-255 --> 0-1 and noly need one channle (m,n,3) -- (m,n,1) and convert to tensor
    """
    def __call__(self, sample):
        image_input, image_label = sample["image_input"], sample["image_label"]
        # bgr to rgb
        image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        # normalize
        temp_image = np.zeros(image_input.shape)
        temp_label = np.zeros((image_label.shape[0], image_label.shape[1], 1))
        temp_image[:, :, 0] = (image_input[:, :, 0] / 255.0 - 0.485) / 0.229
        temp_image[:, :, 1] = (image_input[:, :, 1] / 255.0 - 0.456) / 0.224
        temp_image[:, :, 2] = (image_input[:, :, 2] / 255.0 - 0.406) / 0.225
        temp_label[:, :, 0] = image_label[:, :, 0] / 255.0   # only need one channel

        # (h,w,c)-> (c,h,w)

        temp_image = temp_image.transpose((2, 0, 1))
        temp_label = temp_label.transpose((2, 0, 1))

        # numpy to tensor
        image_input = torch.from_numpy(temp_image)
        image_label = torch.from_numpy(temp_label)

        sample["image_input"] = image_input
        sample["image_label"] = image_label
        return sample


def get_transform(is_train=True):
    if is_train:
        transform = transforms.Compose([
            Rescale(320),
            RandomCrop(288),
            ToTensor()
        ])
    else:
        transform = transforms.Compose([
            Rescale(288),
            ToTensor()
        ])
    return transform
