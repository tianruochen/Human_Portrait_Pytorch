#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :dataloader.py
# @Time     :2020/11/24 下午3:22
# @Author   :Chang Qing

import os
import glob
import cv2
import numpy as np
import torch
import torchvision
from modules.data_trans import get_transform
from torch.utils.data import Dataset, DataLoader


class ProtraitDataSet(Dataset):
    """
    Human protrait Drawing Data Set
    """

    def __init__(self, image_list, is_train=True):
        self.image_list = image_list
        self.is_train = is_train
        self.transform = get_transform(self.is_train)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image_full = cv2.imread(image_path)
        h, w, c = image_full.shape
        image_index = np.array([idx])
        # image_input and image_label are both three channels
        image_input = image_full[:, 0:w // 2, :]
        image_label = image_full[:, w // 2:w, :]

        sample = {
            "image_index": image_index,
            # "image_path": image_path,
            "image_input": image_input,
            "image_label": image_label
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_dataset(data_dir, is_train):
    abs_data_dir = os.path.abspath(data_dir)
    print(abs_data_dir)
    image_list = glob.glob(os.path.join(abs_data_dir, "*.png"))
    print(image_list)
    return ProtraitDataSet(image_list, is_train)


def get_dataloader(train_loader_config, valid_loader_config):
    assert train_loader_config is not None
    train_data_dir = train_loader_config["root_dir"]
    train_data_set = get_dataset(train_data_dir, is_train=True)
    train_data_num = len(train_data_set)
    print(train_data_num)
    valid_dataloader = None
    valid_data_num = 0
    train_dataloader = DataLoader(dataset=train_data_set,
                                  shuffle=train_loader_config["shuffle"],
                                  batch_size=train_loader_config["batch_size"],
                                  num_workers=train_loader_config["n_workers"],
                                  pin_memory=train_loader_config["pin_memory"])
    if valid_loader_config is not None:
        valid_data_dir = valid_loader_config["root_dir"]
        valid_data_set = get_dataset(valid_data_dir, is_train=False)
        valid_data_num = len(valid_data_set)
        print(valid_data_num)
        valid_dataloader = DataLoader(dataset=valid_data_set,
                                      shuffle=valid_loader_config["shuffle"],
                                      batch_size=valid_loader_config["batch_size"],
                                      num_workers=valid_loader_config["n_workers"],
                                      pin_memory=valid_loader_config["pin_memory"])

    return train_dataloader, valid_dataloader, train_data_num, valid_data_num
