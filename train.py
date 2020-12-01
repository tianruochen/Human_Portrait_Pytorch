#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :main.py.py
# @Time     :2020/11/23 下午4:46
# @Author   :Chang Qing

import logging
import warnings
import argparse

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import modules.networks as networks
import modules.losses as loss_module
import modules.metrics as metric_module
from modules.trainer import Trainer
from modules.dataloader import get_dataloader

from utils import parse_config

warnings.filterwarnings("ignore")


def train(model_config, logger_config):
    # print and save config
    # loger_trainer_config(trainer_config, logger_config)

    # set up logger
    # set_up_logger(logger_config)
    # train_logger = logging.getLogger()

    # set up dataset
    # train_dataset = trainer.get_dataset(trainer_config.dataset, train=True)
    # valid_dataset = trainer.get_dataset(trainer_config.dataset, trian=False)

    # set up dataloader
    train_dataloader, valid_dataloader, train_data_num, valid_data_num = get_dataloader(
        model_config.train_loader, valid_loader_config=model_config.valid_loader)

    # build model
    model = getattr(networks, model_config.arch["type"])(**model_config.arch["args"])

    # get optimizer
    optimizer = getattr(optim, model_config.optimizer["type"])(model.parameters(),
                                                               **model_config.optimizer["args"])

    # get lr scheduler
    scheduler = getattr(lr_scheduler, model_config.lr_scheduler["type"])(optimizer,
                                                                         **model_config.lr_scheduler["args"])

    # loss function
    loss = getattr(loss_module, model_config.loss)
    metrics = [getattr(metric_module, metric) for metric in model_config.metrics]

    # set up trainer
    trainer = Trainer(model, loss, metrics, optimizer,
                      model_config=model_config,
                      train_num=train_data_num,
                      valid_num=valid_data_num,
                      train_loader=train_dataloader,
                      valid_loader=valid_dataloader,
                      lr_scheduler=scheduler)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Portrait Drawing")
    parser.add_argument("--model_config", type=str, default="./configs/model_config.json",
                        help="the trainer config path for training model")
    parser.add_argument("--logger_config", type=str, default="./configs/logger_config.json",
                        help="the logger config path for logging model")
    parser.add_argument("--resume", type=str, default=None,
                        help="the checkpoint path for retrain model")
    args = parser.parse_args()

    model_config_path = args.model_config
    logger_config_path = args.logger_config
    model_config = parse_config(model_config_path)
    logger_config = parse_config(logger_config_path)

    if args.resume is not None:
        model_config.__dict__.update(resume=args.resume)
    train(model_config, logger_config)
