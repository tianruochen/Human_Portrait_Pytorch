#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :trainer.py.py
# @Time     :2020/11/23 下午5:12
# @Author   :Chang Qing


import os
import glob
import datetime

import torch
import torchsummary
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import modules.losses as loss
import modules.networks as networks
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, trainer_config):
        self.name = trainer_config.name

        # building model
        self.device, self.gpu_list_ids = self.setup_devices(trainer_config.n_gpu)
        self.model = getattr(networks, trainer_config.arch["type"])(**trainer_config.arch["args"])
        # set up dataloader
        self.train_dataloader, self.valid_dataloader = self.setup_dataloader(trainer_config.train_loader,
                                                                              trainer_config.valid_loader)
        # set up optimizer and lr_scheduler
        self.optimizer = self.setup_optimizer(trainer_config.optimizer)
        self.lr_scheduler = self.setup_lr_schedule(trainer_config.lr_scheduler)

        # set up visualizer
        self.visualization = trainer_config["visualization"]
        # build loss function
        self.loss = self.build_loss(trainer_config.loss)

        # Setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(os.path.abspath(trainer_config.checkpoint_dir),
                                           trainer_config.arch["type"])
        self.models_log_dir = os.path.join(os.path.abspath(trainer_config.visualization['log_dir']),
                                           trainer_config.arch["type"])
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.models_log_dir, exist_ok=True)

        self.n_gpu = trainer_config.n_gpu
        self.start_epoch = 0
        self.epoch_num = trainer_config.epoch_num
        self.resume = trainer_config.resume
        self.checkpoint_dir = trainer_config.checkpoint_dir


    def setup_device(self, n_gpu_need):
        n_gpu_available = torch.cuda.device_count()
        if n_gpu_available == 0 and n_gpu_need > 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_need = 0
        elif n_gpu_need > n_gpu_available:
            n_gpu_need = n_gpu_available
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_need, n_gpu_available))
        device = torch.device("cuda" if n_gpu_need > 0 else "cpu")
        gpu_list_ids = list(range(self.n_gpu))
        return device, gpu_list_ids

    def get_dataset(self, data_dir, is_train):
        abs_data_dir = os.path.abspath(data_dir)
        image_list = glob.glob(abs_data_dir, "*.png")
        return ProtraitDataSet(image_list, is_train)

    def get_dataloader(self):
        return self.train_dataloader, self.valid_dataloader

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_lr_scheduler(self):
        return self.lr_scheduler

    def setup_dataloader(self, train_loader_config, valid_loader_config):
        assert train_loader_config is not None
        train_data_dir = train_loader_config["root_dir"]
        train_data_set = self.get_dataset(train_data_dir, is_train=True)
        train_dataloader = DataLoader(dataset=train_data_set,
                                      shuffle=train_loader_config["shuffle"],
                                      batch_size=train_loader_config["batch_size"],
                                      num_workers=train_loader_config["n_workers"],
                                      pin_memory=train_loader_config["pin_memory"])
        if valid_loader_config is not None:
            valid_data_dir = valid_loader_config["root_dir"]
            valid_data_set = self.get_dataset(valid_data_dir, is_train=False)
            valid_dataloader = DataLoader(dataset=valid_data_set,
                                          shuffle=valid_loader_config["shuffle"],
                                          batch_size=valid_loader_config["batch_size"],
                                          num_workers=valid_loader_config["n_workers"],
                                          pin_memory=valid_loader_config["pin_memory"])
        elif valid_loader_config is None:
            valid_dataloader = None

        return train_dataloader, valid_dataloader

    def build_model(self):
        return getattr(networks, self.arch["type"])(**self.arch["args"])

    def setup_optimizer(self, optimizer_config):
        return getattr(optim, optimizer_config["type"])(self.model.parameters(),
                                                      optimizer_config["args"])

    def setup_lr_schedule(self, lr_scheduler_config):
        return getattr(lr_scheduler, lr_scheduler_config["type"])(self.optimizer, **lr_scheduler_config["args"])

    def build_loss(self, loss_config):
        return getattr(loss, loss_config)

    def train_prepare(self):
        # summary model
        torchsummary.summary(self.model, input_size=(3, 384, 384), device="cpu")

        self.model = self.model.to(self.device)
        if len(self.gpu_list_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_list_ids)

        # resume model for training
        if self.resume is not None:
            self.load_checkpoint(self.resume)

    def train(self):
        pass

    def train_one_epoch(self):
        pass

    def save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__class__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        filename = os.path.join(self.checkpoint_dir, '{}_epoch{}.pth'.format(arch, epoch))
        torch.save(state, filename)
        # if self.save_freq is not None:  # Use None mode to avoid over disk space with large models
        #     if epoch % self.save_freq == 0:
        #         filename = os.path.join(self.checkpoint_dir, 'epoch{}.pth'.format(epoch))
        #         torch.save(state, filename)
        #         self.logger.info("Saving checkpoint at {}".format(filename))
        #
        # # Save the best checkpoint
        # if save_best:
        #     best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        #     torch.save(state, best_path)
        #     print("Saving current best at {}".format(best_path))
        # else:
        #     print("Monitor is not improved from %f" % (self.monitor_best))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["state_dict"])
        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        # 	self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
        # 						'Optimizer parameters not being resumed.')
        # else:
        # 	self.optimizer.load_state_dict(checkpoint['optimizer'])

    # def get_instance(module, name, config, *args):
    #     # module_class = getattr(module, config[name]['type'])
    #     # module_obj = module_class(*args, config['args'])
    #     # return module_obj
    #     return getattr(module, config[name]['type'])(*args, **config[name]['args'])
