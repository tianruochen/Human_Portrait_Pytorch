#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :trainer.py.py
# @Time     :2020/11/23 下午5:12
# @Author   :Chang Qing


import os
import math
import json
import time
import logging
import datetime

import torch
import numpy as np
from tqdm import tqdm

from torchvision.utils import make_grid
from utils import summary_model
from utils import WriterTensorboardX


# ------------------------------------------------------------------------------
#  Poly learning-rate Scheduler
# ------------------------------------------------------------------------------
def poly_lr_scheduler(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    for g in optimizer.param_groups:
        g['lr'] = init_lr * (1 - curr_iter / max_iter) ** power


class Trainer:
    def __init__(self, model, loss, metrics, optimizer, model_config,
                 train_num, valid_num, train_loader, valid_loader, lr_scheduler):

        self.model_config = model_config
        self.trainer_config = self.model_config.trainer
        self.epoch_num = self.trainer_config["epoch_num"]
        self.save_freq = self.trainer_config["save_freq"]
        self.verbosity = self.trainer_config["verbosity"]
        self.monitor = self.trainer_config["monitor"]
        self.monitor_mode = self.trainer_config["monitor_mode"]
        self.monitor_best = math.inf if self.monitor_mode == "min" else -math.inf
        self.resume = self.trainer_config["resume"]
        self.do_validation = self.trainer_config["do_validation"]
        # self.save_config = self.model_config.save_model_config

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.train_num = train_num
        self.valid_num = valid_num
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler

        self.max_iter = len(self.train_loader) * self.epoch_num
        self.init_lr = optimizer.param_groups[0]['lr']

        self.start_epoch = 1
        self.n_gpu_need = self.trainer_config["n_gpu"]
        self.device, self.gpu_list_ids = self.setup_device(self.n_gpu_need)

        start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(os.path.abspath(self.trainer_config["checkpoint_dir"]),
                                           self.model.__class__.__name__, start_time)
        self.model_log_dir = os.path.join(os.path.abspath(self.trainer_config["log_dir"]),
                                          self.model.__class__.__name__, start_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.model_log_dir, exist_ok=True)

        self.setup_logger()
        self.logger = logging.getLogger()

        # setup visualization writer instance
        writer_train_dir = os.path.join(self.model_log_dir, "train")
        writer_valid_dir = os.path.join(self.model_log_dir, "valid")
        self.writer_train = WriterTensorboardX(writer_train_dir, self.logger, self.trainer_config["tensorboardX"])
        self.writer_valid = WriterTensorboardX(writer_valid_dir, self.logger, self.trainer_config["tensorboardX"])
        self.results_log = {}
        self.train_prepare()

    def setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s-%(levelname)s-%(filename)s-%(lineno)s-%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.model_log_dir, "train.log")),
                logging.StreamHandler()
            ]
        )

    def setup_device(self, n_gpu_need):
        n_gpu_available = torch.cuda.device_count()
        if n_gpu_available == 0 and n_gpu_need > 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_need = 0
        elif n_gpu_need > n_gpu_available:
            n_gpu_need = n_gpu_available
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_need, n_gpu_available))
        device = torch.device("cuda" if n_gpu_need > 0 else "cpu")
        gpu_list_ids = list(range(n_gpu_need))
        return device, gpu_list_ids

    def train_prepare(self):

        # save model config to model_log_dir
        if self.model_config.save_model_config:
            config_save_path = os.path.join(self.model_log_dir, "model_config.json")
            with open(config_save_path, "w") as f:
                json.dump(self.model_config.__dict__, f, sort_keys=False, indent=4)

        # torchsummary.summary(self.model, input_size=(3, 384, 384), device="cpu")
        summary_model(self.model, input_size=(3, 288, 288), device="cpu", logger=self.logger)

        self.model = self.model.to(self.device)
        if len(self.gpu_list_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_list_ids)

        # resume model for training
        if self.resume is not None:
            self._resume_checkpoint(self.resume)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] = metric(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Trianing logic for an epoch
        :param epoch: current epoch
        :return: a dict that contains all information that you want to save
        """
        print("Train on epoch...")
        self.model.train()
        self.writer_train.set_step(epoch)

        # Perform training
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        batch_count = len(self.train_loader)
        for batch_idx, batch_data in tqdm(enumerate(self.train_loader), total=batch_count):
            curr_iter = batch_idx + (epoch - 1) * batch_count
            batch_input = batch_data["image_input"].type(torch.FloatTensor)
            # batch_size = batch_input.shape[0]

            batch_label = batch_data["image_label"].type(torch.FloatTensor)
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_label = batch_label.cuda()

            # temp_image[:, :, 0] = (image_input[:, :, 0] / 255.0 - 0.485) / 0.229
            # temp_image[:, :, 1] = (image_input[:, :, 1] / 255.0 - 0.456) / 0.224
            # temp_image[:, :, 2] = (image_input[:, :, 2] / 255.0 - 0.406) / 0.225
            ori_input = torch.zeros_like(batch_input)
            ori_input[:, 0, :, :] = (batch_input[:, 0, :, :] * 0.229 + 0.485) * 255.0
            ori_input[:, 1, :, :] = (batch_input[:, 1, :, :] * 0.224 + 0.456) * 255.0
            ori_input[:, 2, :, :] = (batch_input[:, 2, :, :] * 0.225 + 0.406) * 225.0
            self.optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = self.model(batch_input)
            loss2, loss = self.loss(d0, d1, d2, d3, d4, d5, d6, batch_label)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            output, target = d0, batch_label
            total_metrics += self._eval_metrics(output, target)

            # expand the channels of output and target for tensorboard show
            output = output.repeat([1, 3, 1, 1])
            target = target.repeat([1, 3, 1, 1])

            if batch_idx == batch_count - 2 and self.verbosity >= 2:
                self.writer_train.add_image("train/ori_input",
                                            make_grid(ori_input[:, :3, :, :].cpu(), nrow=4, normalize=False))
                self.writer_train.add_image("train/input",
                                            make_grid(batch_input[:, :3, :, :].cpu(), nrow=4, normalize=False))
                self.writer_train.add_image("train/label",
                                            make_grid(target.cpu(), nrow=4, normalize=False))
                self.writer_train.add_image("train/output",
                                            make_grid(output.cpu(), nrow=4, normalize=True))

            poly_lr_scheduler(self.optimizer, self.init_lr, curr_iter, self.max_iter, power=0.9)

        # Record log
        total_loss /= len(self.train_loader)
        total_metrics /= len(self.train_loader)

        results = {
            "train_loss": total_loss,
            "train_metrics": total_metrics
        }

        self.writer_train.add_scalar("loss", total_loss)
        for i, metric in enumerate(self.metrics):
            self.writer_train.add_scalar("metrics/%s" % metric.__name__, total_metrics[i])
        for i in range(len(self.optimizer.param_groups)):
            self.writer_train.add_scalar('lr/group%d' % i, self.optimizer.param_groups[i]['lr'])

        # Perform validation
        if self.do_validation:
            print("Validate on epoch")
            val_results = self._valid_epoch(epoch)
            results = {**results, **val_results}

        # Leraning rate schedule
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results

    def _valid_epoch(self, epoch):
        """
        Validate after train one epoch
        :return: A dict that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        val_batch_count = len(self.valid_loader)
        self.writer_valid.set_step(epoch)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_loader):
                batch_input = batch_data["image_input"].type(torch.FloatTensor)
                batch_label = batch_data["image_label"].type(torch.FloatTensor)
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                    batch_label = batch_label.cuda()

                ori_input = torch.zeros_like(batch_input)
                ori_input[:, 0, :, :] = (batch_input[:, 0, :, :] * 0.229 + 0.485)   #* 255.0
                ori_input[:, 1, :, :] = (batch_input[:, 1, :, :] * 0.224 + 0.456)   #* 255.0
                ori_input[:, 2, :, :] = (batch_input[:, 2, :, :] * 0.225 + 0.406)   #* 225.0

                output, d1, d2, d3, d4, d5, d6 = self.model(batch_input)
                _, loss = self.loss(output, d1, d2, d3, d4, d5, d6, batch_label)

                total_val_loss += loss
                total_val_metrics += self._eval_metrics(output, batch_label)

                batch_label = batch_label.repeat([1, 3, 1, 1])
                output = output.repeat([1, 3, 1, 1])

                # log image for input, label, output
                if (batch_idx == val_batch_count - 2) and self.verbosity >= 2:
                    self.writer_train.add_image("train/ori_input",
                                                make_grid(ori_input[:, :3, :, :].cpu(), nrow=4, normalize=False))
                    self.writer_valid.add_image("valid/image", make_grid(batch_input[:, :3, :, :].cpu(),
                                                                         nrow=4, normalize=False))
                    self.writer_valid.add_image("valid/label", make_grid(batch_label.cpu(),
                                                                         nrow=4, normalize=False))
                    self.writer_valid.add_image("valid/output", make_grid(output.cpu(),
                                                                          nrow=4, normalize=True))
            total_val_loss /= len(self.valid_loader)
            total_val_metrics /= len(self.valid_loader)

            # log valid results
            valid_results = {
                "valid_loss": total_val_loss,
                "valid_metric": total_val_metrics
            }
            self.writer_valid.add_scalar("loss", total_val_loss)
            for i in range(len(self.metrics)):
                self.writer_valid.add_scalar("metric_{}".format(self.metrics[i].__name__), total_val_metrics[i])

            return valid_results

    def train(self):

        for epoch in range(self.start_epoch, self.epoch_num):
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("[EPOCH %d]", epoch)
            start_time = time.time()
            results = self._train_epoch(epoch)
            end_time = time.time()
            self.logger.info(
                "Finish at {}, Runtime: {:.3f} [s]".format(datetime.datetime.now(), end_time - start_time)
            )

            # save logged information in result_log dict

            for key, value in results.items():
                if key == "train_metrics":
                    self.results_log.update({"train_" + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == "valid_metrics":
                    self.results_log.update({"valid_" + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    self.results_log[key] = value

            # print logged information to the screen and file
            if self.verbosity >= 1:
                for key, value in sorted(list(self.results_log.items())):
                    self.logger.info("{:25s}:{}".format(key, value))

            best = False
            if self.monitor_mode != "off":
                try:
                    if (self.monitor_mode == "min" and self.results_log[self.monitor] < self.monitor_best) or \
                            (self.monitor_mode == "max" and self.results_log[self.monitor] > self.monitor_best):
                        self.logger.info(
                            "Monitor improved from %f to %f" % (self.monitor_best, self.results_log[self.monitor]))
                        self.monitor_best = self.results_log[self.monitor]
                        best = True
                except KeyError:
                    if epoch == 1:
                        msg = f"Warning: Can\'t recognize metric named {self.monitor} for \
                                        performance monitoring. model_best checkpoint won\'t be update"
                        self.logger.warning(msg)

            self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best):
        arch = type(self.model).__class__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.monitor_best,
            "result_log": self.results_log,
            "model_config": self.model_config.__dict__
        }

        # save checkpoint for fix epoch
        # default self.save_freq is None to avoid many disk cost
        if self.save_freq is not None:
            if epoch % self.save_freq == 0:
                filename = os.path.join(self.checkpoint_dir,
                                        '{}_epoch{}_{}{:.4}.pth'.format(arch, epoch, self.monitor,
                                                                        self.results_log[self.monitor]))
                torch.save(state, filename)
                self.logger.info("Saving checkpoint at {}".format(filename))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best at {}".format(best_path))
        else:
            self.logger.info("Monitor is not improved from {}".format(self.monitor_best))

    def _resume_checkpoint(self, checkpoint_path):
        self.logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint["epoch"]
        self.monitor_best = checkpoint["monitor_best"]

        if checkpoint["config"].arch != self.model_config.arch:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.result_log = checkpoint["result_log"]
        self.model_config = checkpoint["model_config"]
        self.logger.info(f"Checkpoint {checkpoint_path} (epoch {self.start_epoch}) loaded!")


