#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :logger_utils.py
# @Time     :2020/11/23 下午5:24
# @Author   :Chang Qing

import os
import json
import logging
import logging.config

from pprint import pprint


class ResultLogger:

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries)] = entry

    def __str__(self):
        json.dumps(self.entries, sort_keys=True, indent=4)


def loger_trainer_config(trianer_config):
    pprint(trianer_config.__dict__)


def set_up_logger(logger_config_path, default_level=logging.INFO):
    if os.path.exists(logger_config_path):
        with open(logger_config_path, "r") as f:
            logger_config = json.load(f)
            logging.config.dictConfig(logger_config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == "__main__":
    set_up_logger("../configs/logger_config.json")
    logger = logging.getLogger()
    print(logger)



