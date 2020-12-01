#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :common_utils.py.py
# @Time     :2020/11/23 下午4:55
# @Author   :Chang Qing

import json

def parse_config(config_path):
    config = json.load(open(config_path))
    print(type(config))

    class Struct:
        def __init__(self, entries):
            self.__dict__.update(entries)

    return Struct(config)


# def get_instance(module, name, config, *args):
#     # module_class = getattr(module, config[name]['type'])
#     # module_obj = module_class(*args, config['args'])
#     # return module_obj
#     return getattr(module, config[name]['type'])(*args, **config[name]['args'])

if __name__ == "__main__":
    config = parse_config("../configs/model_config.json")
    config.loss = "mse"
    print(config.__dict__)

