#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :server_test.py
# @Time     :2020/11/18 上午10:14
# @Author   :Chang Qing

import json
import base64
import requests
from utils import gen_signature


def server_test(secret):
    with open("samples/inputs/girl1.jpg", "rb") as f:
        buff = f.read()
    b64_image_str = base64.b64encode(buff)
    b64_image_str = b64_image_str.decode('ascii')
    t, s = gen_signature(secret)
    print(type(b64_image_str),type(t),type(s))
    payload = {
        "url": "http://tbvideo.ixiaochuan.cn/zyvdorigine/b4/15/f057-f5f9-4acc-9b9b-26b77abf1532",
        #"url": "https://ns-strategy.cdn.bcebos.com/ns-strategy/upload/fc_big_pic/part-00147-2263.jpg",
        #"image": b64_image_str,
        "timestamp": t,
        "sign": s
    }
    resp = requests.post('http://0.0.0.0:6606/api/magic_sky', data=json.dumps(payload))
    print(json.loads(resp.text))

if __name__ == "__main__":
    secret = 'wbrlcmdxifksqvnzhoytpaeug'
    server_test(secret)