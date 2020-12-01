#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :flask_utils.py
# @Time     :2020/11/26 下午8:38
# @Author   :Chang Qing
import base64
import datetime
import hashlib
import hmac
import json
import os
import socket
import time
import urllib
import uuid

import cv2
import requests
import numpy as np

from flask import jsonify
from pymongo import MongoClient


def get_time_str():
    timestamp = time.time()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    return time_str


def gen_signature(secret):
    timestamp = round(time.time() * 1000)
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp, sign

def check_security(timestamp, sign, secrets):
    cur_timestamp = time.time() - timestamp
    # time out
    if cur_timestamp - timestamp > 60:
        return False, None
    for secret in secrets:
        # generate candidate sign
        secret_encode = secret.encode("utf-8")
        str_sign = "{}\n{}".format(timestamp, secret)
        str_sign_encode = str_sign.encode("utf-8")
        hmac_code = hmac.new(secret_encode, str_sign_encode, digestmod=hashlib.sha256).digest()
        candidate_sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        # match
        if candidate_sign == sign:
            return True, secret
    return False, None


def log_info(text):
    with open("skymagic_service_log.txt", "a") as f:
        f.write('%s' % datetime.datetime.now())
        f.write('    ')
        f.write(text)
        f.write('\n')
    return


def get_connection(db_params):

    host, port = db_params["host"], db_params["port"]
    db_name, tb_name = db_params["database"], db_params["table"]
    client = MongoClient(host, int(port))
    database = client[db_name]
    table = client[tb_name]
    return table


def write2db(db_params, info):
    collection = get_connection(db_params)
    if type(info) is dict:
        _id = collection.update_one({'_id': info['_id']}, {'$set': info}, upsert=True)
    elif type(info) is list:
        for _ in info:
            _id = collection.update_one({'_id': _['_id']}, {'$set': _}, upsert=True)
    return _id


def download_data_from_url(url, temp_dir, data_basename):
    try:
        data_type = -1
        resp = requests.head(url)
        print(resp)
        if resp.headers.get('content-type').startswith('video'):
            print("video")
            temp_path = os.path.join(temp_dir, "temp_videos", data_basename)
            if os.path.splitext(temp_path)[-1] == "":
                temp_path = temp_path + ".mp4"
            data_type = 1  # video
        else:
            print("image")
            print(temp_dir)
            temp_path = os.path.join(temp_dir, "temp_images", data_basename)
            print(temp_path)
            if os.path.splitext(temp_path)[-1] == "":
                temp_path = temp_path + ".jpg"
            data_type = 0  # image
        content = urllib.request.urlopen(url, timeout=5).read()
        print(temp_path, data_type)
        with open(temp_path, 'wb') as f:
            f.write(content)
        return temp_path, data_type
    except urllib.URLError:
        return None, -2
    except socket.timeout:
        return None, -3
    except Exception:
        return None, -4


# image or video
def url2nparr(data_url, temp_dir, data_basename):
    data_basename = str(uuid.uuid1()) + data_basename
    try:
        # data_type: 0--image 1--video  <1--download error
        temp_path, data_type = download_data_from_url(data_url, temp_dir, data_basename)
        if temp_path is None:
            print("Download error!")
            return None, None
        else:
            return temp_path, data_type
        # req = urllib.request.urlopen(data_url, data)
        # # bytearray() 方法返回一个新字节数组。这个数组里的元素是可变的，并且每个元素的值范围: 0 <= x < 256
        # img_array = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # # 从网络读取图像数据并转换成图片格式
        # image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # return image
    except:
        return None, -1


# only image
def str2nparr(image_str, temp_dir, image_basename):
    image_basename = str(uuid.uuid1()) + image_basename
    temp_path = os.path.join(temp_dir, "temp_images", image_basename)
    if os.path.splitext(temp_path)[-1] == "":
        temp_path = temp_path + ".jpg"
    image_str = base64.b64decode(image_str)
    img_array = np.asarray(bytearray(image_str), dtype=np.uint8)
    # base64str -- > rgb image
    img_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    print(temp_path)
    cv2.imwrite(temp_path, img_rgb)
    return temp_path, 0


# only image
def npstr2nparr(np_str, temp_dir, image_basename):
    image_basename = str(uuid.uuid1()) + image_basename
    temp_path = os.path.join(temp_dir, "temp_images", image_basename)
    if os.path.splitext(temp_path)[-1] == "":
        temp_path = temp_path + ".jpg"
    info = json.loads(np_str)
    size = info['size']
    # frombuffer将data以流的形式读入转化成ndarray对象
    # 第一参数为stream,第二参数为返回值的数据类型
    img_rgb = np.frombuffer(base64.b64decode(info['image']), dtype=np.uint8).reshape(size)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, img_bgr)
    return temp_path, 0



def parse_and_save_data(data, temp_dir):
    """
    parse_and_save_data
    :param data: post data (json)
    :param temp_dir: (./eval_ouput)
    :return: data_path and data_type(image:0, video:1)
    """
    if "name" in data:
        data_basename = data.get("name")
    else:
        data_basename = "test"

    if 'url' in data:
        url = data.get('url')
        data_path, data_type = url2nparr(data.get('url'), temp_dir, data_basename)
        print(data_path, data_type)
        log_info('Get %s image' % url)
    elif 'image' in data:
        # log_info('Got image buffer')

        data_path, data_type = str2nparr(data.get('image'), temp_dir, data_basename)
    elif 'numpy' in data:
        # log_info('Got numpy string')
        data_path, data_type = npstr2nparr(data.get('numpy'), temp_dir, data_basename)
    else:
        return None, -1
    # bgsky_type: 0-image  1:video
    return data_path, data_type


def parse_and_save_bgsky(data, temp_dir):
    """
        parse_and_save_data
        :param data: post data (json)
        :param temp_dir: (./eval_ouput)
        :return: data_path and data_type(image:0, video:1)
        """
    if "bgsky_name" in data:
        data_basename = data.get("bgsky_name")
    else:
        data_basename = "bgsky_test"

    if 'bgsky_url' in data:
        url = data.get('bgsky_url')
        bgsky_path, bgsky_type = url2nparr(data.get('bgsky_url'), temp_dir, data_basename)
        print(bgsky_path, bgsky_type)
        log_info('Get %s sky background image' % url)
    elif 'bgsky_image' in data:
        # log_info('Got image buffer')
        bgsky_path, bgsky_type = str2nparr(data.get('image'), temp_dir, data_basename)
    elif 'bgsky_numpy' in data:
        # log_info('Got numpy string')
        bgsky_path, bgsky_type = npstr2nparr(data.get('numpy'), temp_dir, data_basename)
    else:
        return None, -1
    # bgsky_type: 0-image  1:video
    return bgsky_path, bgsky_type


def error_resp(error_code, error_message):
    resp = jsonify(error_code=error_code, error_message=error_message)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp
