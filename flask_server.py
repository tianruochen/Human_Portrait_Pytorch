#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :flask_server.py
# @Time     :2020/11/26 下午7:57
# @Author   :Chang Qing


import argparse
import json
import os
import time

from waitress import serve
from flask import Flask, request, jsonify

from inference import portrait_for_full_image, portrait_for_single_face
from utils import parse_config, parse_and_save_data
from utils import error_resp, check_security
from utils.flask_utils import log_info, get_time_str

app = Flask(__name__)


@app.route("/api/portrait_drawing", methods=["POST"])
def portrait_drawing():
    if request.method == "POST":
        data = json.loads(request.data)
        if "timestamp" not in data and "sign" not in data:
            return error_resp(1, "parameters miss, timestamp and sign is necessary for current service")

        # check secrets
        secure, secret = check_security(data["timestamp"], data["sign"], app.config["secrets"])
        if not secure:
            return error_resp(1, "you need a right signature before post a request")

            # get the image or video code and path
        temp_dir = server_config.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        data_path, data_type = parse_and_save_data(data, temp_dir)
        print("data_path:", data_path)
        print("data_type:", data_type)

        if data_type == 0:
            # processing image: modify sky config file
            server_config["input_mode"] = "image"

        server_config["input_path"] = data_path

        # portrait drawing
        log_info("%s : begin..." % get_time_str())
        if not server_config.face_detect:
            res_path = portrait_for_full_image(server_config)
        else:
            res_path = portrait_for_single_face(server_config)
        log_info("%s : end..." % get_time_str())
        # ndarray to str  .__str__()
        data_path_hash = hash(str(time.time()) + data_path)
        # write to db
        db_info = {'_id': data_path_hash,
                   'service': secret,
                   'req_time': str(get_time_str()),
                   'res_path': res_path,
                   "url": None
                   }
        print(db_info)
        # write2db(db_info)
        # log_info('Write to db %s' % data.get('url'))

        resp = jsonify(error_code=0, data=db_info)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        log_info('Edit %s %s done' % (data_path_hash,
                                      '' if db_info['url'] is None else db_info['url']))
        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Portrait Drawing Server")
    parser.add_argument("--server_config", type=str, default="./configs/server_config.json",
                        help="the path of server config file")
    parser.add_argument("--secret_config", type=str, default="./configs/secret_config.json",
                        help="the path of secret config file")
    parser.add_argument("--port", type=str, default=6606,
                        help="service port (default:6606)")

    args = parser.parse_args()
    server_config_path = args.server_config
    server_config = parse_config(server_config_path)

    if args.port:
        server_config.port = args.port
        app.config["port"] = server_config.port
    if args.secret_config:
        app.config["secrets"] = json.load(args.secret_config)

    serve(app, host="0.0.0.0", port=int(server_config.port), threads=3)