#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :inference.py
# @Time     :2020/11/26 上午11:18
# @Author   :Chang Qing

import os
import glob
import warnings
import argparse
from collections import OrderedDict

import cv2
import torch
import numpy as np

import modules.networks as arch_module
from utils import parse_config

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_image(output, output_dir, base_name, shape):
    output_img = output * 255
    # output_img.astype(np.uint8)
    cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.resize(output_img, shape, interpolation=cv2.INTER_CUBIC)
    # base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    cv2.imwrite(output_path, output_img)
    return output_path


def build_best_model(config):
    model = getattr(arch_module, config.arch["type"])(**config.arch["args"])
    best_model_path = config.best_model
    state_dict = torch.load(best_model_path)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def portrait_draw(model, input, output_dir, basename):
    # processing one image
    if isinstance(input, str):
        input_image_path = input
        input_image = cv2.imread(input_image_path)
        print(input_image.shape)
    else:
        input_image = input

    input_image = cv2.resize(input_image, (288, 288))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    input_image_h, input_image_w = input_image.shape[:2]

    # input_tensor = data_process(input_image)
    # data preprocess resize + to tensor
    # 这里要重新定义temp_image, 这样其类型是np.float
    # 如果直接修改原始的input_image 无法得到正确的结果， 因为其类型是byte类型
    temp_image = np.zeros(input_image.shape)
    temp_image[:, :, 0] = (input_image[:, :, 0] / 255.0 - 0.485) / 0.229
    temp_image[:, :, 1] = (input_image[:, :, 1] / 255.0 - 0.456) / 0.224
    temp_image[:, :, 2] = (input_image[:, :, 2] / 255.0 - 0.406) / 0.225
    temp_image = temp_image.transpose([2, 0, 1])
    input_tensor = torch.from_numpy(temp_image).unsqueeze(0).type(torch.FloatTensor).cuda()
    print(input_tensor.shape)
    # build model
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    output_tensors = model(input_tensor)
    output_tensor = normPRED(output_tensors[0])

    output_tensor = output_tensor.repeat([1, 3, 1, 1]).squeeze()

    output_numpy = output_tensor.cpu().data.numpy()
    output_numpy = output_numpy.transpose([1, 2, 0])
    output_path = save_image(output_numpy, output_dir, basename, (input_image_w, input_image_h))
    return output_path


def detect_single_face(face_cascade, ori_img):
    # Convert into grayscale
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        print("Warning: no face detected, the portrait u2net will run on the whole image!")
        return None

    # filer to keep the largest face
    wh = 0
    idx = 0
    for i in range(0, len(faces)):
        x, y, w, h = faces[i]
        if w * h > wh:
            idx = i
            wh = w * h

    return faces[idx]

def crop_face(img, face):
    (x, y, w, h) = face
    height, width = img.shape[0:2]

    l, r, t, b = 0, 0, 0, 0
    lpad = int(float(w) * 0.4)
    left = x - lpad
    if (left < 0):
        l = lpad - x
        left = 0

    rpad = int(float(w) * 0.4)
    right = x + w + rpad
    if (right > width):
        r = right - width
        right = width

    tpad = int(float(h) * 0.6)
    top = y - tpad
    if (top < 0):
        t = tpad - y
        top = 0

    bpad = int(float(h) * 0.2)
    bottom = y + h + bpad
    if (bottom > height):
        b = bottom - height
        bottom = height

    im_face = img[top:bottom, left:right]
    if (len(im_face.shape) == 2):
        im_face = np.repeat(im_face[:, :, np.newaxis], (1, 1, 3))

    im_face = np.pad(im_face, ((t, b), (l, r), (0, 0)), mode='constant',
                     constant_values=((255, 255), (255, 255), (255, 255)))

    # pad to achieve image with square shape for avoding face deformation after resizing
    hf, wf = im_face.shape[0:2]
    if (hf - 2 > wf):
        wfp = int((hf - wf) / 2)
        im_face = np.pad(im_face, ((0, 0), (wfp, wfp), (0, 0)), mode='constant',
                         constant_values=((255, 255), (255, 255), (255, 255)))
    elif (wf - 2 > hf):
        hfp = int((wf - hf) / 2)
        im_face = np.pad(im_face, ((hfp, hfp), (0, 0), (0, 0)), mode='constant',
                         constant_values=((255, 255), (255, 255), (255, 255)))

    # resize to have 512x512 resolution
    im_face = cv2.resize(im_face, (512, 512), interpolation=cv2.INTER_AREA)

    return im_face


def portrait_for_full_image(config):
    # processing input
    model = build_best_model(config)
    if config.input_mode == "image":
        input_path = config.input_path
        output_dir = config.output_dir
        input_basename = os.path.basename(input_path)
        output_path = portrait_draw(model, input_path, output_dir, input_basename)
    else:
        input_dir = config.input_path
        assert os.path.isdir(input_dir)
        output_dir = config.output_dir

        input_paths = glob.glob(os.path.abspath(input_dir + "/*.jpg"))
        input_paths += glob.glob(os.path.abspath(input_dir + "/*.jpeg"))
        input_paths += glob.glob(os.path.abspath(input_dir + "/*.png"))

        output_paths = []
        for input_path in input_paths:
            input_basename = os.path.basename(input_path)
            output_path = portrait_draw(model, input_path, output_dir, input_basename)
            output_paths.append(output_path)
        output_path = os.path.dirname(output_paths[0])
    return output_path


def draw_for_one(model, config, input_path):

    output_dir = config.output_dir
    input_basename = os.path.basename(input_path)

    # load face detection model
    face_model_path = config.face_model_path
    face_cascade = cv2.CascadeClassifier(face_model_path)
    ori_img = cv2.imread(input_path)
    face = detect_single_face(face_cascade, ori_img)
    if face is None:
        output_path = portrait_draw(model, input_path, output_dir, input_basename)
    else:
        input_image = crop_face(ori_img, face)
        output_path = portrait_draw(model, input_image, output_dir, input_basename)
    return output_path


def portrait_for_single_face(config):
    model = build_best_model(config)

    output_path = ""
    if config.input_mode == "image":
        input_path = config.input_path
        output_path = draw_for_one(model, config, input_path)
    else:
        input_dir = config.input_path
        assert os.path.isdir(input_dir)

        input_paths = glob.glob(os.path.abspath(input_dir + "/*.jpg"))
        input_paths += glob.glob(os.path.abspath(input_dir + "/*.jpeg"))
        input_paths += glob.glob(os.path.abspath(input_dir + "/*.png"))
        for input_path in input_paths:
            output_path = draw_for_one(model, config, input_path)
        output_path = os.path.dirname(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Human Portrait Drawing Inference")
    parser.add_argument("--infer_config", type=str, default="./configs/infer_config.json",
                        help="the config file for model inference")
    parser.add_argument("--input_path", type=str, default=None,
                        help="the input path for model inference (image path or dir path)")
    args = parser.parse_args()
    infer_config_path = args.infer_config
    infer_config = parse_config(infer_config_path)
    if args.input_path is not None:
        if os.path.isdir(args.input_path):
            infer_config.input_mode = "dir"
            infer_config.infer_loader["root_dir"] = args.input_path
        else:
            infer_config.input_mode = "image"
            infer_config.input_path = args.input_path

    if not infer_config.face_detect:
        output_path = portrait_for_full_image(infer_config)
    else:
        output_path = portrait_for_single_face(infer_config)

    print(output_path)
