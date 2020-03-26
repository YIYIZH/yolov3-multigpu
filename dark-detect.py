#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/3/23 15:23 
 @Author : ZHANG 
 @File : dark-detect.py 
 @Description:
"""

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

def detect():
    img_size = 416
    cfg = 'cfg/yolov3.cfg'
    names = 'data/coco.names'
    weights = 'weights/yolov3.weights'
    source = 'data/samples'
    conf_thres = 0.3
    iou_thres = 0.4
    classes = [0, 1, 2, 3, 5, 7]
    img_size = (320, 192) if ONNX_EXPORT else img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

    # Initialize
    device = torch.device('cuda')

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)


    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det_cpu = det.detach().cpu().numpy().copy()
                # map coco class id to ours
                for i in range(det_cpu.shape[0]):
                    if det_cpu[i,-1] == 5:
                        det_cpu[i,-1] = 4
                    if det_cpu[i,-1] == 7:
                        det_cpu[i,-1] = 5
        return det_cpu

detect()