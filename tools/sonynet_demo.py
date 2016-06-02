#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'animal:bear', 'animal:bird', 'animal:cat', 'animal:dog',
           'animal:cow', 'animal:elephant', 'animal:giraffe', 'animal:horse',
           'animal:snake', 'animal:sheep', 'animal:zebra', 'person',
           'vehicle:airplane', 'vehicle:bicycle', 'vehicle:boat', 'vehicle:bus',
           'vehicle:car', 'vehicle:golf cart', 'vehicle:motorcycle', 'vehicle:train',
           'vehicle:truck', 'winebottle')

NETS = {
    'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel'),
    'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel'),
    'sonynet': ('SONYNET', 'SONYNET_faster_rcnn_final.caffemodel')
}


def vis_detections (im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo (net, imagePathName):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(imagePathName)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args ():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use [0]', default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--net', dest='net', help='Network to use [vgg16]', choices=NETS.keys(), default='sonynet')
    parser.add_argument('--score', dest='testImageDir', help='score on given dir of test images', default='', type=str)
    parser.add_argument('--demo', dest='testImageDir', help='score on given dir of test images', default='', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #
    # Caffe Setup
    #
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffeModel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
                              NETS[args.net][1])

    if not os.path.isfile(caffeModel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffeModel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
        cfg.GPU_ID = args.gpu

    net = caffe.Net(prototxt, caffeModel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffeModel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    #
    # Classify the images
    #
    imageDir = args.testImageDir

    # /im_names = ['000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
    for base, dirs, files in os.walk(imageDir):
        for imageName in files:
            if imageName.endswith(".jpg"):
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                print 'Demo for data/demo/{}'.format(imageName)
                if len(dirs) > 0:
                    imgPathName = os.path.join(base, dirs, imageName)
                else:
                    imgPathName = os.path.join(base, imageName)
                demo(net, imgPathName)
                plt.show()  # display windows and wait until all are closed
