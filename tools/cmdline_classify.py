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
import argparse

import caffe
import cv2
import matplotlib.pyplot as canvas
import numpy as np
import os

from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect
from utils.file import readLines
from utils.debug import debug
from utils.timer import Timer

NMS_THRESHOLD = 0.3

# FIXME: specify model (or labels file in cfg)
NETS = {
    'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel'),
    'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel'),
    'sonynet': ('sonynet', 'sonynet_faster_rcnn_final_VGG16.caffemodel')
}
CLASSES = ['__background__'] + readLines('/opt/dev/proj/sony/research/py-faster-rcnn/lib/datasets/pascal_voc_labels.txt')


def vis_detections (im, class_name, imageName, dets, scoreThreshold=0.6):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= scoreThreshold)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = canvas.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            canvas.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} >= {:.1f} {}').format(class_name, scoreThreshold, imageName), fontsize=16)
    canvas.axis('off')
    canvas.tight_layout()
    canvas.draw()


def demo (net, imagePathName, scoreThreshold):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(imagePathName)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()

    scores, boxes = im_detect(net, im)

    timer.toc()
    debug('Object detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    path, imageFilename = os.path.split(imagePathName)
    catDir = os.path.split(path)[-1]
    imageName = catDir + '/' + imageFilename
    for i, cls in enumerate(CLASSES[1:]):
        i += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * i:4 * (i + 1)]
        cls_scores = scores[:, i]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESHOLD)
        dets = dets[keep, :]
        vis_detections(im, cls, imageName, dets, scoreThreshold)


def filterScoredRegions (scores, boxes, scoreThreshold):
    """Reduce the given boxes to the relevant ones, and combine with label"""

    for i, cls in enumerate(CLASSES[1:]):
        i += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * i:4 * (i + 1)]
        cls_scores = scores[:, i]       # take only scores for this class
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESHOLD)
        dets = dets[keep, :]
        dets = np.array(filter(lambda scoredBoxes: scoredBoxes[-1] > scoreThreshold, dets))
        if len(dets) > 0:
            filteredScores = dets[:, -1]
            filteredBoxes = dets[:, 0:4]
            yield (cls, zip(map(lambda p: int(100 * p), filteredScores), map(lambda xs: map(lambda x: int(x), xs), filteredBoxes)))


def classify (net, imagePathName, scoreThreshold):
    img = cv2.imread(imagePathName)
    if img is None:
        debug('Image does not exist: ' + imagePathName)
        return

    allScores, allBoxes = im_detect(net, img)  # returns scores + boxes for ALL candidate regions, R (where R > 100)
    # scores: R x  K    array of object class scores (K includes background category 0)
    # boxes:  R x (4*K) array of predicted bounding boxes

    tagScoreBoxes = list()
    for region in filterScoredRegions(allScores, allBoxes, scoreThreshold):
        tagScoreBoxes.append((region[0], region[1][0]))

    def formatResult (tagScoreBox):
        label = tagScoreBox[0]                          # category name
        box = ','.join(map(str, tagScoreBox[1][1]))     # comma-separated coords (left, top, right, bottom)
        score = tagScoreBox[1][0]                       # integral percentage (0-100)
        return '/'.join(map(str, [label, score, box]))  # slash-separated fields

    cmdOutput = ";".join(map(formatResult, tagScoreBoxes))   # semicolon-separated entries, highest scores first

    print(cmdOutput)  # send to STDOUT


def parse_args ():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SonyNet on Faster R-CNN')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use [0]', default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--net', dest='net', help='Network to use [vgg16]', choices=NETS.keys(), default='sonynet')
    parser.add_argument('--demo', dest='testImageDir', help='score on given dir of test images', default='', type=str)
    parser.add_argument('--classify', dest='classifyArg', help='classify the given image', type=str)
    parser.add_argument('--labelfile', action='store_true', help='return the filename containing the classifier labels (one per line)', default=False)
    parser.add_argument('--threshold', dest='threshold', help='the lowest score to accept (0-1.0)', default=0.60, type=float)

    #args = parser.parse_args()
    args, restOfArgs = parser.parse_known_args()
    if args.classifyArg is None and len(restOfArgs) > 0:
        args.classifyArg = restOfArgs[0]

    return args


def main (argv=None):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #
    # Non-Caffe options
    #
    if args.labelfile > 0:  # return the labels file
        print "/opt/dev/proj/sony/research/py-faster-rcnn/lib/datasets/pascal_voc_labels.txt"
        return 0

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
    debug('\n\nLoaded network {:s}'.format(caffeModel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    #
    # Classify
    #
    classifyArg = args.classifyArg
    imageDir = args.testImageDir
    scoreThreshold = args.threshold

    if classifyArg is not None and len(classifyArg) > 0:
        if (classifyArg.endswith('.txt')):
            with open(classifyArg) as imageNames:
                for imageFile in imageNames:
                    # debug('Classifying ' + imageFile)
                    classify(net, imageFile.strip(), scoreThreshold)     # remove trailing \n
        else:
            classify(net, classifyArg, scoreThreshold)       # assume single arg is an image

    elif len(imageDir) > 0:
        for base, dirs, files in os.walk(imageDir):
            for imageName in files:
                if imageName.endswith(".jpg"):
                    imgPathName = os.path.join(base, imageName)
                    demo(net, imgPathName, scoreThreshold)
                    canvas.show()  # display windows and wait until all are closed

    return 0

import sys
if __name__ == '__main__':
    sys.exit(main())

