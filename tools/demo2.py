#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
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
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', # always index 0
            'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk',
            'door', 'dresser', 'garbage-bin', 'lamp', 'monitor', 'night-stand',
            'pillow', 'sink', 'sofa', 'table', 'television', 'toilet')

detclass = ('bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk',
            'door', 'dresser', 'garbage-bin', 'lamp', 'monitor', 'night-stand',
            'pillow', 'sink', 'sofa', 'table', 'television', 'toilet')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
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

def demo(net, image_name, classes, ssdir, imgdir, savefile):
    """Detect object classes in an image using pre-computed object proposals."""

    box_file = os.path.join(ssdir, image_name + '.mat')
    obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    im_file = os.path.join(imgdir, image_name + '.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    thresh = 0.3
    fid = open(savefile,'w')

    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            fid.write('{0:d}'.format(cls))
            fid.write(' ')
            fid.write('{0:.3f}'.format(score))
            for j in range(4):
                fid.write(' ')
                fid.write('{0:.3f}'.format(bbox[j]))
            fid.write('\n')

    fid.close()

        # print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls, CONF_THRESH)
        
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = '/nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-norm2/output/training_demo/alexnet_rgb/test.prototxt.images'
    caffemodel = '/nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-distillation/output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000_rgb.caffemodel'

    # Load pre-computed Selected Search object proposals
    ssdir = '/nfs/hn38/users/xiaolonw/cmp_results/dcgan_normal3_train_3dnormal_joint4_2_ss/'
    imgdir = '/nfs/hn38/users/xiaolonw/cmp_results/dcgan_normal3_train_3dnormal_joint4_2/'
    savedir = '/nfs/hn38/users/xiaolonw/cmp_results/dcgan_normal3_train_3dnormal_joint4_2_txt/'

    sample_num = 2000

    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    for i in xrange(sample_num):
        print(i)
        image_name = 'img_{04d}'.format(i)
        save_name = os.path.join(savedir, image_name + '.txt') 
        demo(net, image_name, detclass, ssdir, imgdir, save_name)

    # plt.show()
