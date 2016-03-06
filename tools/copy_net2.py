#!/usr/bin/env python

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np


caffe.set_mode_cpu()

net_old = caffe.Net('/nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-distillation/output/alexnet_rgb_alexnet_hha/test.prototxt.images+hha', 
	'/nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-distillation/output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000.caffemodel', caffe.TRAIN)

net = caffe.Net('/nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-norm2/scripts/alexnet_rgb/train.prototxt', caffe.TRAIN)
savename = '/nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-distillation/output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000_rgb.caffemodel'


layer_num = 9
# layernames = ('da_conv1', 'da_conv2', 'da_conv3', 'da_conv4', 'da_conv5', 'da_fc6', 'da_fc7', 'da_cls_score', 'da_bbox_pred')
layernames     = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'cls_score', 'bbox_pred')
layernames_old = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'cls_score', 'bbox_pred')




for i in xrange(layer_num):
	layer_name = layernames[i]
	layer_name_old = layernames_old[i]
	weight_dims = np.shape(net.params[layer_name][0].data)
	bias_dims   = np.shape(net.params[layer_name][1].data)

	weight_dims_old = np.shape(net_old.params[layer_name_old][0].data)
	bias_dims_old   = np.shape(net_old.params[layer_name_old][1].data)
	
	for j in xrange(len(weight_dims)): 
		print(weight_dims[j])
		print(weight_dims_old[j])
		assert(weight_dims_old[j] == weight_dims[j])

	for j in xrange(len(bias_dims)): 
		assert(bias_dims_old[j] == bias_dims[j])


	net.params[layer_name][0].data[:] = net_old.params[layer_name_old][0].data[:]
	net.params[layer_name][1].data[:] = net_old.params[layer_name_old][1].data[:]


net.save(str(savename))









