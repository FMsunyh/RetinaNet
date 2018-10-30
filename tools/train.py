# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/30/2018 2:56 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras.layers import  Input
from core.models.resnet import ResNet50RetinaNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def create_model():
	image = Input((512, 512, 3))
	im_info = Input((3,))
	gt_boxes = Input((None, 5))
	return ResNet50RetinaNet([image, im_info, gt_boxes])

if __name__ == '__main__':
    # create the model
    model = create_model()

    print(model.summary())