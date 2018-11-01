#!/usr/bin/python3

"""
Copyright 2018-2019  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 10/31/2018 4:47 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os

import keras
import keras.preprocessing.image
import keras_resnet.models

import cv2
import numpy as np
import sys
import math


import tensorflow as tf
from keras.utils import get_file

from core.models.resnet import ResNet50RetinaNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def create_model(num_classes=2, *args, **kwargs):
    image = keras.layers.Input((512, 512, 3))
    im_info = keras.layers.Input((3,))
    gt_boxes = keras.layers.Input((None, 5))

    return ResNet50RetinaNet([image, im_info, gt_boxes], num_classes=num_classes)


model = create_model()

# load imagenet weights
weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP,
                        cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af')
model.load_weights(weights_path, by_name=True, skip_mismatch=True)

# compile model
model.compile(loss=None, optimizer=keras.optimizers.sgd(lr=0.01, momentum=0.9, decay=0.0001))
# print(model.summary())

# generate really simple image with one object
image = np.zeros((512, 512, 3), dtype=keras.backend.floatx())
image[100:300, 100:300, :] = 1.0
# plt.imshow(image)
# plt.show()

# create input batch blobs
image_batch = np.expand_dims(image, axis=0)
im_info_batch = np.array([[512, 512, 1.0]])
gt_boxes_batch = np.array([[[100, 100, 300, 300, 1]]])

inputs = [image_batch, im_info_batch, gt_boxes_batch]

def simple_data_generator():
    while True:
        yield inputs, None

print(image_batch.shape, im_info_batch.shape, gt_boxes_batch.shape)


# train for some iterations
model.fit_generator(
    generator=simple_data_generator(),
    steps_per_epoch=100,
    epochs=15,
    verbose=1
)