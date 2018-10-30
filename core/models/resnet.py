# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/30/2018 2:09 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras_resnet.models
from keras import Model
from keras.layers import Conv2D, Conv2DTranspose, Add, Activation, Reshape, Concatenate

from core.layers import AnchorTarget


def compute_pyramid_features(res3, res4, res5, feature_size=256):
    # compute deconvolution kernel size based on scale
    scale = 2
    kernel_size = (2 * scale - scale % 2)

    # upsample res5 to get P5 from the FPN paper
    P5 = Conv2D(feature_size, (1, 1), strides=1, padding='same', name='P5')(res5)
    P5_upsampled = Conv2DTranspose(feature_size, kernel_size=kernel_size, strides=scale, padding='same',name='P5_upsampled')(P5)

    P4 = Conv2D(feature_size, (3, 3), strides=1, padding='same', name='res4_reduced')(res4)
    P4 = Add(name='P4')([P5_upsampled, P4])

    # upsample P4 and add elementwise to C3
    P4_upsampled = Conv2DTranspose(feature_size, kernel_size=kernel_size, strides=scale, padding='same', name='P4_upsampled')(P4)
    P3 = Conv2D(feature_size, (3, 3), strides=1, padding='same', name='res3_reduced')(res3)
    P3 = Add(name='P3')([P4_upsampled, P3])

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P6')(res5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = Activation('relu', name='res6_relu')(P6)
    P7 = Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7

def classification_subnet(num_classes=21, num_anchors=9, feature_size=256):
    layers = []
    for i in range(4):
        layers.append(Conv2D(feature_size,(3,3),strides=1,padding='same',name='cls_{}'.format(i)))
    layers.append(Conv2D(num_classes*num_anchors,(3,3),strides=1,padding='same',name='pyramid_classification'))

    return layers

def regression_subnet(num_anchors=9, feature_size=256):
    layers = []
    for i in range(4):
        layers.append(Conv2D(feature_size, (3, 3), strides=1, padding='same', name='reg_{}'.format(i)))
    layers.append(Conv2D(4 * num_anchors, (3, 3), strides=1, padding='same', name='pyramid_regression'))

    return layers

def RetinaNet(inputs, backbone, num_classes=21, feature_size=256, *args, **kwargs):
    image, im_info, gt_boxes = inputs
    num_anchors = 9
    _, res3, res4, res5 = backbone.outputs # ignore res2

    # pyramid features
    pyramid_features = compute_pyramid_features(res3, res4, res5)

    # construct classification and regression subnets
    classification_layers = classification_subnet(num_classes=num_classes, num_anchors=num_anchors,feature_size=feature_size)
    regression_layers     = regression_subnet(num_anchors=num_anchors, feature_size=feature_size)

    # for all pyramid levels, run classification and regression branch and compute anchors
    classification = None
    labels = None
    regression = None
    regression_target = None
    for i, p in enumerate(pyramid_features):
        # run the classification subnet
        cls = p
        for l in classification_layers:
            cls = l(cls)

        # compute labels and bbox_reg_targets
        l, r = AnchorTarget(stride=16, name='boxes_{}'.format(i))([cls, im_info, gt_boxes])
        labels = l if labels == None else Concatenate(axis=0)([labels, l])
        regression_target = r if regression_target == None else Concatenate(axis=0)([regression_target, r])

        cls = Reshape((-1, num_classes), name='classification_{}'.format(i))(cls)
        classification = cls if classification == None else Concatenate(axis=1)([classification, cls])

        # run the regression subnet
        reg = p
        for l in regression_layers:
            reg = l(reg)

        reg = Reshape((-1, 4), name='boxes_reshaped_{}'.format(i))(reg)
        regression = reg if regression == None else Concatenate(axis=1)([regression, reg])

    # TODO: Apply loss on classification / regression
    return Model(inputs=inputs, outputs=[classification, regression], *args, **kwargs)

def ResNet50RetinaNet(inputs, *args, **kwargs):
    image, _, _ = inputs
    resnet = keras_resnet.models.ResNet50(image, include_top=False)

    retinanet = RetinaNet(inputs, resnet, *args, **kwargs)
    return retinanet

#
# if __name__ == '__main__':
#     ResNet50RetinaNet([None,None,None])