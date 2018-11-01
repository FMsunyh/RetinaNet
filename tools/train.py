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

import keras
from keras.layers import  Input
from keras.utils import get_file
from keras_applications.resnet50 import WEIGHTS_PATH_NO_TOP

from core.models.resnet import ResNet50RetinaNet
from core.preprocessing import PascalVocIterator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def create_model():
	image = Input((512, 512, 3))
	im_info = Input((3,))
	gt_boxes = Input((None, 5))
	return ResNet50RetinaNet([image, im_info, gt_boxes])

if __name__=='__main__':
	# create the model
	model = create_model()

	# load pretrained weights
	weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af')
	model.load_weights(weights_path, by_name=True, skip_mismatch=True)

	# compile model (note: set loss to None since loss is added inside layer)
	model.compile(loss=None, optimizer='adam')

	# print model summary
	print(model.summary())

	# create an image data generator object
	image_data_generator = keras.preprocessing.image.ImageDataGenerator(
		rescale=1/255
	)

	# create a generator for training data
	train_generator = PascalVocIterator(
		'/home/syh/train_data/VOCdevkit/VOC2007',
		'trainval',
		image_data_generator
	)

	# create a generator for testing data
	test_generator = PascalVocIterator(
		'/home/syh/train_data/VOCdevkit/VOC2007',
		'test',
		image_data_generator
	)

	# start training
	batch_size = 1
	model.fit_generator(
		generator=train_generator,
		steps_per_epoch=len(train_generator.image_names) // batch_size,
		epochs=100,
		verbose=1,
		validation_data=test_generator,
		validation_steps=100 #len(test_generator.image_names) // batch_size,
		# callbacks=[
		# 	keras.callbacks.ModelCheckpoint('snapshots/resnet50_pascal_voc_2007.h5', monitor='val_loss', verbose=1, save_best_only=True),
		# 	keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
		# ],
	)
