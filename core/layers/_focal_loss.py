# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/30/2018 2:27 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras

class FocalLoss(keras.layers.Layer):
	def __init__(self, alpha=0.25, gamma=2.0, *args, **kwargs):
		self.alpha = alpha
		self.gamma = gamma

		super().__init__(*args, **kwargs)

	def call(self, inputs):
		labels, prediction = inputs

		loss = self.alpha * (1.0 - prediction) ** self.gamma * keras.backend.sparse_categorical_crossentropy(labels, prediction)
		self.add_loss(loss, inputs)
		return loss