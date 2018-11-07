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
import core.backend

"""
As described in https://arxiv.org/abs/1708.02002
"""
class FocalLoss(keras.layers.Layer):
	def __init__(self, num_classes=21, alpha=0.25, gamma=2.0, *args, **kwargs):
		self.num_classes = num_classes
		self.alpha = alpha
		self.gamma = gamma

		super(FocalLoss, self).__init__(*args, **kwargs)

	def classification_loss(self, focal_weight, classification, labels):
		cls_loss = focal_weight * keras.backend.sparse_categorical_crossentropy(labels, classification)
		cls_loss = keras.backend.sum(cls_loss)

		# compute the number of anchors assigned to a ground-truth box
		ones           = keras.backend.ones_like(labels)
		zeros          = keras.backend.zeros_like(labels)
		assigned_boxes = core.backend.where(keras.backend.greater(labels, 0), ones, zeros)

		# "The total focal loss of an image is computed as the sum
		# of the focal loss over all ~100k anchors, normalized by the
		# number of anchors assigned to a ground-truth box."
		cls_loss = cls_loss / (keras.backend.maximum(1.0, keras.backend.sum(assigned_boxes)))
		return cls_loss

	def regression_loss(self, focal_weight, labels, regression, regression_target):
		regression_diff     = regression - regression_target
		abs_regression_diff = keras.backend.abs(regression_diff)

		mask = keras.backend.less_equal(abs_regression_diff, 1.0)
		mask = keras.backend.cast(mask, keras.backend.floatx())

		ones           = keras.backend.ones_like(labels)
		zeros          = keras.backend.zeros_like(labels)
		assigned_boxes = core.backend.where(keras.backend.greater(labels, 0), ones, zeros)

		reg_loss = mask * (0.5 * regression_diff * regression_diff) + (1 - mask) * (abs_regression_diff - 0.5)
		reg_loss = keras.backend.sum(reg_loss, axis=1)
		reg_loss = keras.backend.sum(focal_weight * reg_loss)

		# "The total focal loss of an image is computed as the sum
		# of the focal loss over all ~100k anchors, normalized by the
		# number of anchors assigned to a ground-truth box."
		reg_loss = reg_loss / (keras.backend.maximum(1.0, keras.backend.sum(assigned_boxes)))
		return reg_loss

	def call(self, inputs):
		classification, labels, regression, regression_target = inputs

		classification    = keras.backend.reshape(classification, (-1, self.num_classes))
		labels            = keras.backend.reshape(labels, (-1,))
		regression        = keras.backend.reshape(regression, (-1, 4))
		regression_target = keras.backend.reshape(regression_target, (-1, 4))

		indices = core.backend.where(keras.backend.not_equal(labels, -1))

		regression        = core.backend.gather_nd(regression, indices)
		regression_target = core.backend.gather_nd(regression_target, indices)
		classification    = core.backend.gather_nd(classification, indices)
		labels            = core.backend.gather_nd(labels, indices)

		# compute alpha as (1 - alpha) for background and alpha for foreground
		foreground_alpha = keras.backend.ones_like(labels) * self.alpha
		background_alpha = 1.0 - foreground_alpha
		alpha            = core.backend.where(keras.backend.equal(labels, 0), background_alpha, foreground_alpha)

		# select classification scores for labeled anchors
		indices          = keras.backend.expand_dims(core.backend.range(keras.backend.shape(labels)[0]), axis=1)
		labeled_indices  = keras.backend.concatenate([indices, keras.backend.expand_dims(keras.backend.cast(labels, 'int32'), axis=1)], axis=1)
		probabilities    = core.backend.gather_nd(classification, labeled_indices)
		focal_weight     = alpha * (1.0 - probabilities) ** self.gamma

		cls_loss = self.classification_loss(focal_weight, classification, labels)
		self.add_loss(cls_loss)

		reg_loss = self.regression_loss(focal_weight, labels, regression, regression_target)
		self.add_loss(reg_loss)

		return [cls_loss, reg_loss]

	def compute_mask(self, inputs, mask=None):
		return [None, None]