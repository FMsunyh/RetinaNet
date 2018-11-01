# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/30/2018 2:22 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras.backend
import keras.engine

import core.backend


class AnchorTarget(keras.layers.Layer):
	"""Calculate proposal anchor targets and corresponding labels (label: 1 is positive, 0 is negative, -1 is do not care) for ground truth boxes

	# Arguments
		num_anchors: number of anchors used
		allowed_border: allow boxes to be outside the image by allowed_border pixels
		clobber_positives: if an anchor statisfied by positive and negative conditions given to negative label
		negative_overlap: IoU threshold below which labels should be given negative label
		positive_overlap: IoU threshold above which labels should be given positive label

	# Input shape
		(# of batches, width of feature map, height of feature map, 2 * # of anchors), (# of samples, 4), (width of feature map, height of feature map, channels)

	# Output shape
		(# of samples, ), (# of samples, 4)
	"""

	def __init__(self, features_shape, stride, anchor_size, num_anchors=9, allowed_border=0, clobber_positives=False, negative_overlap=0.4, positive_overlap=0.5, *args, **kwargs):
		self.features_shape = features_shape
		self.stride         = stride
		self.anchor_size    = anchor_size

		self.num_anchors       = num_anchors
		self.allowed_border    = allowed_border
		self.clobber_positives = clobber_positives
		self.negative_overlap  = negative_overlap
		self.positive_overlap  = positive_overlap

		super(AnchorTarget,self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		im_info, gt_boxes = inputs

		# TODO: Fix usage of batch index
		image_shape = im_info[0, :2]

		# TODO: Fix usage of batch index
		gt_boxes = gt_boxes[0]

		total_anchors = self.features_shape[0] * self.features_shape[1] * self.num_anchors

		# 1. Generate proposals from bbox deltas and shifted anchors
		anchors = core.backend.anchor(base_size=self.anchor_size)
		anchors = core.backend.shift(self.features_shape, self.stride, anchors)

		# label: 1 is positive, 0 is negative, -1 is dont care
		ones      = keras.backend.ones((total_anchors,), dtype=keras.backend.floatx())
		zeros     = keras.backend.zeros((total_anchors,), dtype=keras.backend.floatx())
		negatives = ones * -1
		labels    = negatives

		# 2. obtain indices of gt boxes with the greatest overlap, balanced labels
		argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = core.backend.overlapping(anchors, gt_boxes)

		if not self.clobber_positives:
			# assign bg labels first so that positive labels can clobber them
			labels = core.backend.where(keras.backend.less(max_overlaps, self.negative_overlap), zeros, labels)

		# fg label: for each gt, anchor with highest overlap
		# generate a marker to identify where updates should be done
		#marker = tensorflow.ones_like(gt_argmax_overlaps_inds)
		# scatter_nd marker array to labels array shape
		#update_mask = tensorflow.scatter_nd(gt_argmax_overlaps_inds, marker, labels.shape)
		# update labels accordingly
		#labels = core.backend.where(keras.backend.equal(update_mask, 1), ones, labels)

		# fg label: above threshold IOU
		labels = core.backend.where(keras.backend.greater_equal(max_overlaps, self.positive_overlap), ones, labels)

		if self.clobber_positives:
			# assign bg labels last so that negative labels can clobber positives
			labels = core.backend.where(keras.backend.less(max_overlaps, self.negative_overlap), zeros, labels)

		# compute box regression targets
		gt_boxes         = keras.backend.gather(gt_boxes, argmax_overlaps_inds)
		bbox_reg_targets = core.backend.bbox_transform(anchors, gt_boxes)

		# filter out anchors that are outside the image
		labels = core.backend.where(
			(anchors[:, 0] >= -self.allowed_border) &
			(anchors[:, 1] >= -self.allowed_border) &
			(anchors[:, 2] < self.allowed_border + image_shape[1]) & # width
			(anchors[:, 3] < self.allowed_border + image_shape[0]),  # height
			labels,
			negatives
		)

		# select correct label from gt_boxes
		labels = core.backend.where(keras.backend.equal(labels, 1), gt_boxes[:, 4], labels)

		labels           = keras.backend.expand_dims(labels, axis=0)
		bbox_reg_targets = keras.backend.expand_dims(bbox_reg_targets, axis=0)
		anchors          = keras.backend.expand_dims(anchors, axis=0)

		# TODO: implement inside and outside weights
		return [labels, bbox_reg_targets, anchors]

	def compute_output_shape(self, input_shape):
		return [(None, 1), (None, 4), (None, 4)]

	def compute_mask(self, inputs, mask=None):
		return [None, None, None]