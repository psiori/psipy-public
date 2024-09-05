# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
#
# Authors:
#   Alexander Hoereth <alexander@psiori.com>, January 2019

import tensorflow as tf


class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(ExpandDims, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ExpandDims, self).build(input_shape)

    def call(self, x):
        return tf.expand_dims(x, self.axis)

    def compute_output_shape(self, input_shape):
        axis = self.axis
        if self.axis < 0:
            axis = len(input_shape) - axis
        return input_shape[:axis] + (1,) + input_shape[axis:]

    def get_config(self):
        return dict(axis=self.axis)
