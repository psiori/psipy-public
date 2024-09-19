# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
#
# Authors:
#   Alexander Hoereth <alexander@psiori.com>, January 2019

import tensorflow as tf


class Squeeze(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(Squeeze, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Squeeze, self).build(input_shape)

    def call(self, x):
        return tf.keras.backend.squeeze(x, self.axis)

    def compute_output_shape(self, input_shape):
        axis = self.axis
        if axis > 0:
            axis -= 1  # shape does not include batch_size here
        if input_shape[axis] != 1:
            raise ValueError("Can only squeeze axis of size 1.")
        input_shape = input_shape[:axis] + input_shape[axis + 1 :]
        return input_shape

    def get_config(self):
        return dict(axis=self.axis)
