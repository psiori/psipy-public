# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
#
# Authors:
#   Alexander Hoereth <alexander@psiori.com>, March 2019

import numpy as np
import pytest
import tensorflow as tf

from psipy.nn.layers import ExpandDims


@pytest.mark.usefixtures("tensorflow")
class TestExpandDimsLayer:
    @staticmethod
    def test_output_shape():
        layer = ExpandDims(axis=1)
        shape = layer.compute_output_shape((2, 4))
        assert len(shape) == 3  # no batch_size dim here
        assert shape == (2, 1, 4)
        inp = tf.keras.Input((2, 1, 1))
        out = layer(inp)
        assert len(out.shape) == 5
        assert out.shape.as_list() == [None, 1, 2, 1, 1]

    @staticmethod
    def test_negative_axis():
        layer = ExpandDims(axis=-1)
        shape = layer.compute_output_shape((2, 1, 4))
        assert len(shape) == 4  # no batch_size dim here
        assert shape == (2, 1, 4, 1)
        inp = tf.keras.Input((2, 1, 4))
        out = layer(inp)
        assert len(out.shape) == 5
        assert out.shape.as_list() == [None, 2, 1, 4, 1]

    @staticmethod
    def test_model():
        inp = tf.keras.Input((2, 4))
        out = ExpandDims(axis=2)(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)
        assert len(model.layers[1].output_shape) == 4
        assert len(model.output_shape) == 4
        prediction = model.predict(np.random.rand(32, 2, 4))
        assert prediction.shape == (32, 2, 1, 4)
