# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
#
# Authors:
#   Alexander Hoereth <alexander@psiori.com>, January 2019

import numpy as np
import pytest
import tensorflow as tf

from psipy.nn.layers import Squeeze


@pytest.mark.usefixtures("tensorflow")
class TestSqueezeLayer:
    @staticmethod
    def test_output_shape():
        layer = Squeeze(axis=3)
        shape = layer.compute_output_shape((2, 1, 1))
        assert len(shape) == 2  # no batch_size dim here
        assert shape == (2, 1)
        inp = tf.keras.Input((2, 1, 1))
        out = layer(inp)
        assert len(out.shape) == 3
        assert out.shape.as_list() == [None, 2, 1]

    @staticmethod
    def test_negative_axis():
        layer = Squeeze(axis=-2)
        shape = layer.compute_output_shape((2, 1, 4))
        assert len(shape) == 2  # no batch_size dim here
        assert shape == (2, 4)
        inp = tf.keras.Input((2, 1, 4))
        out = layer(inp)
        assert len(out.shape) == 3
        assert out.shape.as_list() == [None, 2, 4]

    @staticmethod
    def test_non_squeezable_axis():
        layer = Squeeze(axis=-1)
        with pytest.raises(ValueError):
            layer.compute_output_shape((2, 1, 4))
        inp = tf.keras.Input((2, 1, 4))
        with pytest.raises(ValueError):
            layer(inp)

    @staticmethod
    def test_model():
        inp = tf.keras.Input((2, 1, 4))
        out = Squeeze(axis=2)(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)
        assert len(model.layers[1].output_shape) == 3
        assert len(model.output_shape) == 3
        prediction = model.predict(np.random.rand(32, 2, 1, 4))
        assert prediction.shape == (32, 2, 4)

    @staticmethod
    def test_config():
        layer = Squeeze(axis=3)
        assert "axis" in layer.get_config()
        assert layer.get_config()["axis"] == 3
