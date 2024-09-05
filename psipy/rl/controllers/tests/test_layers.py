# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import io

import h5py
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from psipy.rl.control.layers import ArgMaxLayer, ArgMinLayer, ClipLayer
from psipy.rl.control.layers import ExtractIndexLayer, MinLayer


@pytest.mark.usefixtures("tensorflow")
class TestArgMinLayer:
    @staticmethod
    def test_default():
        data = np.array([[0, 1], [1, 0], [0, 0]])
        inp = tfkl.Input((2,))
        out = ArgMinLayer()
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [[0], [1], [0]]
        assert out.output_shape == (None, 1)

    @staticmethod
    def test_keepdims_false():
        data = np.array([[0, 1], [1, 0], [0, 0]])
        inp = tfkl.Input((2,))
        out = ArgMinLayer(keepdims=False)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [0, 1, 0]
        assert out.output_shape == (None,)

    @staticmethod
    def test_axis():
        data = np.array([[[2, 3], [1, 0]], [[0, 3], [5, 4]], [[0, 0], [1, 1]]])
        inp = tfkl.Input((2, 2))
        # "stacked tuples"
        out = ArgMinLayer(keepdims=False, axis=1)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [[1, 1], [0, 0], [0, 0]]
        assert out.output_shape == (None, 2)
        # "within tuples"
        out = ArgMinLayer(keepdims=False, axis=2)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [[0, 1], [0, 1], [0, 0]]
        assert out.output_shape == (None, 2)

    @staticmethod
    def test_saveload():
        inp = tfkl.Input((2, 2))
        out = ArgMinLayer(keepdims=False, axis=0)
        model = tfk.Model(inp, out(inp))
        # Save model in memory
        data = io.BytesIO()
        with h5py.File(data, "w") as h5f:
            tf.keras.models.save_model(model, h5f, save_format="h5")
        # Load model from memory
        with h5py.File(data, "r") as h5f:
            loaded = tf.keras.models.load_model(
                h5f,
                custom_objects={"ArgMinLayer": ArgMinLayer},
            )
        assert loaded.layers[-1].get_config() == out.get_config()


@pytest.mark.usefixtures("tensorflow")
class TestArgMaxLayer:
    @staticmethod
    def test_default():
        data = np.array([[0, 1], [1, 0], [0, 0]])
        inp = tfkl.Input((2,))
        out = ArgMaxLayer()
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().dtype == np.int64
        assert model(data).numpy().tolist() == [[1], [0], [0]]
        assert out.output_shape == (None, 1)

    @staticmethod
    def test_keepdims_false():
        data = np.array([[0, 1], [1, 0], [0, 0]])
        inp = tfkl.Input((2,))
        out = ArgMaxLayer(keepdims=False)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [1, 0, 0]
        assert out.output_shape == (None,)

    @staticmethod
    def test_dtype():
        data = np.array([[0, 1], [1, 0], [0, 0]])
        inp = tfkl.Input((2,))
        out = ArgMaxLayer(keepdims=False, dtype=tf.float32)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().dtype == np.float32
        out = ArgMaxLayer(keepdims=False, dtype=tf.uint8)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().dtype == np.uint8

    @staticmethod
    def test_axis():
        data = np.array([[[2, 3], [1, 0]], [[0, 3], [5, 4]], [[0, 0], [1, 1]]])
        inp = tfkl.Input((2, 2))
        # "stacked tuples"
        out = ArgMaxLayer(keepdims=False, axis=1)
        model = tfk.Model(inp, out(inp))
        print(model(data).numpy().tolist())
        assert model(data).numpy().tolist() == [[0, 0], [1, 1], [1, 1]]
        assert out.output_shape == (None, 2)
        # "within tuples"
        out = ArgMaxLayer(keepdims=False, axis=2)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [[1, 0], [1, 0], [0, 0]]
        assert out.output_shape == (None, 2)

    @staticmethod
    def test_saveload():
        inp = tfkl.Input((2, 2))
        out = ArgMaxLayer(keepdims=False, axis=0)
        model = tfk.Model(inp, out(inp))
        # Save model in memory
        data = io.BytesIO()
        with h5py.File(data, "w") as h5f:
            tf.keras.models.save_model(model, h5f, save_format="h5")
        # Load model from memory
        with h5py.File(data, "r") as h5f:
            loaded = tf.keras.models.load_model(
                h5f,
                custom_objects={"ArgMaxLayer": ArgMaxLayer},
            )
        assert loaded.layers[-1].get_config() == out.get_config()


@pytest.mark.usefixtures("tensorflow")
class TestExtractIndexLayer:
    @staticmethod
    def test_default():
        inp = tfkl.Input((2,))
        idx = tfkl.Input((1,), dtype="int32")
        out = ExtractIndexLayer()
        model = tfk.Model([inp, idx], out([inp, idx]))
        input_shapes = ((None, 2), (None, 1))
        assert out.compute_output_shape(input_shapes) == (None, 1)
        input_shapes = (tf.TensorShape((None, 2)), tf.TensorShape((None, 1)))
        assert out.compute_output_shape(input_shapes) == (None, 1)
        assert out.output_shape == (None, 1)
        inputs = np.array([[2, 34]]), np.array([[0]])
        assert model(inputs).numpy().tolist() == [[2]]
        inputs = np.array([[2, 34]]), np.array([[1]])
        assert model(inputs).numpy().tolist() == [[34]]
        inputs = np.array([[2, 34], [3, 2]]), np.array([[0], [1]])
        assert model(inputs).numpy().tolist() == [[2], [2]]
        inputs = np.array([[2, 34], [5, 3]]), np.array([[1], [1]])
        assert model(inputs).numpy().tolist() == [[34], [3]]
        # Works with single dimension indices?
        inputs = np.array([[2, 34]]), np.array([0])
        assert model(inputs).numpy().tolist() == [[2]]
        inputs = np.array([[2, 34]]), np.array([1])
        assert model(inputs).numpy().tolist() == [[34]]
        inputs = np.array([[2, 34], [3, 2]]), np.array([0, 1])
        assert model(inputs).numpy().tolist() == [[2], [2]]
        inputs = np.array([[2, 34], [5, 3]]), np.array([1, 1])
        assert model(inputs).numpy().tolist() == [[34], [3]]


@pytest.mark.usefixtures("tensorflow")
def test_cliplayer():
    data = np.array([[0, 1], [1, 0], [23, 12]])
    inp = tfkl.Input((2,), dtype="float32")
    out = ClipLayer(0.5, 1)
    model = tfk.Model(inp, out(inp))
    assert model(data).numpy().tolist() == [[0.5, 1], [1, 0.5], [1.0, 1]]
    assert out.output_shape == (None, 2)

    inp = tfkl.Input((2,), dtype="int32")
    out = ClipLayer(0, 1)
    model = tfk.Model(inp, out(inp))
    assert model(data).numpy().tolist() == [[0, 1], [1, 0], [1.0, 1]]
    assert out.output_shape == (None, 2)

    # Save model in memory
    data = io.BytesIO()
    with h5py.File(data, "w") as h5f:
        tf.keras.models.save_model(model, h5f, save_format="h5")
    # Load model from memory
    with h5py.File(data, "r") as h5f:
        loaded = tf.keras.models.load_model(
            h5f,
            custom_objects={"ClipLayer": ClipLayer},
        )
    assert loaded.layers[-1].get_config() == out.get_config()


@pytest.mark.usefixtures("tensorflow")
class TestMinLayer:
    @staticmethod
    def test_default():
        data = np.array([[0, 1], [1, 0], [23, 12]])
        inp = tfkl.Input((2,), dtype="int32")
        out = MinLayer()
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [[0], [0], [12]]
        assert out.output_shape == (None, 1)

    @staticmethod
    def test_keepdims():
        data = np.array([[0, 1], [1, 0], [23, 12]])
        inp = tfkl.Input((2,), dtype="int32")
        out = MinLayer(keepdims=False)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [0, 0, 12]
        assert out.output_shape == (None,)

    @staticmethod
    def test_axis():
        data = np.array([[[2, 3], [1, 0]], [[0, 3], [5, 4]], [[0, 0], [1, 1]]])
        inp = tfkl.Input((2, 2), dtype="int32")
        # "stacked tuples"
        out = MinLayer(keepdims=False, axis=1)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [[1, 0], [0, 3], [0, 0]]
        assert out.output_shape == (None, 2)
        # "within tuples"
        out = MinLayer(keepdims=False, axis=2)
        model = tfk.Model(inp, out(inp))
        assert model(data).numpy().tolist() == [[2, 0], [0, 4], [0, 1]]
        assert out.output_shape == (None, 2)

    @staticmethod
    def test_saveload():
        inp = tfkl.Input((2, 2))
        out = MinLayer(keepdims=False, axis=0)
        model = tfk.Model(inp, out(inp))
        # Save model in memory
        data = io.BytesIO()
        with h5py.File(data, "w") as h5f:
            tf.keras.models.save_model(model, h5f, save_format="h5")
        # Load model from memory
        with h5py.File(data, "r") as h5f:
            loaded = tf.keras.models.load_model(
                h5f,
                custom_objects={"MinLayer": MinLayer},
            )
        assert loaded.layers[-1].get_config() == out.get_config()
