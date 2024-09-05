# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Custom Keras layers.

.. autosummary::

    ArgMinLayer
    ArgMaxLayer
    ClipLayer
    ExtractIndexLayer
    MinLayer

"""

from typing import Optional, Tuple, Union

import tensorflow as tf

__all__ = ["ArgMinLayer", "ArgMaxLayer", "ClipLayer", "ExtractIndexLayer", "MinLayer"]


class ArgMinLayer(tf.keras.layers.Layer):
    """Gets argmin over specific axis, maybe keeping dimensionality."""

    def __init__(self, *, axis=1, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        argmin = tf.argmin(inputs, axis=self.axis)
        if self.keepdims:
            argmin = tf.expand_dims(argmin, self.axis)
        argmin = tf.cast(argmin, tf.float32)
        return argmin

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        input_shape = tuple(tf.TensorShape(input_shape).as_list())
        shapes = enumerate(input_shape)
        if self.keepdims:
            return tuple([s if i != self.axis else 1 for i, s in shapes])
        return tuple([s for i, s in shapes if i != self.axis])

    def get_config(self):
        return dict(axis=self.axis, keepdims=self.keepdims)


class ArgMaxLayer(tf.keras.layers.Layer):
    """Get argmax over specific axis, maybe keeping dimensionality.

    Args:
        axis: Axis to reduce over.
        keepdims: Whether to keep the reduced axis, will have size 1.
        dtype: Datatype to use for the output.
    """

    def __init__(
        self,
        *,
        axis: int = 1,
        keepdims: bool = True,
        dtype: Union[str, tf.dtypes.DType] = "int64",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
        if isinstance(dtype, tf.dtypes.DType):
            dtype = dtype.name
        self.outtype = dtype

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        argmax = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            argmax = tf.expand_dims(argmax, self.axis)
        argmax = tf.cast(argmax, self.outtype)
        return argmax

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        input_shape = tuple(tf.TensorShape(input_shape).as_list())
        shapes = enumerate(input_shape)
        if self.keepdims:
            return tuple([s if i != self.axis else 1 for i, s in shapes])
        return tuple([s for i, s in shapes if i != self.axis])

    def get_config(self):
        return dict(axis=self.axis, keepdims=self.keepdims, dtype=self.outtype)


class ClipLayer(tf.keras.layers.Layer):
    """Clips incoming tensor to given min and max."""

    def __init__(self, lower, upper, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(inputs, self.lower, self.upper)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self):
        return dict(lower=self.lower, upper=self.upper)


class ExtractIndexLayer(tf.keras.layers.Layer):
    """Extracts value according to index along the first axis, keeping dimensionality.

    NOTE: Currently only supports (BatchSize, Any) shaped data, so no more than
          two dimensionalities. The indices need to be 2D as well, having a
          singleton dimension as the second axis. More flexibility may be added
          when required.
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Initialize layer given a pair of (value, index) input tensors."""
        value, index = inputs
        tf.debugging.assert_shapes([(value, ("N", None)), (index, ("N", 1))])
        num_cols = value.get_shape().as_list()[-1]
        by_index = tf.squeeze(tf.one_hot(index, num_cols), 1)
        return tf.reduce_sum(value * by_index, axis=1, keepdims=True)

    def compute_output_shape(
        self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> Tuple[int, ...]:
        value_shape, index_shape = input_shape
        value_shape = tuple(tf.TensorShape(value_shape).as_list())
        return (value_shape[0], 1)


class GaussianNoiseLayer(tf.keras.layers.Layer):
    """Adds gaussian noise to passed tensor.

    Args:
        stddev: Standard deviation of added noise.
        noise_clip: Value to clip the noise at.
        output_clip: Value to clip the noised output at.
    """

    def __init__(
        self,
        stddev: float,
        noise_clip: Optional[float] = None,
        output_clip: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stddev = stddev
        self.noise_clip = noise_clip
        self.output_clip = output_clip

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(tf.shape(inputs), mean=0, stddev=self.stddev)
        if self.noise_clip is not None:
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        inputs = inputs + noise
        if self.output_clip is not None:
            inputs = tf.clip_by_value(inputs, -self.output_clip, self.output_clip)
        return inputs

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self):
        return dict(
            stddev=self.stddev,
            noise_clip=self.noise_clip,
            output_clip=self.output_clip,
        )


class MinLayer(tf.keras.layers.Layer):
    """Extracts minimum value along given axis, maybe keeping dimensionality."""

    def __init__(self, *, axis=1, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_min(inputs, axis=self.axis, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        input_shape = tuple(tf.TensorShape(input_shape).as_list())
        shapes = enumerate(input_shape)
        if self.keepdims:
            return tuple([s if i != self.axis else 1 for i, s in shapes])
        return tuple([s for i, s in shapes if i != self.axis])

    def get_config(self):
        return dict(axis=self.axis, keepdims=self.keepdims)
