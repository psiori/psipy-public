# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""StackNormalizer, a data normalizer specialized for stacked observation data.

When normalizing :mod:`~psipy.rl.io.Batch` data, it is common to fit a
normalizer on the observations, but applying it later on to the
stacked-observations as they would be fed into a neural network. The
:class:`StackNormalizer` within this module is specifically designed for that
use case. The :class:`StackNormalizer` supports different scaling methods:

"""

import logging
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np

from psipy.core.io import Saveable

__all__ = ["StackNormalizer"]


LOG = logging.getLogger(__name__)


class StackNormalizer(Saveable):
    """StackNormalizer, a data normalizer specialized for stacked observation data.

    The StackNormalizer may be applied to data of higher dimensionality then the
    one it has been originally fitted to. This is for example of interest
    when fitting on observation data shaped ``(324, 4)`` and then later on
    transforming stacked-observation data (with lookback) like ``(32, 4, 5)``.
    This also works in higher dimensionality as with RGD images.

    Args:
        method: Method to use for normalizing the data.

            - ``max``: Scale the data according to its absolute maximum value,
              leaving its center untouched. Scaled data will be in ``[-1, 1]``.
            - ``std``: Scale the data according to its standard deviation,
              leaving its center untouched. Scaled data will be in
              ``[-std - mean, std + mean]``.
            - ``minmax``: Scale the data using both its minimum and maximum
              values, using the minimum to shift the center and the shifted
              data's absolute maximum as scale. Scaled data will be in
              ``[0, 1]``.
            - ``meanstd``: Simillar to ``minmax``, but instead using the data's
              mean as center and standard deviation as scale. Scaled data will
              be in ``[-std, std]``.
            - ``meanmax``: Simillar to ``meanstd``, but using the shifted
              data's absolute maximum value as scale. Scaled data will be in
              ``[-1, 1]``.
            - ``identity``: Passes through all values with no transformation.

    """

    #: Available normalization methods.
    METHODS: ClassVar[Tuple[str, ...]] = (
        "max",
        "std",
        "minmax",
        "meanstd",
        "meanmax",
        "identity",
    )

    _method: str
    _center: np.ndarray
    _scale: np.ndarray

    def __init__(self, method: str = "minmax"):
        super().__init__(method=method)
        self._method = method
        self._hasWarned = False


    def fit(self, data: np.ndarray, axis: Optional[int] = None) -> "StackNormalizer":
        """Fits the normalizer on the input data.

        Args:
            data: Data to fit the normalizer to.
            axis: When the normalizer has been fit already, one can fit it again
                specifying an axis to adjust max and scale for a certain subset
                of the data. For example, when enriching the state with actions,
                you would alter the normalizer fit on the last axis to match the
                true action distribution and not what was found in the historic
                data. NOTE: When specifying an axis, the passed data has to be
                single dimensional!
        """
        data = np.asarray(data)

        if self.method not in self.METHODS:
            raise ValueError(f"Unsupported normalization method '{self.method}'.")

        if axis is not None and (self.center is None or self.scale is None):
            raise ValueError("Normalizer must be fit first to alter an axis.")

        shape = data.shape
        if axis is None:
            if len(data.shape) < 2:
                raise ValueError('Data needs to have a "Batch" dimension.')
        else:
            if len(shape) == 2 and shape[1] == 1:  # stacked vector
                data = data.ravel()
            elif len(shape) > 1:
                raise ValueError("When fitting to a single axis, data has to be 1d.")

        if self._method == "max":
            scale = np.abs(data).max(axis=0, keepdims=True)
            center = np.zeros_like(scale)

        elif self._method == "std":
            scale = data.std(axis=0, keepdims=True)
            center = np.zeros_like(scale)

        elif self._method == "minmax":
            center = data.min(axis=0, keepdims=True)
            scale = (data - center).max(axis=0, keepdims=True)

        elif self._method == "meanstd":
            center = data.mean(axis=0, keepdims=True)
            scale = data.std(axis=0, keepdims=True)

        elif self._method == "meanmax":
            center = data.mean(axis=0, keepdims=True)
            scale = np.abs(data - center).max(axis=0, keepdims=True)

        elif self._method == "identity":
            center = np.zeros(1)
            scale = np.ones(1)

        if axis:
            self._center[..., axis] = center
            self._scale[..., axis] = scale
        else:
            self._center = center
            self._scale = scale
        return self

    def set(
        self,
        center: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        method: Optional[str] = None,
    ) -> "StackNormalizer":
        """Forces a center and scale for when the true bounds are known.

        Useful for example when having continuous actions in ``[-10, 10]``,
        where fitting on historic data may not have the proper center and scale.
        """
        if center is not None:
            self._center = np.asarray(center)
        if scale is not None:
            self._scale = np.asarray(scale)
        if method is not None:
            self._method = method
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Applies the previously fit normalization to a given data.

        This method does not manipulate the array if the normalization
        parameters have not been set through calling the fit method beforehand.
        """
        data = np.asarray(data)
        assert len(data.shape) >= 2, 'Data has no "Batch" dimension.'

        try:
            center = self.center
            scale = self.scale
        except AttributeError:
            if not self._hasWarned:
                LOG.warning("Normalizer not fitted, returning values unchanged.")
                self._hasWarned = True
            return data

        # Ensure that there are no zeros in the denominator.
        if isinstance(scale, np.ndarray) and np.any(scale == 0):
            scale = scale.copy()
            scale[scale == 0] = 1e-8

        # The normalizer may be applied to higher dimensional data, requiring
        # added empty dimensions in order to properly vectorized subtract and
        # divide scale and center. While other dimensions like first and n-th
        # may be higher dimensional, the original data-dimensionality (besides
        # "batch"), has to be the same.
        assert (
            scale.shape[1:] == data.shape[1 : len(scale.shape)] == center.shape[1:]
        ), f"{scale.shape}, {data.shape}, {center.shape}"
        while len(data.shape) > len(scale.shape):
            center = center[..., None]
            scale = scale[..., None]

        return (data - center) / scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverses the normalization applied to an data."""
        data = np.asarray(data)
        try:
            center = self.center
            scale = self.scale
        except AttributeError:
            LOG.warning("Normalizer not fitted, returning values unchanged.")
            return data

        # The normalizer may be applied to higher dimensional data, requiring
        # added empty dimensions in order to properly vectorized subtract and
        # divide scale and center. While other dimensions like first and n-th
        # may be higher dimensional, the original data-dimensionality (besides
        # "batch"), has to be the same.
        assert (
            scale.shape[1:] == data.shape[1 : len(scale.shape)] == center.shape[1:]
        ), f"{scale.shape}, {data.shape}, {center.shape}"
        while len(data.shape) > len(scale.shape):
            center = center[..., None]
            scale = scale[..., None]

        return data * scale + center

    @property
    def method(self) -> str:
        return self._method

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    def get_config(self) -> Dict:
        return {
            "method": self.method,
            "scale": self.scale.tolist() if hasattr(self, "_scale") else None,
            "center": self.center.tolist() if hasattr(self, "_center") else None,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StackNormalizer":
        return cls(config["method"]).set(center=config["center"], scale=config["scale"])

    def __str__(self):
        config = self.get_config()
        return (
            f"{self.__class__.__name__}: Method {config['method']}, "
            f"center {config['center']}, scale {config['scale']}"
        )
