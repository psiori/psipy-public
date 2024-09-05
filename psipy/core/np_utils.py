# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Utility functions for working with numpy.

.. autosummary::

    cache
    rolling_window
    count_unique_values_per_row

"""

from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import xxhash

__all__ = [
    "cache",
    "rolling_window",
    "count_unique_values_per_row",
]


def _digest(value: Any) -> Union[Tuple[Any, ...], bytes]:
    """Produce a hash digest for a given value. Used in :meth:`cache`.

    Currently supports :class:`numpy.ndarray` and bytes. To be extended to
    other datatypes as needed.

    Args:
        value: Value to hash.
    """
    if isinstance(value, np.ndarray):
        return _digest(value.tobytes())
    if isinstance(value, dict):  # Recursive dict traversal.
        return tuple((key, _digest(val)) for key, val in value.items())
    if isinstance(value, (list, tuple)):  # Recursive list & tuple traversal.
        return tuple(_digest(val) for val in value)
    if isinstance(value, (bytes, str)):
        return xxhash.xxh64(value).hexdigest()
    raise ValueError(f"Didn't know how to hash value of type {type(value)}: {value}.")


def cache(cache_size: int = -1):
    """:meth:`~functools.lru_cache`-like decorator for methods receiving numpy data.

    Example::

        >>> calls = 0
        >>> @cache(-1)
        ... def factorial(n: np.ndarray):
        ...     '''Calculate the factorial for each column in a 2D array.'''
        ...     global calls
        ...     calls += 1
        ...     if np.all(n == 1): return 1
        ...     return n * factorial(np.clip(n - 1, 1, n.max()))
        >>> factorial(np.arange(5) + 1).tolist()  # from scratch, 5 iterations
        [1, 2, 6, 24, 120]
        >>> calls
        5
        >>> factorial(np.arange(5) + 2).tolist()  # only one more iteration
        [2, 6, 24, 120, 720]
        >>> calls
        6

    Args:
        cache_size: Number of the most recent unique method calls to be cached.
    """

    def np_cache_decorator(function: Callable):
        """Decorator for adding the actual caching to the passed function.

        Args:
            function: A callable for which to cache its calls.
        """
        cache: Dict[Tuple[str, ...], Any] = dict()
        keys: List[Tuple[str, ...]] = list()

        @wraps(function)
        def cached_function(*args):
            """The wrapped function, calling it only when the cache misses.

            Args:
                *args: Arguments to pass through to the actual function.
            """
            key = _digest(args)
            if key not in cache:
                # Do some FIFO housekeeping according to 'cache_size'.
                if cache_size == len(cache):
                    del cache[keys.pop(0)]
                # Cache the new value.
                cache[key] = function(*args)
                keys.append(key)
            keys.remove(key)
            keys.append(key)
            return cache[key]

        return cached_function

    return np_cache_decorator


def count_unique_values_per_row(arr: np.ndarray) -> np.ndarray:
    """Gets the count of unique values per row in the given 2D numpy array."""
    arr = np.asarray(arr)
    if not len(arr.shape) == 2:
        raise ValueError("Can only handle 2D numpy arrays.")
    arr = np.sort(arr, axis=1)
    return (arr[:, 1:] != arr[:, :-1]).sum(axis=1) + 1


def rolling_window(arr: np.ndarray, window: int) -> np.ndarray:
    """Creates a rolling window view of an array.

    Example::

        >>> rolling_window(np.arange(5), 3)
        array([[0, 1, 2],
               [1, 2, 3],
               [2, 3, 4]])

    Args:
        arr: The array to create the rolled view of.
        window: The size of the rolling window.
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
