from typing import List, Union

import numpy as np


class WelfordsAlgorithm:
    """Online running mean and standard deviation.

    A running mean and variance is kept by sequentially updating the internal
    mean and M2 parameters for each sample.  Multidimensional input will iterate through
    the first dimension, considering them to be samples.  The output will then be
    a mean and standard deviation with a shape equalling the last n-1 dimensions,
    e.g. an set of MNIST images of shape (100, 28, 28) will have a mean and standard
    deviation of shape (28, 28).

    When using the standard deviation as a divisor, the existence of zeros must be
    dealt with by the user, otherwise there will be a division by zero error.

    Core implementation taken from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self):
        self._mean = 0
        self._count = 0
        self._M2 = 0

    def _update(self, value):
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._M2 += delta * delta2

    def update(self, values: Union[np.ndarray, int, float, List[Union[float, int]]]):
        if isinstance(values, (int, float)):
            values = [values]  # ensure iterable
        for val in values:
            self._update(val)

    @property
    def mean(self):
        """The running mean."""
        return self._mean

    @property
    def std(self):
        """The running standard deviation."""
        if self._count < 2:  # 0 or 1 sample
            # Return 1 since there is no variance from singular/no samples
            # and a division by zero error must be avoided.
            # The following assures std has same shape as the mean;
            # the mean could be a single number, and so
            # it is altered the following way to assure both
            # arrays and numbers work
            return self.mean * 0 + 1
        variance = self._M2 / self._count
        return np.sqrt(variance)

    @property
    def num_samples(self):
        return self._count
