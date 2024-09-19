# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Noise to add to continuous actions to promote exploration.

.. autosummary::

    Noise
    OrnsteinUhlenbeckProcess
    RandomNormalNoise

"""

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Optional

import numpy as np

from psipy.core import Saveable
from psipy.core.io import MemoryZipFile

__all__ = ["Noise", "OrnsteinUhlenbeckProcess", "OU"]


class Noise(Saveable, metaclass=ABCMeta):
    """Noise Abstract Class

    Noise is added to continuous controllers to explore their state space.

    Args:
        size: number of action channels
        **kwargs: specific noise implementation keywords
    """

    def __init__(self, size: int, **kwargs):
        self.size = size
        super().__init__(size=size, **kwargs)

    @abstractmethod
    def __call__(self, value: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @staticmethod
    def load_impl(zipfile: MemoryZipFile) -> "Noise":
        for NoiseImplementation in [OrnsteinUhlenbeckProcess, RandomNormalNoise]:
            try:
                return NoiseImplementation.load(zipfile)  # type: ignore
            except FileNotFoundError:
                pass
        raise FileNotFoundError("No Noise implementation found in zipfile.")

    def __repr__(self):
        kwargs = ",".join([f"{k}={v}" for k, v in self.get_config().items()])
        return f"{self.__class__.__name__}({kwargs})"


class OrnsteinUhlenbeckProcess(Noise):
    """Ornstein Uhlenbeck Noise

    This noise is a stochastic process and essentially is a damped random walk.
    It is mean-reverting, and is Gaussian, Markov, and temporarily homogeneous.
    It is also a model for the velocity of a massive Brownian particle under the
    influence of friction.

    Further reading:

    - `Wikipedia <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>`_
    - `Helpful blog post <https://eventuallyalmosteverywhere.wordpress.com/2014/\
       10/11/ornstein-uhlenbeck-process/>`_

    Args:
        mu: Mean to tend to, basically a bias.
        sigma: Magnitude of brownian motion.
        theta: Strength of mean reversion; ``theta == 0 := brownian motion``.
        dt: Delta time between updates.
        initial_value: Initial value for the internal state.
    """

    def __init__(
        self,
        size: int,
        sigma: float,
        mu: float = 0.0,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_value: float = 0,
    ):
        super().__init__(
            size=size,
            sigma=sigma,
            mu=mu,
            theta=theta,
            dt=dt,
            initial_value=initial_value,
        )
        self.sigma = sigma
        self.mu = mu
        self.theta = theta
        self.dt = dt
        self.initial_value = initial_value
        self.state = np.ones(self.size) * self.initial_value
        self.reset()

    def __call__(self, value: Optional[np.ndarray] = None) -> np.ndarray:
        self.state = (
            self.state
            + self.theta * (self.mu - self.state) * self.dt
            + np.sqrt(self.dt) * np.random.normal(0, self.sigma, self.size)
        )
        if value is not None:
            return value + self.state
        return self.state

    def reset(self):
        self.state = np.ones(self.size) * self.initial_value

    def set_sigma(self, sigma: float):
        self.sigma = sigma


OrnsteinUhlenbeckActionNoise = OrnsteinUhlenbeckProcess
OU = OrnsteinUhlenbeckProcess


class RandomNormalNoise(Noise):
    """Random Normal Noise"""

    def __init__(self, size, std: float = 1.0):
        super().__init__(size=size, std=std)
        # TODO: Does this need to be (10,1) or (10,)?
        self._noise = partial(np.random.normal, size=size, scale=std)

    def __call__(self, value: Optional[np.ndarray] = None) -> np.ndarray:
        if value is not None:
            return value + self._noise()
        return self._noise()

    def reset(self):
        pass


if __name__ == "__main__":
    # for i in range(5):
    #     n = RandomNormalNoise(size=10)
    #     print(n())
    ou = OrnsteinUhlenbeckActionNoise(
        size=1, sigma=20, mu=400.0, theta=0.15, dt=1e-2, initial_value=400
    )
    data = []
    for _ in range(10000):
        data.append(ou())
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(20, 9))
    ax.plot(data)
    plt.show()
