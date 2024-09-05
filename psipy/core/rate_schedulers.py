# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Rate schedulers that alter a value based on some step count."""

from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np


class RateScheduler:
    """A parameter scheduler that anneals/alters a value based on some counter.

    It is common in machine learning to anneal (increase/decrease) values
    over time in order to help aid convergence. This base class implements an
    interface that alters the parameter as desired, and provides a history
    of the parameter for later plotting/analysis.

    Args:
        initial_value: The initial value of the parameter in question
        min: The minimum value this parameter can take. Can also not have a minimum
        max: The maximum value this parameter can take. Can also not have a maximum
    """

    def __init__(
        self,
        initial_value: Union[int, float],
        *,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ):
        self._init_val = initial_value
        self._current_val = initial_value
        self.min = min
        self.max = max
        self._history: List[Union[int, float]] = [initial_value]

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def _update(self, step: Optional[int]) -> Union[int, float]:
        """The update rule for the parameter, e.g. linear, exponential decay, etc.

        Set `self.current_value` here. When set, it will automatically be clamped
        between [min, max] if provided in the class' init.
        """
        raise NotImplementedError

    def update(self, step: Optional[int] = None) -> Union[int, float]:  # TODO: TEST
        """Update the parameter and append it to the history.

        Args:
            step: A counter with which this parameter can be changed.
                  This could be episode count, iteration, etc.
        """
        val = self._update(step)
        self._history.append(val)
        return val

    def clamp(self, val) -> Union[int, float]:
        """Clamp the value between the min and max value, if provided."""
        if self.min is not None:
            val = max(val, self.min)
        if self.max is not None:
            val = min(val, self.max)
        return val

    def reset(self) -> None:
        """Reset the schedule to the starting state."""
        self.current_value = self._init_val

    @property
    def current_value(self) -> Union[int, float]:
        return self._current_val

    @current_value.setter
    def current_value(self, val: Union[int, float]):
        self._current_val = self.clamp(val)

    @property
    def history(self) -> List[Union[int, float]]:
        return self._history

    def check_curve(self, n_steps: int) -> None:
        """Plots the curve for n many steps to check if it behaves as expected."""
        import matplotlib.pyplot as plt

        vals = [self.update(i) for i in range(n_steps)]
        # Reset the scheduler since this function altered its state
        self.reset()

        figure, ax = plt.subplots(figsize=(20, 9))
        ax.plot(vals)
        ax.set_title(f"{self} for {n_steps} steps")
        plt.show()


class LinearRateScheduler(RateScheduler):
    """Linearly increase or decrease the parameter.

    Args:
        rate: How much the value is changed each step. Provide a negative number
              to decrease its value.
    """

    def __init__(
        self, initial_value: Union[int, float], rate: Union[int, float], **kwargs
    ):
        self.rate = rate
        super().__init__(initial_value, **kwargs)

    def _update(self, step: Optional[int] = None):
        self.current_value += self.rate
        return self.current_value


class LaggedExponentialDecayScheduler(RateScheduler):
    """Exponentially decrease the parameter with a slow start.

    The parameter starts to decrease slowly, then decreases rapidly, until
    it starts to decrease slowly again. Graphically it looks like a mirrored sigmoid.

    Args:
        rate: The exponential exponent. The step is divided by this.
    """

    def __init__(
        self, initial_value: Union[int, float], rate: Union[int, float], **kwargs
    ):
        self.rate = rate
        super().__init__(initial_value, **kwargs)

    def _update(self, step: Optional[int] = None):
        if step is None:
            raise ValueError("Step must be provided for this scheduler.")
        self.current_value = self.current_value * np.e ** (-step / self.rate)
        return self.current_value


class ExponentialRateScheduler(RateScheduler):
    """Exponentially increase or decrease the parameter.

    Args:
        rate: The exponential exponent. Negative values indicate decay.
    """

    def __init__(
        self, initial_value: Union[int, float], rate: Union[int, float], **kwargs
    ):
        self.rate = rate
        super().__init__(initial_value, **kwargs)

    def _update(self, step: Optional[int] = None):
        self.current_value *= np.e ** self.rate
        return self.current_value
