# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Multi-armed Bandits
============================

.. autosummary::

    MultiArmedBandit

"""
import logging
from abc import abstractmethod
from typing import Any, List, Optional, Tuple

from psipy.rl.control.bandits.bandit_optimizers import BanditOptimizer

LOG = logging.getLogger(__name__)


class MultiArmedBandit:  # TODO: Could also do a contextual bandit...
    """Base multi-armed bandit class.

    Multi-armed bandits learn the most rewarding choice from a set of choices (arms)
    in which the probability of receiving a reward is unknown. Interacting with the
    system increases confidence in an estimation about the distribution across
    the arms.

    There are various optimizers to solve the exploration/exploitation problem posed
    by the multi-armed bandit setting. See :class:`BanditOptimizer` for more
    information.

    The canonical example of a multi-armed bandit problem is playing multiple slot
    machines (the arms) in a casino, each with a varying payout probability. The goal
    is the explore just enough to figure out which machine pays out the most amount
    the most amount of time, and then exploit it.

    Once the bandit has learned the optimal decision, it can always act in an optimal
    manner by choosing that arm via the :meth:`choose_best_arm` method.

    Args:
        arms: a tuple of arms. The arms can be anything, and the selection of the arm
              and what the arm does must be implemented in the :meth:`._evaluate_arm`
              method.
        optimizer: any :class:`BanditOptimizer` chosen to solve the problem
        callbacks: Optional list of bandit callbacks to plot stats during training
    """

    def __init__(
        self,
        arms: Tuple[Any, ...],
        optimizer: BanditOptimizer,
        callbacks: Optional[List] = None,
    ):
        self.arms = arms
        self.optimizer = optimizer
        self.callbacks = callbacks

        self._arm_history: List[int] = []

    def __str__(self):
        return self.__class__.__name__

    def choose_arm(self) -> float:
        """Select an arm, evaluate it, and then update the optimizer's parameters.

        The reward for the arm is returned.
        """
        arm = self.optimizer.select_arm()
        reward = self._evaluate_arm(arm)
        self.optimizer.optimize(arm, reward)
        self._arm_history.append(arm)
        LOG.info(f"Arm probabilities: {self.optimizer.arm_probabilities}")
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.plot(self)
        return reward

    @abstractmethod
    def _evaluate_arm(self, arm: int) -> float:
        """Evaluate the selected arm.

        This function can perform anything, as long as some value can be returned
        that rates the performance of said action. Below are two examples:

            1. The arms are a tuple of probabilities of success. The arm provided
               is sampled and a 1 or 0 is returned if successful. This is akin to
               a binary payout slot machine. See :class:`SlotsMultiArmedBandit`
               in `psipy.examples.rl.bandits` for an implementation.
            2. The arms are `psipy.rl` :class:`Loop`s, each controlled by a different
               controller. The bandit runs the specific loop corresponding to the arm
               chosen and the reward accumulated over the course of the episode is used
               to update the probabilities.
        """
        raise NotImplementedError

    def choose_best_arm(self) -> float:
        """Choose the current best arm, always."""
        return self._evaluate_arm(self.optimizer.get_best_arm())

    def get_best_arm(self) -> int:
        """Get the index of the best arm from the optimizer."""
        return self.optimizer.get_best_arm()

    @property
    def arm_history(self) -> List[int]:
        """A history of which arms were chosen."""
        return self._arm_history
