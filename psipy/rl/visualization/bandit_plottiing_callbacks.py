# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Multi-armed Bandit Plotting Callbacks

These callbacks are used to plot the process of multi armed bandit training.

.. autosummary::

    ArmProbabilityCallback
    ArmSelectionCallback
    UCBProbabilityCallback

"""
from typing import List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np

from psipy.core.notebook_tools import is_notebook
from psipy.rl.control.bandits.multiarmed_bandits import MultiArmedBandit


class ArmProbabilityCallback:
    """Callback to plot the arm probabilities over the course of training."""

    def __init__(
        self,
        axis: Optional[plt.axis] = None,
        title="Arm Probabilities",
        filepath: Optional[str] = None,
    ):
        if is_notebook():
            raise NotImplementedError
        self.axis = cast(plt.axis, axis)
        if axis is None:
            fig, self.axis = plt.subplots()
        self.title = title
        self.filepath = filepath
        self.in_notebook = is_notebook()

    def plot(self, bandit: MultiArmedBandit) -> None:
        num_arms = len(bandit.arms)
        self.axis.clear()
        self.axis.bar(range(num_arms), bandit.optimizer.arm_probabilities)
        self.axis.set_xlabel("Arm")
        self.axis.set_ylabel("Selection Probability")
        self.axis.set_title(self.title)
        self.axis.set_xticks(range(num_arms))
        self.axis.set_xticklabels([f"Arm {i}" for i in range(num_arms)])
        plt.pause(0.01)


class ArmSelectionCallback:
    """Callback to plot the selected arm and the best arm per step during training."""

    def __init__(
        self,
        axis: Optional[plt.axis] = None,
        title="Arm Selection",
        filepath: Optional[str] = None,
    ):
        if is_notebook():
            raise NotImplementedError
        self.axis = cast(plt.axis, axis)
        if axis is None:
            fig, self.axis = plt.subplots()
        self.title = title
        self.filepath = filepath
        self.in_notebook = is_notebook()

        self._best_arms: List[int] = []

    def plot(self, bandit: MultiArmedBandit) -> None:
        num_arms = len(bandit.arms)
        self.axis.clear()
        self._best_arms.append(bandit.get_best_arm())
        self.axis.scatter(
            range(len(bandit.arm_history)), bandit.arm_history, marker="x"
        )
        self.axis.scatter(
            range(len(bandit.arm_history)),
            self._best_arms,
            color="red",
            label="Current best",
            s=7.5,
        )
        self.axis.set_yticks(range(num_arms))
        self.axis.set_yticklabels([f"Arm {i}" for i in range(num_arms)])
        self.axis.set_xlabel("Episode")
        self.axis.set_ylabel("Pulled arm")
        self.axis.set_title(self.title)
        self.axis.legend()
        plt.pause(0.01)


class UCBProbabilityCallback:
    """Callback plotting the arm probabilities and upper bounds throughout training."""

    def __init__(
        self,
        axis: Optional[plt.axis] = None,
        title="UCB Arm Weighting",
        filepath: Optional[str] = None,
        side_by_side: bool = True,  # TODO Docstring
    ):
        if is_notebook():
            raise NotImplementedError
        self.axis = cast(plt.axis, axis)
        if axis is None:
            fig, self.axis = plt.subplots()
        self.title = title
        self.filepath = filepath
        self.side_by_side = side_by_side
        self.in_notebook = is_notebook()

    def plot(self, bandit: MultiArmedBandit) -> None:
        num_arms = len(bandit.arms)
        self.axis.clear()
        arm_probabilities = bandit.optimizer.arm_probabilities
        upper_bounds = bandit.optimizer.upper_bounds

        if not self.side_by_side:
            self.axis.bar(range(num_arms), arm_probabilities, label="Mean Reward")
            self.axis.bar(
                range(num_arms),
                upper_bounds,
                bottom=arm_probabilities,
                label="Upper Bound",
            )
        else:
            self.axis.bar(
                np.arange(len(arm_probabilities)) - 0.35 / 2,
                arm_probabilities,
                0.35,
                label="Mean Reward",
            )
            self.axis.bar(
                np.arange(len(upper_bounds)) + 0.35 / 2,
                upper_bounds,
                0.35,
                bottom=arm_probabilities,
                label="Upper Bound",
                color="gray",
            )
            self.axis.axhline(
                max(arm_probabilities + upper_bounds),
                color="black",
                alpha=0.5,
                ls="--",
                label="Next Selection",
            )
            self.axis.axhline(max(arm_probabilities), color="green", label="Best arm")

        self.axis.set_xlabel("Arm")
        self.axis.set_ylabel("Mean Reward and Upper Bound")
        self.axis.set_title(self.title)
        self.axis.set_xticks(range(num_arms))
        self.axis.set_xticklabels([f"Arm {i}" for i in range(num_arms)])
        self.axis.legend()
        plt.pause(0.01)
