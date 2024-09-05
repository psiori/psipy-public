from typing import Any

import numpy as np
import pytest

from psipy.rl.controllers.bandits.bandit_optimizers import (
    EpsilonGreedyBanditOptimizer,
    EpsilonGreedySlidingWindowUCBBanditOptimizer,
    SlidingWindowUCBBanditOptimizer,
    SoftmaxBanditOptimizer,
    ThompsonSamplingBanditOptimizer,
    UCB1BanditOptimizer,
)
from psipy.rl.controllers.bandits.multiarmed_bandits import MultiArmedBandit


def test_choose_best_arm():
    optimizer = EpsilonGreedyBanditOptimizer(2, 0.5)
    optimizer.rewards[1] = 100
    optimizer.arm_counts[1] = 100
    # Assert best arm when a nan exists in the probabilities
    assert optimizer.get_best_arm() == 1
    optimizer.arm_counts[0] = 1
    # Assert best arm when all probabilities exist
    assert optimizer.get_best_arm() == 1


def test_nan_always_argmax():
    # Implies test of max as well
    assert np.argmax(np.array([np.nan, 1])) == 0
    assert np.argmax(np.array([1, np.nan])) == 1
    assert np.argmax(np.array([-1, 0, np.nan])) == 2


PROBLEM = (0.1, 0.9)  # simple two arm problem


class ProblemBandit(MultiArmedBandit):
    """Simple slot machine bandit."""

    def _evaluate_arm(self, arm: Any) -> float:
        print(self.arms, arm)
        if np.random.random() < self.arms[arm]:
            return 1
        else:
            return 0


@pytest.mark.parametrize(
    "optimizer",
    [
        EpsilonGreedyBanditOptimizer(2, 0.5),
        SoftmaxBanditOptimizer(2, 0.1),
        ThompsonSamplingBanditOptimizer(2, 1, 1),
        UCB1BanditOptimizer(2),
        SlidingWindowUCBBanditOptimizer(2, 100),
        EpsilonGreedySlidingWindowUCBBanditOptimizer(2, 100, 0.5),
    ],
)
def test_convergence(optimizer):
    """Train and return the best arm"""
    bandit = ProblemBandit(PROBLEM, optimizer)
    for _ in range(1000):
        bandit.choose_arm()
    assert bandit.choose_best_arm() == 1


def test_running_average_and_instant_average_equal():
    """Test the online running average reward calculation.

    Online there is two different methods, plus our method. This test makes sure
    they are all equivalent.

        * reward_inverse := self.estimate += 1 / (self.count) * (r - self.estimate)
        * reward_add := self.mean_reward + (reward - self.mean_reward) / self.n
        * reward_ours := total_reward / total_counts

    """

    class RunningAverage:
        def __init__(self):
            self.counts = 0
            self.reward_inverse = 0
            self.reward_add = 0
            self.reward_ours = 0

        def update_r(self, r):
            self.counts += 1
            self.reward_ours += r
            self.reward_inverse += 1.0 / (self.counts) * (r - self.reward_inverse)
            self.reward_add += (r - self.reward_add) / self.counts

    ra = RunningAverage()
    values = [-10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    true_mean = np.mean(values)
    for r in values:
        ra.update_r(r)
    assert true_mean == ra.reward_inverse == ra.reward_add == ra.reward_ours / ra.counts
