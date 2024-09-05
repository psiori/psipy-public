"""Multi-armed Bandit Optimizers
============================

.. autosummary::

    BanditOptimizer
    EpsilonGreedyBanditOptimizer
    SoftmaxBanditOptimizer
    ThompsonSamplingBanditOptimizer
    UCB1BanditOptimizer
    SlidingWindowUCBBanditOptimizer
    EpsilonGreedySlidingWindowUCBBanditOptimizer

"""

from abc import abstractmethod
from collections import deque
from typing import Deque, List, Set, Union

import numpy as np


class BanditOptimizer:
    """Base class for multi-armed bandit optimizers.

    An optimizer's goal is to alter the chances of selecting different arms
    such that over a given amount of time, the optimizer has figured out which
    arm is the best by seeing the rewards it has accumulated.

    Each optimizer has its own advantages and disadvantages. Some work best for
    stationary (arm probabilities do not change over time) problems, while others
    can handle nonstationary problems. Some assume certain underlying distributions.
    See the docstrings for the various optimizers for more information about their
    specific use cases. Below is a quick summary for reference:

    * Epsilon greedy: all problems
    * UCB1: stationary problems
    * Sliding Window UCB: nonstationary problems
    * Softmax: stationary problems
    * Thompson Sampling: stationary, bernoulli distributed problems

    Args:
        num_arms: the number of arms to consider
    """

    def __init__(self, num_arms: int):
        self._num_arms = num_arms

    def __str__(self):
        return self.__class__.__name__

    def get_best_arm(self) -> int:
        """Greedily select the current best arm."""
        # Since NaNs are considered max, set all NaNs to zero first
        # to avoid selecting them.
        return np.argmax(np.nan_to_num(self.arm_probabilities, nan=0))

    @abstractmethod
    def select_arm(self) -> int:
        """Select an arm based on the optimizer and the current probabilities."""
        raise NotImplementedError

    @abstractmethod
    def optimize(self, arm: int, reward: float) -> None:
        """Perform an optimization step for the given arm.

        Args:
            arm: the selected arm that generated the provided reward
            reward: the reward generated by selecting the provided arm
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def arm_probabilities(self) -> np.ndarray:
        """The pseudo probabilities for each arm.

        These are "pseudo probabilities" because they do not necessarily have
        to add up to 1. For an example, see the :class:`EpsilonGreedyBanditOptimizer`.
        """
        raise NotImplementedError


class EpsilonGreedyBanditOptimizer(BanditOptimizer):
    """Epsilon greedy optimizer for multi-armed bandits.

    The optimizer will select its current best guess at the best arm
    with a probability of 1-epsilon, and randomly otherwise. The probability
    for the selected arm is then updated according to the following formula:
    .. math::

        \\frac{1}{N_a} \\sum_{t=0}^{T} r_t^a

    where math:`N_a` is the number of times the given arm was chosen, and math:`r_t^a`
    is the sum of all rewards so far provided by this arm.

    Epsilon defaults to a constant. This can be annealed over time as desired by
    setting the .epsilon attribute.

    The probabilities are initialized implicitly (due to division by zero), to NaNs.
    NaNs are always the max in an array, and is thus equivalent to an optimistic
    initialization. Any reward received will reduce the probability of the arm, leaving
    the others as still viable greedy targets (due to the NaN, until all are tried
    and then reduced).

    This optimizer is very generic and can be used in any setting.

    Args:
        num_arms: the number of arms to consider
        epsilon: the probability to select an arm at random
    """

    def __init__(self, num_arms: int, epsilon: float):
        super().__init__(num_arms)
        assert 0 < epsilon <= 1.0
        self._epsilon = epsilon

        self.rewards = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)

    @property
    def epsilon(self) -> float:
        """The probability an arm will be chosen at random, best otherwise."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        """Set epsilon, e.g. to decay it over time."""
        self._epsilon = epsilon

    def select_arm(self) -> int:
        """Randomly select an arm epsilon amount of the time."""
        if np.random.random() < self.epsilon:
            arm = np.random.randint(0, self._num_arms)
        else:
            arm = np.argmax(self.arm_probabilities)
        return arm

    def optimize(self, arm: int, reward: float) -> None:
        """Update running count and reward for the given arm."""
        self.arm_counts[arm] += 1
        self.rewards[arm] += reward

    @property
    def arm_probabilities(self) -> np.ndarray:
        """Average reward for each arm individually."""
        return self.rewards / self.arm_counts


class SoftmaxBanditOptimizer(BanditOptimizer):
    """Boltzmann Exploration (Softmax) optimizer for multi-armed bandits.

    This optimizer picks each arm with a probability based on a Boltzmann distribution
    proportional to its average reward. The temperature parameter, tau, controls
    how greedy the distribution is shaped (i.e., giving higher probabilities to the
    current best guesses). When tau = 0 (which it can not due to division by zero), the
    optimizer will pick the greedy arm always. An infinitely large tau picks arms
    uniformly.

    Based on toy problem experiments, taus << 1 work best (e.g. 0.1). Note that
    currently this implementation runs very slowly.

    References:
        https://www.cs.mcgill.ca/~vkules/bandits.pdf, Sec. "Boltzmann Exploration"

    Args:
        num_arms: the number of arms to consider
        temperature: how greedy the optimizer is; tau, see docstring
    """

    def __init__(self, num_arms: int, temperature: float):
        super().__init__(num_arms)
        self.temp = temperature
        self.rewards = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)

    def select_arm(self) -> int:
        """Randomly draw an arm based on their probabilities."""
        return np.random.choice(self._num_arms, 1, p=self.arm_probabilities)[0]

    def optimize(self, arm: int, reward: float) -> None:
        """Update running count and reward for the given arm."""
        self.arm_counts[arm] += 1
        self.rewards[arm] += reward

    @property
    def arm_probabilities(self) -> np.ndarray:
        """Boltzmann distribution over average rewards.

        Formula:
        .. math::

            p_i = \\frac{e^{\\mu_i(t)/\\tau}}{\\sum_{a=1}^{n_{arms}}e^{\\mu_a(t)/\\tau}}

        """
        mu = np.array(
            [self.rewards[i] / self.arm_counts[i] for i in range(self._num_arms)]
        )
        # Replace any nans occurring from no samples (arm_counts == 0)
        # to the an optimistic 1
        mu[self.arm_counts == 0] = 1
        mu_temped = mu / self.temp
        denominator = np.sum([np.e ** (mu_temped[i]) for i in range(self._num_arms)])
        p = np.array(
            [np.e ** (mu_temped[i]) / denominator for i in range(self._num_arms)]
        )
        return p


class ThompsonSamplingBanditOptimizer(BanditOptimizer):
    """Thompson sampling optimizer for multi-armed bandits.

    Thompson sampling is a probability matching procedure in which the currently
    best estimated arm has the highest probability (drawn from a beta distribution)
    of being chosen, but since it is a probability, another arm can still be chosen
    instead. As more and more data is collected, the distributions tighten and the
    (hopefully) optimal solution is chosen more often. Thompson sampling can only be
    used in a binary (e.g., win/loss, success/failure, 1/0) reward problem. This is
    because it is based on the Bernoulli distribution and deals with counts of
    successes and failures only. Another way of looking at Thompson sampling is to
    say that each arm has a prior distribution, and the observed reward updates their
    posteriors.

    The probability an arm will be selected is defined by the following equation:
    .. math::

        \\frac{\\alpha}{\\alpha + \\beta}

    The alpha and beta parameters of the Beta distribution can be interpreted
    as the expected number of successes and failures, respectively, of the randomly
    generated value. For example, if alpha = beta = 1, the reward probability
    (probability of success) is 50% (uniform), with low confidence. If alpha = 1000 and
    beta = 9000, the reward probability is 10% with high confidence. In general, when
    alpha becomes larger (more successful events), the bulk of the probability
    distribution will shift towards the right, whereas an increase in beta moves the
    distribution towards the left (more failures). The distribution will narrow if both
    alpha and beta increase, implying certainty.

    Updating the probabilities involves updating the alpha and beta parameters for
    the given arm. The following equation gives the current alpha and beta params:
    .. math::

        \\alpha_t, \\beta_t = \\alpha_0 + successs, \\beta_0 + failures

    Beta Distribution Interpretation Sources:

        * https://lilianweng.github.io/lil-log/2018/01/23/
          the-multi-armed-bandit-problem-and-its-solutions.html
        * https://towardsdatascience.com/beta-distribution-intuition
          -examples-and-derivation-cf00f4db57af

    Args:
        num_arms: the number of arms to consider
        success_prior_a: the beta distributions' alpha parameter;
                         can be interpreted as the number of assumed successes.
                         A list of values initializes each arm's alpha individually;
                         otherwise the same value is repeated for each arm.
        failure_prior_b: the beta distributions' beta parameter;
                         can be interpreted as the number of assumed failures.
                         A list of values initializes each arm's alpha individually;
                         otherwise the same value is repeated for each arm.
    """

    def __init__(
        self,
        num_arms: int,
        success_prior_a: Union[int, float, List[Union[int, float]]],
        failure_prior_b: Union[int, float, List[Union[int, float]]],
    ):
        super().__init__(num_arms)
        if isinstance(success_prior_a, list):
            self._beta_a = np.array(success_prior_a)
        else:
            self._beta_a = np.array([success_prior_a] * num_arms)

        if isinstance(failure_prior_b, list):
            self._beta_b = failure_prior_b
        else:
            self._beta_b = np.array([failure_prior_b] * num_arms)

    def beliefs(self) -> np.ndarray:
        """The expected rewards (Q, or beliefs thereof) of each arm.

        The expected rewards are drawn from a beta distribution, and thus are variable.
        """
        return np.random.beta(self._beta_a, self._beta_b)

    def select_arm(self) -> int:
        return np.argmax(self.beliefs())

    def optimize(self, arm: int, reward: float) -> None:
        """Update alpha and beta with counts of successes/failures."""
        assert reward in [0, 1], f"TS requires a binary reward structure. ({reward})"
        self._beta_a[arm] += reward
        self._beta_b[arm] += 1 - reward

    @property
    def arm_probabilities(self) -> np.ndarray:
        """Arm probabilities are defined as the mean of the distributions."""
        return np.array([a / (a + b) for a, b in zip(self._beta_a, self._beta_b)])


class UCB1BanditOptimizer(BanditOptimizer):
    """Upper Confidence Bounds 1 sampling optimizer for multi-armed bandits.

    UCB1 adds a confidence bound on the arm's probability estimate to help choose
    which arm to pull. Higher confidence bounds are chosen more often, in order to
    increase confidence in the arm. In this way, the algorithm is optimistic about
    options with high uncertainty and acts greedily with respect to that uncertainty.

    Specifically, the confidence bound is added to the estimated mean and this sum is
    greater than or equal to the true mean, with a high probability (Hoeffding's
    inequality). The upper bound is a function of the number of times an arm is played;
    a larger number of trials should give us a smaller bound:
    .. math::

        \\sqrt{\\frac{2\\mathrm{ln}t}{N_a}}

    for any given arm, where math:`N_a` is the number of times said arm was pulled, and
    math:`t` is the total number of trials.

    The optimizer always starts by pulling each arm sequentially until all have
    been pulled at least once.

    References:
        https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html
        https://www.cs.mcgill.ca/~vkules/bandits.pdf

    Args:
        num_arms: the number of arms to consider
    """

    def __init__(self, num_arms: int):
        super().__init__(num_arms)
        self.rewards = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)

    def select_arm(self) -> int:
        """Select the arm with the maximal estimated mean plus upper confidence bound.

        .. math::

            A_i = \\underset{a=1...n_{arms}}{\\mathrm{arg\\,max}}
            \\left(\\mu_a + UCB_a\\right)

        If the number of trials is less than the number of arms, return each
        arm sequentially until the number of trials is greater than the number
        of arms.
        """
        if self.num_trials < self._num_arms:
            return int(self.num_trials)
        return np.argmax(self.arm_probabilities + self.upper_bounds)

    def optimize(self, arm: int, reward: float) -> None:
        """Update running count and reward for the given arm."""
        self.arm_counts[arm] += 1
        self.rewards[arm] += reward

    @property
    def upper_bounds(self) -> np.ndarray:
        """The upper confidence bounds for each arm."""
        return np.sqrt(
            [
                (2 * np.log(self.num_trials)) / self.arm_counts[i]
                for i in range(self._num_arms)
            ]
        )

    @property
    def num_trials(self) -> int:
        """The number of arm pulls thus far."""
        return np.sum(self.arm_counts)

    @property
    def arm_probabilities(self) -> np.ndarray:
        """Average reward for each arm individually."""
        return self.rewards / self.arm_counts


class SlidingWindowUCBBanditOptimizer(BanditOptimizer):
    """Sliding Window UCB optimizer for multi-armed bandits.

    The sliding window UCB optimizer is the same as the :class:`UCB1BanditOptimizer`,
    except that it only considers the last w trials when computing the confidence
    bounds and means. Due to the window, the optimizer can adapt to a nonstationary
    distribution and thereby alter the best arm over time.

    Note that the reference implementation includes a beta parameter (which was set
    to 1 in the paper) and does not multiply the UCB log by 2. For consistency,
    this implementation uses the UCB calculation from :class:`UCB1BanditOptimizer`
    and only adds on top the window calculation.

    References:
        https://arxiv.org/pdf/2003.13350.pdf, Appendix D

    Args:
        num_arms: the number of arms to consider
        window_size: How many past trials to compute means and confidence bounds over
    """

    def __init__(self, num_arms: int, window_size: int):
        super().__init__(num_arms)
        self.window_size = window_size
        # Windowed history of rewards and pulls for each arm
        self.rewards: List[Deque[float]] = []
        self.arm_pulls: List[Deque[int]] = []
        # Instantiate in a loop to avoid undesired copies
        for _ in range(num_arms):
            self.rewards.append(deque(maxlen=window_size))
            self.arm_pulls.append(deque(maxlen=window_size))

        # Keep track of arms that have gone through at least one optimization step.
        # This ensures all arms are chosen before considering the upper bounds.
        # A more complicated calculation is required than just returning the current
        # step, since that value can get corrupted when mixing in random actions, as
        # in :class:`EpsilonGreedySlidingWindowUCBBanditOptimizer`.
        self._optimized_arms: Set[int] = set()

    def select_arm(self) -> int:
        """Select the arm with the maximal estimated mean plus upper confidence bound.

        See :class:`UCB1BanditOptimizer`'s :meth:`select_arm` method for details.
        """
        if set(range(self._num_arms)) != self._optimized_arms:
            return sorted(list(set(range(self._num_arms)) - self._optimized_arms))[0]
        return np.argmax(self.arm_probabilities + self.upper_bounds)

    def optimize(self, arm: int, reward: float) -> None:
        """Update the running counts and rewards for all arms.

        Zeros are added for non used arms in order to be used as a mask to
        calculate counts and rewards over a window.
        """
        self._optimized_arms.add(arm)
        self.rewards[arm].append(reward)
        self.arm_pulls[arm].append(1)
        unused_arms = list(range(self._num_arms))
        del unused_arms[arm]
        for unused_arm in unused_arms:
            # Append a NaN to to unused arm's reward.
            # It is masked out during average calculation.
            self.rewards[unused_arm].append(np.nan)
            self.arm_pulls[unused_arm].append(0)

    @property
    def windowed_rewards(self) -> np.ndarray:
        arms = [np.array(r) for r in self.rewards]
        pulls = [np.array(p) for p in self.arm_pulls]
        return np.array([np.sum(arm[pull == 1]) for arm, pull in zip(arms, pulls)])

    @property
    def windowed_pulls(self) -> np.ndarray:
        return np.array([np.sum(p) for p in self.arm_pulls])

    @property
    def upper_bounds(self):
        """The upper confidence bounds for each arm."""
        bounds = np.sqrt(
            [
                (2 * np.log(self.num_trials)) / self.windowed_pulls[i]
                for i in range(self._num_arms)
            ]
        )
        return bounds

    @property
    def num_trials(self) -> int:
        """The count of all arm pulls thus far, up to the window size."""
        # All deques are of the same size, so only look at one
        trials = len(self.arm_pulls[0])
        assert trials <= self.window_size
        return trials

    @property
    def arm_probabilities(self) -> np.ndarray:
        """Average reward for each arm individually."""
        return self.windowed_rewards / self.windowed_pulls


class EpsilonGreedySlidingWindowUCBBanditOptimizer(SlidingWindowUCBBanditOptimizer):
    """Sliding Window UCB optimizer mixed with epsilon greedy for multi-armed bandits.

    This optimizer behaves the same as the :class:`SlidingWindowUCBBanditOptimizer`,
    but also includes an element of randomness to increase the exploration. It was
    proposed for Agent57, and also used a simplified UCB calculation, which can
    be turned on by using the :param:`agent57_ucb` boolean.

    References:
        https://arxiv.org/pdf/2003.13350.pdf, Appendix D

    Args:
        num_arms: the number of arms to consider
        window_size: How many past trials to compute means and confidence bounds over
        epsilon: the probability to select an arm at random
        agent_57_ucb: Use the simplified confidence bound calculation as seen in the
                      Agent57 paper.
    """

    def __init__(
        self,
        num_arms: int,
        window_size: int,
        epsilon: float,
        agent_57_ucb: bool = False,
    ):
        super().__init__(num_arms, window_size)
        self._epsilon = epsilon
        self.use_agent57_ucb = agent_57_ucb

    def select_arm(self) -> int:
        """Randomly select an arm epsilon amount of the time, UCB's arm otherwise."""
        if (
            set(range(self._num_arms)) == self._optimized_arms
            and np.random.random() < self.epsilon
        ):
            # Only start eps greedy once all arms were chosen at least once.
            # This happens after the first n arm pulls, where n is the number of arms.
            arm = np.random.randint(0, self._num_arms)
        else:
            arm = super().select_arm()
        return arm

    @property
    def upper_bounds(self):
        """The upper confidence bounds for each arm."""
        if self.use_agent57_ucb:
            return np.sqrt(
                [(1 / self.windowed_pulls[i]) for i in range(self._num_arms)]
            )
        return super().upper_bounds

    @property
    def epsilon(self) -> float:
        """The probability an arm will be chosen at random."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        """Set epsilon, e.g. to decay it over time."""
        self._epsilon = epsilon
