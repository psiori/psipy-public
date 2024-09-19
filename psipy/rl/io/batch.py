# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""
TODO: Module description.

TODO: Maybe move to :mod:`psipy.rl.preprocessing`?
"""

import glob
import itertools
import logging
import os
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from tensorflow.keras.utils import Sequence as KSequence

from psipy.rl.core.controller import Controller
from psipy.rl.io.sart import SARTReader
from psipy.rl.core.plant import Numeric

__all__ = ["Batch", "Episode"]


LOG = logging.getLogger(__name__)

#: Small epsilon used in prioritization.
e = 1e-8


class Episode:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
        costs: np.ndarray,
        lookback: int = 1,
    ):
        """A history of all observations, actions, terminals, and costs in an episode.

        The episode class is the base for all RL learning.  It contains all the
        relevant information about the performance of a controller in a plant.
        Episodes are also the basis of Batch, which allows for random sampling
        across multiple episodes for training.

        Note how some data needed to be dropped since they are hanging (either missing
        an action or missing a state).

        Args:
            observations: float32 matrix of observations, time as first dimension.
            actions: float32 vector of actions over time.
            terminals: bool vector of terminal over time.
            costs: float32 vector of costs over time.
            lookback: Amount of timesteps to merge into a single state stack.
        """
        assert len(observations) == len(actions) == len(terminals) == len(costs)
        self._observations = np.asarray(observations, dtype=np.float32)
        self._actions = np.asarray(actions, dtype=np.float32)
        if len(self._actions.shape) == 1:
            self._actions = self._actions[..., None]
        assert len(self._actions.shape) == 2
        self._terminals = np.asarray(terminals, dtype=bool).ravel()
        self._costs = np.asarray(costs, dtype=np.float32).ravel()
        if lookback < 1:
            LOG.warning("Lookback is invalid, setting to 1.")
        self.lookback = max(lookback, 1)

    def __len__(self) -> int:
        """Amount of transitions stored, max legal index + 1."""
        return max(len(self._actions) - self.lookback, 0)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_transitions(index)

    def __eq__(self, other):
        if not isinstance(other, Episode):
            return False
        this = [
            self._observations,
            self._actions,
            self._terminals,
            self._costs,
            self.lookback,
        ]
        that = [
            other._observations,
            other._actions,
            other._terminals,
            other._costs,
            other.lookback,
        ]
        return all(np.array_equal(x, y) for x, y in zip(this, that))

    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.lookback - 1, len(self._actions) - 1, dtype=np.uint32)

    def get_transitions(
        self, index: Union[List[int], Tuple[int, ...], int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns transitions at the given indices.

        A transition here is given as a stack of observations, lookback + 1 many.
        The first lookback many of those observations are the first state of the
        transition, the last lookback many are the second state of the transition.

        states = stack[..., :-1]
        next_states = stack[..., 1:]

        In other words, stack is of the form (when lookback == 1)::

            [[s, s+1],
             [s+1, s+2],
             [s+2, s+3],
              ...]

        Example:

            >>> eps = Episode([[1]] * 4 + [[2]], [[1]] * 4 + [[0]],
            ...               [False] * 5, [0] * 5, lookback=2)
            >>> stack, act = eps.get_transitions([0, 2])
            >>> stack.shape[0] == 2
            True
            >>> stack.shape[-1] == eps.lookback + 1 == 3
            True
            >>> act.shape[0] == 2
            True
            >>> stack[1, ..., :-1].tolist(), stack[1, ..., 1:].tolist()
            ([[1.0, 1.0]], [[1.0, 2.0]])

        """
        indices = np.asarray([index]).ravel().astype(int).tolist()
        assert max(indices) < len(self), f"{max(indices)} < {len(self)}"
        indices = [index + self.lookback - 1 for index in indices]
        assert max(indices) < len(self._observations)

        lookup = [range(idx - self.lookback + 1, idx + 2) for idx in indices]
        stack = self._observations[np.asarray(lookup, dtype=np.uint32), ...]
        stack = np.swapaxes(stack, 1, -1)
        assert stack.shape[0] == len(indices)
        assert stack.shape[-1] == self.lookback + 1  # s and s_ in one

        actions = self._actions[indices]
        return stack, actions

    @property
    def observations(self) -> np.ndarray:
        """Observations excluding initial lookback - 1 many observations"""
        return self._observations[self.lookback - 1 :]

    @property
    def all_observations(self) -> np.ndarray:
        """All observations, including initial lookback - 1 many observations"""
        return self._observations

    @property
    def reward(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def costs(self) -> np.ndarray:
        """Costs excluding initial lookback - 1 many observations"""
        return self._costs[self.lookback - 1 :]

    @property
    def terminals(self) -> np.ndarray:
        """Terminal state of each transition.

        For each observation in self.observations, this holds whether employing its
        given action resulted in a terminal at the following state.

        TODO: Maybe convert to a method to avoid copying the data before get_samples in
              Batch accesses the actual indices its interested in?
        """
        return self._terminals[self.indices + 1]

    @staticmethod
    def remove_string_axes(arr: np.ndarray) -> np.ndarray:
        """String axes within an array (such as action addtl data), are removed."""
        deletable = []
        arr = arr.T
        for axis in range(len(arr)):
            if isinstance(arr[axis][0], str):
                deletable.append(axis)
        if len(deletable) > 0:
            arr = np.delete(arr, deletable, 0)
        return arr.T

    @classmethod
    def from_hdf5(
        cls,
        filepath: str,
        lookback: int = 1,
        state_channels: Optional[Sequence[str]] = None,
        action_channels: Optional[Sequence[str]] = None,
    ) -> "Episode":
        """Loads a SART hdf5 episode file to create an Episode instance.

        Args:
            state_channels: only load these channels
            action_channels: only load these channels
        """
        LOG.debug(f"Loading episode: {filepath}")
        with SARTReader(filepath) as reader:
            episode = reader.load_full_episode(state_channels, action_channels)
        o, a, t, c = episode
        a = cls.remove_string_axes(a)
        return cls(observations=o, actions=a, terminals=t, costs=c, lookback=lookback,)

    @classmethod
    def multiple_from_key_hdf5(
        cls,
        filepath: str,
        key_source: str,
        key: str,
        value: Optional[Union[str, Numeric]] = None,
        lookback: int = 1,
        state_channels: Optional[Sequence[str]] = None,
        action_channels: Optional[Sequence[str]] = None,
    ) -> Dict[Union[Numeric, str], List["Episode"]]:
        """Load potentially n-many Episodes from .h5, clustered by an action-/meta key.

        With a key, a SART file can be split into n-many clusters of transitions,
        split based on the given key.  A certain value for the key can also be
        provided, in order to drop other values of the key. For example:

            t1 = (1,2,3), option=left
            t2 = (0,1,2), option=left
            t3 = (1,2,3), option=right
            t4 = (2,3,4), option=right
            t4 = (3,4,5), option=right
            t5 = (2,3,4), option=left
            t6 = (1,2,3), option=left

        and the key value pair used was "option/left", the resulting episodes would
        be created:

            Episodes = [Episode((1,2,3),(0,1,2)), Episode((2,3,4),(1,2,3))]

        Args:
            filepath: filepath to the SART file
            key_source: where to find the key, either in the action addtl info or meta
            key: the key to split on
            value: optional value of the key to keep and drop others
            state_channels: only load these channels
            action_channels: only load these channels

        Returns:
            Dict of episodes where the dict key is the value of the meta key
            that was split.
        """

        def get_meta_channel(channels: List[str], key: str):
            """Find the full path of the given key."""
            for channel in channels:
                if channel.endswith(key):
                    return [channel]  # pack as iterable
            raise KeyError(f"{key} not found in {key_source}!")

        assert key_source in ["action", "meta"]
        LOG.debug(f"Loading episode to split on {key}: {filepath}")

        # Get the meta information
        with SARTReader(filepath) as reader:
            if key_source == "meta":
                meta_channels = reader.file.attrs["meta"][:]
                meta = reader.load_meta(get_meta_channel(meta_channels, key))
                # Unpack from dict
                meta = meta[key]
            else:
                meta_channels = reader.file.attrs["action"][:]
                _, meta, _, _ = reader.load_full_episode(
                    action_channels=get_meta_channel(meta_channels, key)
                )
                # Unpack from single element array
                meta = meta.ravel()

        # Load the episode
        with SARTReader(filepath) as reader:
            episode = reader.load_full_episode(
                state_channels=state_channels, action_channels=action_channels
            )
        o, a, t, c = episode
        a = cls.remove_string_axes(a)
        # In the action case, the last action will contain empty information
        # because it is a terminal
        if len(meta) != len(o):
            meta.append(meta[-1])

        # Now split the data based on the provided key
        split_episodes: Dict[Union[Numeric, str], List["Episode"]] = defaultdict(list)
        end = 0
        for value_key, group in itertools.groupby(
            zip(meta, range(len(meta))), key=lambda x: x[0]
        ):
            start = end
            # Group is a list of n many steps.
            # We only want the last one's range value (index:=1)
            end = list(group)[-1][1] + 1
            episode = cls(
                observations=o[start:end],
                actions=a[start:end],
                terminals=t[start:end],
                costs=c[start:end],
                lookback=lookback,
            )
            split_episodes[value_key].append(episode)

        # We now have a dictionary where the keys are the values of the key
        # and values are episodes that fit that key.
        # If provided, drop key values that don't match the desired value
        if value is not None:
            if value not in split_episodes.keys():
                raise KeyError(
                    f"{value} can not be selected from meta: {split_episodes.keys()}"
                )
            split_episodes = {value: split_episodes[value]}

        return split_episodes

    def is_valid(self) -> bool:
        return not np.any(np.isnan(self._observations))


class Batch(KSequence):
    """A Batch consists of many Episodes.

    Primary goal of this class is to provide an easy way to work with an arbitrary
    amount of episodes for neural network training.

    Indexing on the :class:`Batch` class instance (`batch[0]`, `batch[1]`...) returns
    minibatches of data according to `batch.minibatch_size`, not individual
    samples. The indices resolving minibatches to individual samples are stored
    in `batch._minibatch_indices`, pointing to (episode, sample) tuples in
    `batch._indices`. Indexing on :class:`Episode` class instances on the other hand
    returns individual samples.

    The samples within the batch can be sampled via a prioritization distribution
    which gives more weight to higher error samples. Currently, rank based and
    proportional prioritization modes are available. There are two parameters that
    control the strength of prioritization, detailed in the Args below.

    Args:
        episodes: sequence of Episodes for creating minibatches of their transitions
        control: the controller learning from this batch
        prioritization: "proportional", "rank", or None
        alpha: how much prioritization to use in [0,1]. 1 implies "full strength"
               prioritization, and 0 is normal uniform sampling.
        beta: importance sampling strength for prioritization in [0,1]. These
              weights are multiplied by the loss in order to correct for the
              altered data distribution that prioritization causes. 0 implies
              no correction and 1 full correction. The PER paper suggests
              annealing beta from ~.5 to 1 over the course of training. Note
              that the choice of this hyperparameter interacts with choice of
              prioritization exponent alpha; increasing both simultaneously
              prioritizes sampling more aggressively at the same time as
              correcting for it more strongly (PER Sec. 3.4).
    """

    legal_modes: ClassVar[Tuple[str, ...]] = (
        "costs",
        "costs_terminals",
        "nextstates",
        "states",
        "statesactions_targets",
        "states_actions",
        "states_costs",
        "states_targets",
    )

    _episodes: List[Episode]
    _minibatch_size: int
    _state_shape: Tuple[int, ...]
    _action_shape: Tuple[int, ...]
    _mode: str

    #: The .h5 paths that were loaded into the batch.
    _loaded_sart_paths: Set[str]

    #: The .h5 paths that will be not loaded by the batch.
    _ignored_sart_paths: Set[str] = set()

    #: 2D np.uint8, num_samples many rows, self._episodes index in col one,
    #: transition-inside-episode index in col 2
    _indices: np.ndarray

    #: List of indices into self._indices
    _minibatch_indices: List[np.ndarray]

    def __init__(
        self,
        episodes: Sequence[Episode],
        control: Optional[Controller] = None,
        prioritization: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        if len(episodes) == 0:
            raise ValueError("No episodes passed to Batch!")
        self._episodes = [e for e in episodes if len(e) > 0]
        self._state_shape = self._episodes[0]._observations[0].shape
        self._action_shape = self._episodes[0]._actions[0].shape
        self._is_shuffled = False
        self._control = control
        # self.minibatch_size = -1
        assert all(eps.lookback == self.lookback for eps in self._episodes)
        self._initialize()
        self.minibatch_size = -1

        # Prioritization parameters
        assert prioritization in ["proportional", "rank", "time", None]
        self.prioritization = prioritization
        self.alpha = alpha  # prioritization exponent
        self.beta = beta  # importance sampling exponent
        self.weights: Optional[np.ndarray] = None

        self.shuffle()  # Trigger initial shuffle

    def __len__(self) -> int:
        return self.num_minibatches

    def __getitem__(self, index: int) -> Union[Tuple, np.ndarray]:
        """Get single minibatch."""
        # Because len(episode) excludes the full lookback length, but accessing
        # any obs or cost is x[lookback-1:], len(episode) excludes the terminal
        # state in its indices (i.e. x[indices] = x[lookback-1:-1] when lookback==2)
        indices = self._minibatch_indices[index]  # 1, 5, 3, 2, 78, 2134
        return self.get_samples(indices)

    def _initialize(self):
        """Init the batch by creating indices and applying transformations"""
        indices: List[Tuple[int, int]] = []
        for i, ep in enumerate(self._episodes):
            # Creates flat list of [episode #, transition #]
            indices.extend(zip([i] * len(ep), range(len(ep))))
        self._indices = np.asarray(indices, dtype=np.uint32)

    def get_samples(
        self, indices: Union[int, List[int], Tuple[int, ...], slice]
    ) -> Union[Tuple, np.ndarray]:
        """Conditionally collect and return different types of samples."""
        assert hasattr(self, "_mode") and self._mode in self.legal_modes
        if isinstance(indices, slice):
            start = indices.start if indices.start is not None else 0
            step = indices.step if indices.step is not None else 1
            stop = indices.stop if indices.stop is not None else self.num_samples
            indices = list(range(start, stop, step))
        elif isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)
        indices = np.asarray(indices, dtype=np.uint32)

        # Convert a list of indices pointing to individual transitions in the
        # full batch to sorted groups of index pairs pointing to episodes and
        # transitions within these episodes. The groups and indices within are
        # sorted for performance reasons, in order to be able to extract all
        # transitions from a single episode in one go, instead of doing so
        # transition by transition. A side effect of sorting the transition
        # indices is that also the initial incoming indices need to be
        # permuted accordingly, because they are used to extract costs and
        # target values on the batch level.
        gindices = self._indices[indices].tolist()  # [1, 32], [6, 1231], [4, 1235]
        permutation, gindices = zip(*sorted(enumerate(gindices), key=itemgetter(1)))
        indices = indices[np.asarray(permutation)]
        groups = groupby(gindices, key=itemgetter(0))

        # Allocate memory.
        if "states" in self._mode:
            shape = (len(indices),) + self._state_shape + (self.lookback + 1,)
            stacks = np.empty(shape, dtype=np.float32)
        if "actions" in self._mode:
            actions = np.empty((len(indices),) + self._action_shape, dtype=np.float32)
        if "terminals" in self._mode:
            terminals = np.empty(len(indices), dtype=bool)
        if "costs" in self._mode:
            if not hasattr(self, "_costs"):
                costs = np.empty((len(indices),), dtype=np.float32)
            else:
                costs = self._costs[indices]

        # Collect samples from each episode in groups.
        head = 0
        for g, index_pairs in groups:
            pairs = list(index_pairs)
            episode = self._episodes[g]
            eindices = np.asarray(pairs)[:, 1]
            if "states" in self._mode:
                stacks_, actions_ = episode.get_transitions(eindices)
                stacks[head : head + len(pairs), ...] = stacks_
            if "actions" in self._mode:
                actions[head : head + len(pairs), ...] = actions_
            if "terminals" in self._mode:
                terminals[head : head + len(pairs)] = episode.terminals[eindices]
            if "costs" in self._mode and not hasattr(self, "_costs"):
                costs[head : head + len(pairs), ...] = episode.costs[eindices]
            head += len(pairs)

        # Importance sampling weight collection
        ip_weights = np.ones(len(indices))
        if self.prioritization and self.weights is not None:
            # Weights are not saved in episodes, and so can be indexed directly
            ip_weights = np.array(self.weights)[indices][..., None]

        if self._mode == "costs":
            return costs
        if self._mode == "costs_terminals":
            return costs, terminals[..., None]
        if self._mode == "nextstates":
            return self.preprocess_stacks(stacks[..., 1:])
        if self._mode == "states":
            return self.preprocess_stacks(stacks[..., :-1])
        if self._mode == "statesactions_targets":
            assert hasattr(self, "_targets"), "Targets need to be set before get."
            targets = self._targets[indices]
            # assert np.all(targets >= 0)
            states = self.preprocess_stacks(stacks[..., :-1])
            actions = self.preprocess_actions(actions)
            if isinstance(states, dict):
                states["actions"] = actions
                return states, targets, ip_weights  # (sa, t, sample_weights)
            return (states, actions), targets, ip_weights  # ((sa), t, sample_weights)
        if self._mode == "states_actions":
            states = self.preprocess_stacks(stacks[..., :-1])
            actions = self.preprocess_actions(actions)
            return states, actions
        if self._mode == "states_costs":
            states = self.preprocess_stacks(stacks[..., :-1])
            return states, costs
        if self._mode == "states_targets":
            assert hasattr(self, "_targets"), "Targets need to be set before get."
            targets = self._targets[indices]
            assert np.all(targets >= 0)
            states = self.preprocess_stacks(stacks[..., :-1])
            return states, targets
        raise ValueError("Bad mode")

    def on_epoch_end(self) -> None:
        """Reshuffles samples into minibatches, called by keras after every batch."""
        self.shuffle()  # also regenerates the priorities

    def set_delta(self, delta: np.ndarray) -> None:
        """Set the TD-errors for the all samples."""
        assert (
            len(delta) == self.num_samples
        ), f"Deltas must be calculated on all data. ({len(delta)}/{self.num_samples})"
        self._sorted_delta = delta

    def _calculate_priorities(self) -> np.ndarray:
        """Calculate the prioritization probabilities.

        The sorted TD error (_sorted_delta) will always be used to calculate the
        priorities so that the batch can be shuffled multiple times in a row
        without messing up the order.

        Three modes are implemented: proportional, rank based, and time based.
        Proportional is based entirely off of the TD error values, where higher
        errors yield higher priority proportional to their own error. Rank based
        gives higher value to higher errors based on their rank (position) within
        the sorted errors. Lastly, time gives more priority to newer samples, and
        is, as far as is known, not in the literature.
        
        See "Prioritized Experience Replay" for more information:
            https://arxiv.org/abs/1511.05952
        """
        if self.prioritization == "proportional":
            priorities = np.array(
                [(np.abs(delta) + e) ** self.alpha for delta in self._sorted_delta]
            )
        elif self.prioritization == "rank":
            # Sort based on value, not by sample
            priorities = np.zeros(len(self._sorted_delta))
            # Set priority based on rank; equal deltas receive the same rank
            prev_delta = np.max(self._sorted_delta)
            i = 0
            for index, delta in sorted(
                enumerate(np.abs(self._sorted_delta)), key=itemgetter(1), reverse=True,
            ):
                if delta != prev_delta:
                    i += 1  # only increase rank if deltas are not the same
                    prev_delta = delta
                priorities[index] = 1 / (i + 1)  # avoid 0 division
        elif self.prioritization == "time":
            priorities = np.array(
                list(reversed([1 / i for i in range(1, len(self._sorted_delta) + 1)]))
            )
        else:
            raise ValueError(f"Invalid prioritization mode '{self.prioritization}'.")

        total_priorities = np.sum(priorities)
        return (priorities / total_priorities).ravel()

    def get_sample_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the sorted sample probabilities and importance sampling weights."""
        p = self._calculate_priorities()
        weights = (self.num_samples * p) ** -self.beta
        weights = weights / np.max(weights)
        LOG.debug(f"Priorities: min({np.min(p)}); max({np.max(p)}); alpha={self.alpha}")
        LOG.debug(
            f"IP Weights: min({np.min(weights)}); "
            f"max({np.max(weights)}); "
            f"beta={self.beta}"
        )

        return p, weights

    def sort(self) -> "Batch":
        # Ensure equal sized batches
        num_samples = self.num_minibatches * self.minibatch_size
        indices = np.arange(0, num_samples)
        self._minibatch_indices = np.array_split(indices, self.num_minibatches)
        self._is_shuffled = False
        if hasattr(self, "_sorted_delta"):
            # Sorts the weights, p is unused since the batch is sorted
            _, self.weights = self.get_sample_distribution()
        return self

    def shuffle(self) -> "Batch":
        # Ensure equal sized batches
        num_samples = self.num_minibatches * self.minibatch_size
        if self.prioritization and hasattr(self, "_sorted_delta"):
            p, self.weights = self.get_sample_distribution()
            indices = np.random.choice(self.num_samples, num_samples, p=p)
        else:
            # Permute actual indices, then truncate len to multiple of batchsize
            indices = np.random.permutation(self.num_samples)[:num_samples]
        self._minibatch_indices = np.array_split(indices, self.num_minibatches)
        self._is_shuffled = True
        return self

    @property
    def num_samples(self) -> int:
        return sum(len(episode) for episode in self._episodes)

    @property
    def num_minibatches(self) -> int:
        return self.num_samples // self.minibatch_size

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def lookback(self) -> int:
        return self._episodes[0].lookback

    @property
    def controller(self) -> Optional[Controller]:
        return self._control

    def compute_costs(self, costfunc: Callable[[np.ndarray], np.ndarray]) -> "Batch":
        costs = np.zeros((self.num_samples,), dtype=np.float32)
        head = 0
        for episode in self._episodes:
            costs_ = costfunc(episode.all_observations)
            # We roll because costs are based on next states, and s,u == s'
            costs_ = np.roll(costs_, -1)  # costs(s, u) == costs(s')
            # Keep track of all but the lookback many first costs.
            costs[head : head + len(episode)] = costs_[episode.lookback - 1 : -1]
            head += len(episode)
        self._costs = costs[..., None]
        return self

    def set_targets(self, targets: np.ndarray) -> "Batch":
        targets = np.asarray(targets).astype(np.float32)
        assert len(targets) == self.num_samples, f"{len(targets)} == {self.num_samples}"
        if len(targets.shape) == 1:
            targets = targets[..., None]
        self._targets = targets
        return self

    @property
    def minibatch_size(self) -> int:
        return min(self.num_samples, self._minibatch_size)

    @minibatch_size.setter
    def minibatch_size(self, minibatch_size: int):
        if minibatch_size < 1:
            minibatch_size = self.num_samples
        self._minibatch_size = minibatch_size
        if self.is_shuffled():
            self.shuffle()
        else:
            self.sort()

    def set_minibatch_size(self, minibatch_size: int) -> "Batch":
        self.minibatch_size = minibatch_size
        return self

    def is_shuffled(self) -> bool:
        return self._is_shuffled

    @property
    def costs(self) -> "Batch":
        self._mode = "costs"
        return self

    @property
    def costs_terminals(self) -> "Batch":
        self._mode = "costs_terminals"
        return self

    @property
    def nextstates(self) -> "Batch":
        self._mode = "nextstates"
        return self

    @property
    def states(self) -> "Batch":
        self._mode = "states"
        return self

    @property
    def statesactions_targets(self) -> "Batch":
        self._mode = "statesactions_targets"
        return self

    @property
    def states_actions(self):
        self._mode = "states_actions"
        return self

    @property
    def states_costs(self):
        self._mode = "states_costs"
        return self

    @property
    def states_targets(self):
        self._mode = "states_targets"
        return self

    @property
    def observations(self) -> np.ndarray:
        return np.vstack([eps.observations for eps in self._episodes])

    def all(self):
        """Get all data as a single array in the current mode.

        Returns data from a cache. Cache gets invalidated by other methods.
        """
        if not hasattr(self, "_caches"):
            self._caches = dict()
        if self._mode not in self._caches:
            self._caches[self._mode] = self[0]
        return self._caches[self._mode]

    def preprocess_stacks(self, stacks: np.ndarray):
        if self._control is None or not hasattr(
            self._control, "preprocess_observations"
        ):
            return stacks
        return self._control.preprocess_observations(stacks)

    def preprocess_actions(self, stacks: np.ndarray):
        if self._control is None or not hasattr(self._control, "action_normalizer"):
            return stacks
        return self._control.action_normalizer.transform(stacks)  # type: ignore

    def append(self, episodes: Sequence[Episode]) -> None:
        """Append a list of episodes to the Batch instance"""
        if hasattr(self, "_caches"):
            self._caches = dict()
        episodes = list(episodes)
        assert all(e.lookback == self.lookback for e in episodes)

        #SL going back to the original code extending the episodes to the end of the existing ones
        #SL because that allows a user to find the latest episode at the end of the list.
        #SL I am assuming there was a reason to prepent and replace the existing episodes (thread safety?)
        #SL but since there was no comment here detailing the decision going from extending to prepending,
        #SL I changed it back to prepending. If something breaks, and you need to prepend (again),
        #SL leave a comment here explaining why prepending is necessary. 
        
        ## Prepend the episodes
        #episodes.extend(self._episodes)
        #self._episodes = episodes
        self._episodes.extend(episodes)
        LOG.info(f"Batch now contains {len(self._episodes)} episodes.")
        # Regenerate the minibatch indices
        self._initialize()
        if self.is_shuffled():
            self.shuffle()
        else:
            self.sort()

    @classmethod
    def _generate_file_list(
        cls,
        dirpaths: Tuple[str, ...],
        only_newest: Optional[int] = None,
        override_mtime: bool = False,  # TODO Test
    ) -> List[str]:
        """Generate a list of hdf5 files, excluding old ones if desired."""

        def _sort_by_time(filename):
            date_time = filename.split("-")[-4:-2]
            return datetime.strptime("-".join(date_time), "%y%m%d-%H%M%S")

        files = []
        for dirpath in dirpaths:
            files.extend(glob.glob(os.path.join(dirpath, "*.h5")))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No episodes found! Check your loading directory?\n{dirpaths}"
            )
        # Remove old files if desired
        files = sorted(files, key=lambda f: os.path.getmtime(f))
        if override_mtime:
            files = sorted(files, key=_sort_by_time)
        if only_newest is not None:
            sort = files  # Cache files back into sort so that we can ignore paths
            files = sort[-only_newest:]
            cls._ignored_sart_paths = set(sort[:only_newest])
        return files

    @classmethod
    def from_hdf5(
        cls,
        *dirpaths: str,
        lookback: Optional[int] = None,
        state_channels: Optional[Sequence[str]] = None,
        action_channels: Optional[Sequence[str]] = None,
        prioritization: Optional[str] = None,
        control: Optional[Controller] = None,
        only_newest: Optional[int] = None,
        override_mtime: bool = False,
    ) -> "Batch":
        """Loads a series of SART hdf5 episode files to create a batch instance.

        Args:
            prioritization: the type of prioritization desired, None if none
            only_newest: if provided, will only load the newest n many SART files.
                         The ignored files will be ignored in any future appends.
            override_mtime: don't use mtime to sort the files; use datetimes in file
                            names instead (fails if improper naming format).
        """
        if lookback is None:
            lookback = 1
            if hasattr(control, "lookback"):
                lookback = control.lookback  # type: ignore
        assert lookback is not None  # satisfy mypy
        episodes = []
        files = cls._generate_file_list(
            dirpaths=dirpaths, only_newest=only_newest, override_mtime=override_mtime
        )
        for hdf5_file in files:
            try:
                eps = Episode.from_hdf5(
                    hdf5_file,
                    lookback=lookback,
                    state_channels=state_channels,
                    action_channels=action_channels,
                )
            except (KeyError, OSError) as e:
                LOG.warning(f"{e} in file {hdf5_file}")
                continue
            if eps.is_valid():
                episodes.append(eps)
        LOG.info(f"Loaded {len(episodes)} episodes (of {len(files)} files)")
        # Create the batch and set the loaded SART paths
        batch = cls(episodes, control=control, prioritization=prioritization)
        batch._loaded_sart_paths = set(files)
        return batch

    def append_from_hdf5(
        self,
        *dirpaths: str,
        state_channels: Optional[Sequence[str]] = None,
        action_channels: Optional[Sequence[str]] = None,
    ) -> None:
        """Append new SART files to the Batch instance.

        Given some directories, this will load the episodes of all files
        not already loaded previously and add them to the Batch.
        """
        episodes = []
        files = self._generate_file_list(dirpaths=dirpaths)
        new_files = set(files).difference(self._loaded_sart_paths)
        new_files = new_files.difference(self._ignored_sart_paths)
        if len(new_files) == 0:
            LOG.info("No new files to append.")
            return
        for hdf5_file in new_files:
            try:
                eps = Episode.from_hdf5(
                    hdf5_file,
                    lookback=self.lookback,
                    state_channels=state_channels,
                    action_channels=action_channels,
                )
            except (KeyError, OSError) as e:
                LOG.warning(f"{e} in file {hdf5_file}")
                continue
            if eps.is_valid():
                episodes.append(eps)
        LOG.info(f"Loaded {len(episodes)} new episodes (of {len(new_files)} new files)")
        self._loaded_sart_paths |= new_files
        self.append(episodes)

    @classmethod
    def multiple_from_key_hdf5(
        cls,
        *dirpaths: str,
        key_source: str,
        key: str,
        prioritization: Optional[str] = None,
        value: Optional[Union[str, Numeric]] = None,
        lookback: Optional[int] = None,
        state_channels: Optional[Sequence[str]] = None,
        action_channels: Optional[Sequence[str]] = None,
        control: Optional[Controller] = None,
        only_newest: Optional[int] = None,
    ) -> Dict[Union[Numeric, str], "Batch"]:
        """Load n-many Episodes from .h5, clustered by an action/meta key into a batch.

        See documentation for the same method in :class:`Episode` for more information.
        To append, use the specifically designated :meth:`append_multiple_from_key_hdf5`
        method, as the key/value pair is required for consistency within the batch.

        Args:
            key_source: where to find the key, either in the action addtl info or meta
            key: the key to split on
            prioritization: the type of prioritization desired, None if none
            value: optional value of the key to keep and drop others
            state_channels: only load these channels
            action_channels: only load these channels
            only_newest: if provided, will only load the newest n many SART files.
                         The ignored files will be ignored in any future appends.

        Returns:
            Dictionary of Batches of at least as many episodes as there are SART files,
            with more based on the occurrence of the key in the data. Keys of the dict
            correspond to values of the key in key source. If provided a value, the dict
            will have one key of that value.
        """
        if lookback is None:
            lookback = 1
            if hasattr(control, "lookback"):
                lookback = control.lookback  # type: ignore
        assert lookback is not None  # satisfy mypy
        num_eps: int = 0
        episodes: Dict[Union[Numeric, str], List["Episode"]] = defaultdict(list)
        files = cls._generate_file_list(dirpaths=dirpaths, only_newest=only_newest)
        for hdf5_file in files:
            try:
                eps = Episode.multiple_from_key_hdf5(
                    filepath=hdf5_file,
                    key_source=key_source,
                    key=key,
                    value=value,
                    lookback=lookback,
                    state_channels=state_channels,
                    action_channels=action_channels,
                )
            except (KeyError, OSError) as e:
                LOG.warning(f"{e} in file {hdf5_file}")
                continue
            for ep_key in eps.keys():
                if all(e.is_valid() for e in eps[ep_key]):
                    num_eps += len(eps[ep_key])
                    episodes[ep_key].extend(eps[ep_key])
        batches = dict()
        # Create the batch
        for ep_key in episodes.keys():
            batches[ep_key] = cls(
                episodes[ep_key], control=control, prioritization=prioritization
            )
            batches[ep_key]._loaded_sart_paths = set(files)
            # Ugly, but the _ignored_sart_paths are set on this class which is
            # then discarded. Here we set them to each individual batch instance.
            batches[ep_key]._ignored_sart_paths = cls._ignored_sart_paths
        LOG.info(
            f"Loaded {num_eps} episodes (of {len(files)} files), "
            f"split on {key_source}/{key} key."
        )
        return batches

    def append_multiple_from_key_hdf5(
        self,
        *dirpaths: str,
        key_source: str,
        key: str,
        value: Union[str, Numeric],
        state_channels: Optional[Sequence[str]] = None,
        action_channels: Optional[Sequence[str]] = None,
    ) -> None:
        """Append new SART files to the Batch instance based on a key/value pair.

        Given some directories, this will load all files and sort out the key/value
        related episodes, and append them to the current Batch.
        """
        episodes = []
        files = self._generate_file_list(dirpaths=dirpaths)
        new_files = set(files).difference(self._loaded_sart_paths)
        new_files = new_files.difference(self._ignored_sart_paths)
        if len(new_files) == 0:
            LOG.info("No new files to append.")
            return
        num_eps: int = 0
        for hdf5_file in new_files:
            try:
                eps = Episode.multiple_from_key_hdf5(
                    filepath=hdf5_file,
                    key_source=key_source,
                    key=key,
                    value=value,
                    lookback=self.lookback,
                    state_channels=state_channels,
                    action_channels=action_channels,
                )[value]
            except (KeyError, OSError) as e:
                LOG.warning(f"{e} in file {hdf5_file}")
                continue
            if all(e.is_valid() for e in eps):
                num_eps += len(eps)
                episodes.extend(eps)
        LOG.info(f"Loaded {num_eps} new episodes (of {len(new_files)} new files)")
        self._loaded_sart_paths |= new_files
        self.append(episodes)


if __name__ == "__main__":
    obs = np.arange(1, 101)[..., None]
    act = np.arange(1, 101)[..., None]
    term = np.array([False] * 99 + [True])[..., None]
    cost = obs
    eps = Episode(obs, act, term, cost, lookback=1)
    obs2 = np.arange(101, 201)[..., None]
    act2 = np.arange(101, 201)[..., None]
    term2 = np.array([False] * 99 + [True])[..., None]
    cost2 = obs
    eps2 = Episode(obs2, act2, term2, cost2, lookback=1)
    batch = Batch([eps, eps2])

    print("State1")
    print(obs.ravel(), len(obs))
    print("State2")
    print(obs2.ravel(), len(obs2))
    print("States inside")
    states_in = batch.set_minibatch_size(-1).sort().states[0].ravel()  # type:ignore
    print(states_in, len(states_in))
    print("Costs inside")
    costs_in = batch.set_minibatch_size(-1).sort().costs_terminals[0][0].ravel()
    print(costs_in, len(costs_in))
    print("Terminals inside")
    term_in = batch.set_minibatch_size(-1).sort().costs_terminals[0][1].ravel()
    print(term_in, len(term_in))
    print("States associated with Terminals inside")
    print(batch.set_minibatch_size(-1).sort().states[0].ravel()[term_in])  # type:ignore
    print("Computed costs")
    batch.compute_costs(lambda x: x.ravel())
    comp = batch.set_minibatch_size(-1).sort().costs_terminals[0][0].ravel()
    print(comp, len(comp))
