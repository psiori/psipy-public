# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
#
# Author: Alexander Höreth (2020)
#


"""Single Output Neural Fitted Q-Iteration
==========================================

Implementation of a Neural Fitted Q controller, that includes both state and
action as input and returns a single q value as output. This stands in contrast
to :mod:`~psipy.rl.control.nfq`, where the network receives only the state as
input and returns one q value per discrete action as output.

A central advantage of using actions as part of the input is that continuous
values of actions can be used and discretization of action values can be defined
at runtime. Additionally one does not need to keep track of action indices (in
contrast to :mod:`~psipy.rl.control.nfq`) and therefore is more flexible of
applying the algorithm to historic and/or off-policy data.

.. autosummary::

    NFQs

"""

import itertools
import logging
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl
from tensorflow.keras.utils import Sequence

from psipy.core.io import MemoryZipFile
from psipy.core.np_utils import cache
from psipy.nn.layers import ExpandDims, Squeeze
from psipy.rl.controllers.nfq import ObservationStack, enrich_errors
from psipy.rl.core.controller import Controller
from psipy.rl.core.cycle_manager import CM
from psipy.rl.core.plant import Action, State
from psipy.rl.io.batch import Batch
from psipy.rl.preprocessing import StackNormalizer

__all__ = ["NFQs"]

LOG = logging.getLogger(__name__)


def generate_multi_dimensional_action_combinations(
    legal_values: Tuple[Tuple, ...],
) -> np.ndarray:
    """Generate all possible combinations of multi-dimensional discrete actions and also the combinations of indices of these combined actions.

    Args:
        legal_values: Tuple of tuples, where each inner tuple contains the
                      legal values for one action dimension.

    Returns:
        Tuple of shape (array(N_ACTIONS, N_DIMENSIONS), array(N_ACTIONS,
        N_DIMENSIONS)) containing all possible action combinations in the first
        element and the corresponding indices of these combinations in the
        second element.

    Example:
        >>> legal_values = ((-1, 0, 1), (-1, 0, 1))  # 2D actions with 3 values each
        >>> generate_multi_dimensional_action_combinations(legal_values)
        (array([[-1., -1.],
               [-1.,  0.],
               [-1.,  1.],
               [ 0., -1.],
               [ 0.,  0.],
               [ 0.,  1.],
               [ 1., -1.],
               [ 1.,  0.],
               [ 1.,  1.]]), array([[0, 0],
               [0, 1],
               [0, 2],
               [1, 0],
               [1, 1],
               [1, 2],
               [2, 0],
               [2, 1],
               [2, 2]]))
    """
    combinations = list(itertools.product(*legal_values))
    index_combinations = list(
        itertools.product(*[range(len(dim)) for dim in legal_values])
    )
    return (
        np.array(combinations, dtype=float),
        np.array(index_combinations, dtype=int),
    )


def make_state_action_pairs(
    states: Union[Dict[str, np.ndarray], np.ndarray], action_values: np.ndarray
) -> Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the product of all states with all action values.

    - Tiles actions from ``[A1, A2, A3]`` to ``[A1, A2, A3, A1, A2, A3]``.
    - Repeats states from ``[S1, S2]`` to ``[S1, S1, S1, S2, S2, S2]``.

    Both states and actions are returned in shape ``(BATCH * N_ACT, ...)``.

    Example::

        >>> states = np.array([[1, 2, 3], [4, 5, 6]])
        >>> actions = np.array([[-1, -2], [-3, -4]])
        >>> states, actions = make_state_action_pairs(states, actions)
        >>> states.tolist()
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]
        >>> actions.tolist()
        [[-1, -2], [-3, -4], [-1, -2], [-3, -4]]

        >>> states = dict(a=np.array([[1, 2], [4, 5]]), b=np.array([[3, 6], [7, 8]]))
        >>> actions = np.array([[-1, -2], [-3, -4]])
        >>> sa = make_state_action_pairs(states, actions)
        >>> sa["a"].tolist()
        [[1, 2], [1, 2], [4, 5], [4, 5]]
        >>> sa["b"].tolist()
        [[3, 6], [3, 6], [7, 8], [7, 8]]
        >>> sa["actions"].tolist()
        [[-1, -2], [-3, -4], [-1, -2], [-3, -4]]

    """
    n_act = len(action_values)

    if isinstance(states, np.ndarray):
        # Input states are a single numpy array.
        actions = np.tile(action_values, (len(states), 1))
        states = np.repeat(states, n_act, axis=0)
        states = (states, actions)
    else:  # if isinstance(states, dict)
        # Input states are dictionary of numpy arrays.
        n_states = len(next(iter(states.values())))
        states = {k: np.repeat(v, n_act, axis=0) for k, v in states.items()}
        assert "actions" not in states
        states["actions"] = np.tile(action_values, (n_states, 1))

    LOG.debug("STATE_ACTION_PAIRS", states)
    return states


def argmin_q(predictions: np.ndarray, n_actions):
    predictions = predictions.reshape(-1, n_actions)  # (BATCH, N_ACT)
    return predictions.argmin(axis=1)  # (BATCH,)


@cache(1)
def sticky_state_action_pairs(next_states, action_values):
    """Create state action pairs cached to the GPU."""
    next_states = make_state_action_pairs(next_states, action_values)

    if isinstance(next_states, dict):
        with tf.device(tf.test.gpu_device_name() or "CPU:0"):
            return {key: tf.constant(val) for key, val in next_states.items()}
    with tf.device(tf.test.gpu_device_name() or "CPU:0"):
        return tuple(tf.constant(arg) for arg in next_states)


class PickyBatch(Sequence):
    """A batch wrapper which calculates targets for NFQs.

    Args:
        batch: The batch instance with data to be trained on
        model: The train model for NFQs
        action_values_normalized: all possible action values in proper order, normalized
        gamma: Discount parameter
        doubleq: if double q targets should be computed
        prioritized: If the batch is sampled according to transition priority or not.
        disable_terminals: true if terminals should NOT be clamped to 1
    """

    def __init__(
        self,
        batch: Batch,
        model: tf.keras.Model,
        action_values_normalized: Tuple[Union[int, float], ...],
        gamma: float = 0.99,
        doubleq: bool = False,
        prioritized: bool = False,
        disable_terminals: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch = batch
        self.action_values_normalized = action_values_normalized
        self.model = model
        self.prioritized = prioritized
        self.doubleq = doubleq

        # Prepare the Batch for doing some computations on all its sorted values.
        minibatch_size = self.batch.minibatch_size
        self.batch.set_minibatch_size(-1).sort()

        # This currently does not use the Batch cache, on purpose!
        costs, terminals = self.batch.costs_terminals[0]
        if len(costs.shape) == 1:
            costs = costs[:, None]
        assert len(costs.shape) == 2, "Be aware of broadcasting!"

        # states, actions = self.batch.states_actions[0]

        # .all() uses cache!
        next_states = self.batch.nextstates.all()  # (BATCH, state, dim, lookback)
        next_states = sticky_state_action_pairs(
            next_states, self.action_values_normalized
        )

        # print (next_states)

        # Note: The following does perform single-step inference! This might
        #       result in OOM errors when there is too much data.
        raw_qs = self.model(next_states).numpy()  # (BATCH * N_ACT, 1)
        qs = raw_qs.reshape(-1, len(self.action_values_normalized))  # (BATCH, N_ACT)

        if not doubleq:
            qs = qs.min(axis=1, keepdims=True)  # (BATCH, 1)
        q_target = costs + gamma * qs * (1 - terminals)

        if not disable_terminals:
            # All terminals are bad terminals, as we are considering infinite
            # horizon regulator problems only. Terminal states are states which
            # the system may never reach, or it might break. Terminal states can
            # not be left anymore, therefore they have no future cost, but only
            # maximum immediate cost.
            q_target[terminals.ravel() == 1] = 1

        # print(f"\n\n\n>>>>>>>>>>\n\nqtargets n: { len(q_target) } max: {q_target.max()} min: {q_target.min()}")

        # for s, a, t, c, qt in zip(states, actions, terminals, costs, q_target):
        #    if t:
        #        print (">> TERMINAL transition ({}, {}, {}) with qtarget: {}".format(s, a, c, qt))
        #
        # print ("\n>>>>>>>>>>>> {} TERMINALS\n\n".format(np.sum(terminals)))

        # Clamp down q values given their minimum value and clip values to
        # within sigmoid bounds to prevent saturation.
        # q_target = np.clip((q_target - q_target.min()) + 0.05, 0.05, 0.95)

        # qs[costs.ravel() == 0] = 0

        # Store the q target in the batch; it is altered below if
        # using a prioritized double NFQ in order to reduce dims
        self.batch.set_targets(q_target)

        if self.prioritized:
            start = time.time()
            states, actions = self.batch.states_actions[0]
            states = make_state_action_pairs(states, self.action_values_normalized)
            q_pred = self.model(states).numpy()
            q_pred = q_pred.reshape(-1, len(self.action_values_normalized))
            action_idx = np.array(
                [np.where(self.action_values_normalized == a)[0] for a in actions]
            )
            q_pred = np.array([q[i] for q, i in zip(q_pred, action_idx)])
            # If using double q, use the current ("online") network's action
            # when calculating the TD error (PER Alg. 1)
            if doubleq:
                # Here the online network is used to predict what action index should
                # be used in the next state. The targets are then indexed by these
                # values. Note that these targets are static, i.e. they are only
                # computed at batch creation time. The targets used during training
                # will be different since the actual online network is being used
                # mid-training. This could be altered if the batch is allowed to
                # prioritize and resample during fitting.
                action_idx = argmin_q(raw_qs, len(self.action_values_normalized))
                q_target = np.array([q[i] for q, i in zip(q_target, action_idx)])
                if len(q_target.shape) == 1:
                    q_target = q_target[..., None]
            else:  # normal NFQ
                q_pred = q_pred.min(axis=1, keepdims=True)
            delta = q_target - q_pred
            assert delta.shape[1] == 1, "Improper broadcasting!"
            LOG.debug(
                f"Time taken for prioritization: {round((time.time() - start), 4)}s"
            )
            self.batch.set_delta(delta)

        # Reset the Batch to its original config.
        self.batch.set_minibatch_size(minibatch_size).shuffle()

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        sa, t, w = self.batch.statesactions_targets[idx]

        # The targets are either shaped (N_SAMPLES, 1) or (N_SAMPLES, N_ACT).
        # The latter is the case in a Double Q setting, where the q value
        # is not picked by the minimum but by the online network's argmin q
        # value, aka the online network's action.
        if self.doubleq:
            nextstates = self.batch.nextstates[idx]  # (BATCH, state, dim, lookback)
            inputs = make_state_action_pairs(nextstates, self.action_values_normalized)
            qs = self.model(inputs).numpy()  # (BATCH * N_ACT, 1)
            action_idx = argmin_q(qs, len(self.action_values_normalized))
            t = t[np.arange(len(t)), action_idx]
        if self.prioritized:
            return sa, t, w

        return sa, t


class NFQs(Controller):
    """Neural Fitted Q-Iteration with (State, Action) input and a single Q output.

    Implementation of a Neural Fitted Q controller, which is based on a neural
    model with states and actions as input and a single q value as output.

    Args:
        model: Neural Q-Function approximator.
        action_values: Neural Q-Function approximator output values.
        lookback: Number of observation timesteps to stack into a single input.
        control_pairs: ``(SP, PV)`` tuples of channel names to each merge into a
                       single control deviations channel in the neural network's
                       input.
        doubleq: Whether to employ "Double Q Learning".
        num_repeat: Each sampled action is repeated for 'num_repeat' steps before a new one is sampled. Set to 1 to disable.
        optimizer: The network optimizer to use, as a string or Keras optimizer.
        prioritized: Whether or not the incoming minibatches are prioritized;
             if so, the loss will be weighted via replay importance
             sampling.
        disable_terminals: Whether to disable transition ending high terminal
                           state cost.
        **kwargs: Further keyword arguments for :class:`Controller`.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        action_values: Optional[Tuple[Union[int, float], ...]] = None,
        action_indices: Optional[Tuple[int, ...]] = None,
        lookback: int = 1,
        control_pairs: Optional[Tuple[Tuple[str, str], ...]] = None,
        num_repeat: int = 0,
        doubleq: bool = False,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "Adam",
        prioritized: bool = False,
        disable_terminals: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            action_values=action_values,
            lookback=lookback,
            control_pairs=control_pairs,
            doubleq=doubleq,
            disable_terminals=disable_terminals,
            num_repeat=num_repeat,
            **kwargs,
        )
        self.disable_terminals = disable_terminals
        self.doubleq = doubleq
        self.prioritized = prioritized
        self.idoe = self.action_type.dtype == "continuous"
        if self.idoe:
            LOG.info("Running NFQ in I-DOE mode.")
        self._optimizer = optimizer
        self._model = model
        self.control_pairs = control_pairs

        LOG.debug("ACTION VALUES HANDED TO INIT", action_values)

        if action_indices is not None:
            self._action_indices = np.asarray(action_indices, dtype=int)
        else:
            self._action_indices = None

        if action_values is not None:
            action_values = np.asarray(action_values, dtype=float)

            if len(action_values.shape) == 1:
                action_values = action_values[
                    ..., None
                ]  # we make every potential action a vector, even if it may be 1-dimensional. This is to unify and ease further processing between 1-dimensional and multi-dimensional actions, and to, nevertheless, accept whatever the user gives as argument.
                if self._action_indices is None:
                    self._action_indices = np.arange(
                        len(action_values)
                    )[
                        ..., None
                    ]  # legacy method to allow providing 1-d action values via the constructor. Indices for n-d actions provided via the constructor are not generated automatically, as the order is unclear.

            elif action_indices is None:
                LOG.warning(
                    "Action values provided via the constructor, but action indices are not. Ignoring action indices, data will not be usable by DQN - like methods."
                )

        else:
            if len(self.action_channels) == 1:
                action_values = self.action_type.legal_values
                self._action_indices = np.arange(len(action_values))
            else:
                action_values, action_indices = (
                    generate_multi_dimensional_action_combinations(
                        self.action_type.legal_values
                    )
                )
                self._action_indices = action_indices

        self._action_values = np.asarray(action_values, dtype=float)

        LOG.info("USED ACTION VALUES", self._action_values)
        LOG.info("USED ACTION INDICES", self._action_indices)
        print("USED ACTION VALUES", self._action_values)
        print("USED ACTION INDICES", self._action_indices)
        print("USED ACTION VALUES SHAPE", self._action_values.shape)

        self.epsilon = 0.0
        self._memory = ObservationStack((len(self.state_channels),), lookback=lookback)

        self.normalizer = StackNormalizer("meanstd")

        # As the action values are directly fed into the network, those as well
        # need to be normalized.
        action_normalizer = StackNormalizer("minmax")
        action_normalizer.fit(self._action_values)
        self.action_normalizer = action_normalizer  # this will automatically repopulate the self.action_values_normalized

        # Maximum number of times an action can be repeated
        self.action_repeat_max = num_repeat
        # How many times this action has been repeated. We count the original action as the first repeat.
        self.action_repeat_count = 0
        self._prev_raw_act_and_meta: Optional[
            Tuple[np.ndarray, Dict[str, np.ndarray]]
        ] = None

        # assert len(self.action_channels) == 1, "Only supports single actions."

        # Prepopulate `self.input_channels` and warmup model inference.
        try:
            inputs = self.preprocess_observations(self._memory.stack[None, ...])
            inputs = make_state_action_pairs(inputs, self.action_values_normalized)
            self._model(inputs)
        except Exception:
            LOG.warning("Model warmup failed, might still work as expected tho.")
            print("Model warmup failed, might still work as expected tho.")
            pass

    def notify_episode_starts(self) -> None:
        if self.idoe:
            self.idoe_state = np.zeros((1, len(self.action_channels)))

    def get_action(self, state: State) -> Action:
        observation = state.as_array(*self.state_channels)
        stack = self._memory.append(observation).stack

        action, meta = self.get_actions(stack[None, ...])
        assert action.shape[0] == 1
        action = action.ravel()

        # Splitting the meta data vectors into the individual action channels.
        individual_meta = dict()
        for key, values in meta.items():
            for channel, value in zip(self.action_channels, values[0]):
                individual_meta[f"{channel}_{key}"] = value.item()

        mapping = dict(zip(self.action_channels, action))
        return self.action_type(mapping, additional_data=individual_meta)

    def get_actions(
        self, stacks: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Gets the actions for given stacks of observations.

        Returns an additional metadata dict containing the NFQ action indices
        and the original action values before the postprocessing step.

        Args:
            stack: ``(N, CHANNELS, LOOKBACK)`` shaped stacks of observations.
        """
        if (
            self.action_repeat_count > 0
            and self.action_repeat_count < self.action_repeat_max
        ):  # Repeating an action
            actions, meta = cast(Tuple, self._prev_raw_act_and_meta)
            self.action_repeat_count += 1
        elif random.random() < self.epsilon:  # Random exploration
            action_indices = np.random.randint(
                low=0, high=len(self.action_values), size=(stacks.shape[0], 1)
            ).ravel()  # TODO (AH/SL): not 100% sure why this is needed after switch to multi-dimensional actions.
            actions = self.action_values[action_indices]  # shape (N, ACT_DIM)
            if self._action_indices is not None:
                meta = dict(
                    index=self._action_indices[action_indices],
                    nodoe=actions,
                    is_random=np.ones((stacks.shape[0], actions.shape[1]), dtype=bool),
                )
            else:
                meta = dict(
                    nodoe=actions,
                    is_random=np.ones((stacks.shape[0], actions.shape[1]), dtype=bool),
                )

            # TODO: Not sure why this is wanted
            # Randomly alter how long actions are held
            if self.action_repeat_max >= 1:
                self.action_repeat_count = random.randint(1, self.action_repeat_max)
            else:
                self.action_repeat_count = 0
            self._prev_raw_act_and_meta = (actions, meta)
        else:  # Choose best action
            stacks = self.preprocess_observations(stacks)
            stacks_batch_size = stacks.shape[0]
            stacks = make_state_action_pairs(stacks, self.action_values_normalized)

            with CM["get-actions-predict"]:
                q_values = self._model(stacks).numpy()
            action_indices = argmin_q(q_values, len(self.action_values))
            action_indices = action_indices.astype(np.int32).ravel()
            actions = self.action_values[action_indices]  # shape (N, ACT_DIM)
            if self._action_indices is not None:
                meta = dict(
                    index=self._action_indices[action_indices],
                    nodoe=actions,
                    is_random=np.zeros(
                        (stacks_batch_size, actions.shape[1]), dtype=bool
                    ),
                )
            else:
                meta = dict(
                    nodoe=actions,
                    is_random=np.zeros(
                        (stacks_batch_size, actions.shape[1]), dtype=bool
                    ),
                )
            self.action_repeat_count = 1
            self._prev_raw_act_and_meta = (actions, meta)
        if self.idoe:
            actions = self.doe_transform(actions)
        return actions, meta

    @property
    def action_normalizer(self):
        return self._action_normalizer

    @action_normalizer.setter
    def action_normalizer(self, action_normalizer: StackNormalizer):
        self._action_normalizer = action_normalizer
        action_values = self.action_values
        self.action_values_normalized = self.action_normalizer.transform(action_values)

    @property
    def action_values(self):
        return self._action_values

    @action_values.setter
    def action_values(self, action_values: np.ndarray):
        self._action_values = np.asarray(action_values, dtype=float)
        self._config["action_values"] = (
            action_values  # not converted, like in constructor
        )
        print("CONFIG AFTER NEW ACTION VALUES", self._config)

    def preprocess_observations(self, stacks: np.ndarray) -> np.ndarray:
        """Preprocesses observation stacks before those are passed to the network.

        Employed in both the local :meth:`get_actions` method as well as in the
        :class:`~Batch` during training.

        Args:
          stacks: Observation stacks of shape ``(BATCH, CHANNELS, LOOKBACK)``.
        """
        channels = self.state_channels
        if self.control_pairs is not None:
            stacks, channels = enrich_errors(stacks, channels, *self.control_pairs)
        self.input_channels = channels
        stacks = self.normalizer.transform(stacks)
        return stacks

    def doe_transform(self, actions: np.ndarray) -> np.ndarray:
        """Transforms actions according to the Dynamic Output Element (DOE).

        DOEs are action accumulators which do some transformation on
        actions coming from a :class:`~Controller` to create the final action
        that is sent to the plant.  This transformation can be technically
        anything, from integration to low pass filtering.

        The default DOE is the integrator DOE (IDOE), which adds up previous
        actions. To change the DOE method, subclass the controller.
        """
        if actions.shape[1] > 1:
            raise NotImplementedError("DOE is stateful and cannot be used on batches.")
        self.idoe_state += actions

        # Clip because the idoe state might be outside the legal values.
        for i, (lower, upper) in enumerate(self.action_type.legal_values):
            self.idoe_state[..., i] = np.clip(self.idoe_state[..., i], lower, upper)

        return self.idoe_state

    def doe_inverse(self, actions: np.ndarray) -> np.ndarray:
        """Used from :class:`~Batch` to get the original NFQ actions.

        For the integrator DOE (IDOE), the first action is unchanged as
        the state of the IDOE was nothing at the start of an episode, and
        thus the original action is also the first state, while the rest
        need to be differenced to get the IDOE controller deltas.
        """
        return np.array([actions[0], *np.diff(actions, axis=0)])

    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop."""
        self._memory.clear()
        self.action_repeat_count = 0
        self._prev_raw_act_and_meta = None

    @property
    def train_model(self):
        self.maybe_make_model()
        return self._train_model

    def maybe_make_model(self) -> None:
        if hasattr(self, "_train_model"):
            return

        def min_q(y_true, y_preds):
            return tf.reduce_min(y_preds)

        def avg_q(y_true, y_preds):
            return tf.reduce_mean(y_preds)

        def max_q(y_true, y_preds):
            return tf.reduce_max(y_preds)

        def avg_qdelta(y_true, y_preds):
            return tf.reduce_mean(
                tf.abs(y_preds - y_true)
            )  # mean absolute error, easier to interpret than loss/MSE

        def median_qdelta(y_true, y_preds):
            return tfp.stats.percentile(tf.abs(y_preds - y_true), 50)

        train_model = tf.keras.Model(self._model.inputs, self._model.outputs)
        train_model.compile(
            optimizer=self._optimizer,
            loss="MSE",
            metrics=[avg_qdelta, median_qdelta, min_q, avg_q, max_q],
        )

        # Workaround for `AssertionError`s seen during multi-threaded fit.
        # NOTE: No longer needed with tf2.2+?
        # train_model._make_train_function()

        self._train_model = train_model

    def fit(
        self,
        batch: Batch,
        costfunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        iterations: int = 1,
        epochs: int = 1,
        minibatch_size: int = -1,
        gamma: float = 0.99,
        callbacks: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Train the underlying model.

        Args:
            batch:
            costfunc: A function accepting a numpy array of observations to compute a
                      single cost for each.
            **kwargs: arguments going to keras.model.fit()
                      Example: verbose=0 to suppress keras output
        """
        if callbacks is None:
            callbacks = []

        self.increment_generation()

        if costfunc is not None:
            batch.compute_costs(costfunc)

        for iteration in range(1, iterations + 1):
            LOG.info(f"Fit iteration: {iteration}")
            batch.set_minibatch_size(minibatch_size).shuffle()
            self.train_model.fit(
                PickyBatch(
                    batch,
                    self._model,
                    self.action_values_normalized,
                    gamma=gamma,
                    doubleq=self.doubleq,
                    prioritized=self.prioritized,
                    disable_terminals=self.disable_terminals,
                ),
                epochs=epochs,
                callbacks=callbacks,
                **kwargs,
                # initial_epoch=(iteration - 1) * epochs,
            )

    def fit_normalizer(
        self, observations: np.ndarray, method: Optional[str] = None
    ) -> None:
        """Fit the internal StackNormalizer."""
        if method and method != self.normalizer.method:
            self.normalizer = StackNormalizer(method)
        self.normalizer.fit(observations)

    @classmethod
    def default_model(
        cls,
        state_dim: int,
        action_dim: int = 1,
        lookback: int = 1,
        hidden_dim: int = 256,
        feature_dim: int = 100,
    ) -> tf.keras.Model:
        """Create a default model.

        Args:
            state_dim: Dimension of the state.
            action_dim: Dimension of the action.
            lookback: Number of history steps from the state to feed into the actor.
        """
        inp = tfkl.Input((state_dim, lookback), name="states")
        act = tfkl.Input((action_dim,), name="actions")
        net = tfkl.Flatten()(inp)
        net = tfkl.concatenate([act, net])
        net = tfkl.Dense(hidden_dim, activation="relu")(net)
        net = tfkl.Dense(hidden_dim, activation="relu")(net)
        net = tfkl.Dense(feature_dim, activation="tanh")(net)
        net = tfkl.Dense(1, activation="sigmoid")(net)
        model = tf.keras.Model([inp, act], net, name="nfqsmodel")
        return model

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError("Cannot initialize from config.")

    def reload(self, file_path: str, load_all_settings: bool = False) -> None:
        # Load the saved file
        zipfile = MemoryZipFile(file_path)

        top_level = [d for d in zipfile.ls(include_directories=True) if d.endswith("/")]
        if len(top_level) == 0:
            raise ValueError(
                "No top-level directory found, files contained in zip: "
                f"{zipfile.ls(include_directories=True, recursive=True)}"
            )
        if len(top_level) > 1:
            raise ValueError(
                "ZIP ambiguous, multiple top-level directories: "
                f"{zipfile.ls(include_directories=True)}"
            )
        zipfile.cd(top_level[0])
        zipfile.cd("NFQs")

        self._reload(zipfile, load_all_settings)

    def _reload(self, zipfile: MemoryZipFile, load_all_settings: bool = False) -> None:
        self._model = zipfile.get_keras("model.keras", support_legacy=True)
        self.normalizer = StackNormalizer.load(zipfile)

        if load_all_settings:
            pass

    def _save(self, zipfile: MemoryZipFile) -> MemoryZipFile:
        zipfile.add("config.json", self.get_config())
        zipfile.add("model.keras", self._model)
        zipfile.add_json(
            "Action.json",
            dict(
                class_name=self.action_type.__name__,
                class_module=self.action_type.__module__,
            ),
        )
        self.normalizer.save(zipfile)
        self.action_normalizer.save(zipfile, "action_normalizer")
        return zipfile

    @classmethod
    def _load(
        cls, zipfile: MemoryZipFile, custom_objects: Optional[List[Type[object]]] = None
    ):
        if custom_objects is None:
            custom_objects = [ExpandDims, Squeeze]
        config = zipfile.get("config.json")
        model = zipfile.get_keras("model.keras", custom_objects)
        action_meta = zipfile.get_json("Action.json")
        print("NFQs._load action meta:", action_meta)
        print("NFQs._load config:", config)
        assert isinstance(action_meta, dict)
        action_type = cls.load_action_type(action_meta, custom_objects)
        obj = cls(model=model, action=action_type, **config)
        obj.normalizer = StackNormalizer.load(zipfile)
        try:
            obj.action_normalizer = StackNormalizer.load(zipfile, "action_normalizer")
            LOG.info(
                "NFQs._load loaded action_normalizer with configuration:",
                obj.action_normalizer.get_config(),
            )
        except Exception as e:
            LOG.warning(
                "POSSIBLY BROKEN MODEL: Failed to load action normalizer. This file is likely from an old NFQs model format. Will use default normalization instead. If you changed the action values after training, especially adding larger actions, this will likely lead to errors and an undefined behavior of the controller."
            )
            LOG.warning(e)

        return obj
