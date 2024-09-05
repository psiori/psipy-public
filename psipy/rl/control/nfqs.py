# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

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

import logging
import random
import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from psipy.core.io import MemoryZipFile
from psipy.core.np_utils import cache
from psipy.nn.layers import ExpandDims, Squeeze
from psipy.rl.control.controller import Controller
from psipy.rl.control.nfq import ObservationStack, enrich_errors
from psipy.rl.cycle_manager import CM
from psipy.rl.io.batch import Batch
from psipy.rl.plant import Action, State
from psipy.rl.preprocessing import StackNormalizer

__all__ = ["NFQs"]

LOG = logging.getLogger(__name__)


def make_state_action_pairs(
    states: Union[Dict[str, np.ndarray], np.ndarray], action_values: np.ndarray
) -> Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Creates the product of all states with all action values.

    - Tiles actions from ``[A1, A2, A3]`` to ``[A1, A2, A3, A1, A2, A3]``.
    - Repeats states from ``[S1, S2]`` to ``[S1, S1, S1, S2, S2, S2]``.

    Both states and actions are returned in shape ``(BATCH * N_ACT, ...)``.

    Example::

        >>> states = np.array([[1, 2, 3], [4, 5, 6]])
        >>> actions = np.array([-1, -2])
        >>> states, actions = make_state_action_pairs(states, actions)
        >>> states.tolist()
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]
        >>> actions.tolist()
        [[-1], [-2], [-1], [-2]]

        >>> states = dict(a=np.array([[1, 2], [4, 5]]), b=np.array([[3, 6], [7, 8]]))
        >>> actions = np.array([-1, -2])
        >>> sa = make_state_action_pairs(states, actions)
        >>> sa["a"].tolist()
        [[1, 2], [1, 2], [4, 5], [4, 5]]
        >>> sa["b"].tolist()
        [[3, 6], [3, 6], [7, 8], [7, 8]]
        >>> sa["actions"].tolist()
        [[-1], [-2], [-1], [-2]]

    """
    n_act = len(action_values)
    if isinstance(states, np.ndarray):
        # Input states single numpy array.
        actions = np.tile(action_values, len(states))[:, None]
        states = np.repeat(states, n_act, axis=0)
        states = (states, actions)
    else:  # if isinstance(states, dict)
        # Input states are dictionary of numpy arrays.
        n_states = len(next(iter(states.values())))
        states = {k: np.repeat(v, n_act, axis=0) for k, v in states.items()}
        assert "actions" not in states
        states["actions"] = np.tile(action_values, n_states)[:, None]
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
    ):
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

        # .all() uses cache!
        next_states = self.batch.nextstates.all()  # (BATCH, state, dim, lookback)
        next_states = sticky_state_action_pairs(
            next_states, self.action_values_normalized
        )

        # Note: The following does perform single-step inference! This might
        #       result in OOM errors when there is too much data.
        raw_qs = self.model(next_states).numpy()  # (BATCH * N_ACT, 1)
        qs = raw_qs.reshape(-1, len(self.action_values_normalized))  # (BATCH, N_ACT)
        if not doubleq:
            qs = qs.min(axis=1, keepdims=True)  # (BATCH, 1)
        q_target = costs + gamma * qs

        if not disable_terminals:
            # All terminals are bad terminals, as we are considering infinite
            # horizon regulator problems only. Terminal states are states which
            # the system may never reach, or it might break. Terminal states can
            # not be left anymore, therefore they have no future cost, but only
            # maximum immediate cost.
            q_target[terminals.ravel() == 1] = 1

        # Clamp down q values given their minimum value and clip values to
        # within sigmoid bounds to prevent saturation.
        q_target = np.clip((q_target - q_target.min()) + 0.05, 0.05, 0.95)

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
        lookback: int = 1,
        control_pairs: Optional[Tuple[Tuple[str, str], ...]] = None,
        num_repeat: int = 0,
        doubleq: bool = False,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "RMSProp",
        prioritized: bool = False,
        disable_terminals: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            action_values=action_values,
            lookback=lookback,
            control_pairs=control_pairs,
            num_repeat=num_repeat,
            doubleq=doubleq,
            disable_terminals=disable_terminals,
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
        if action_values is None:
            action_values = tuple(self.action_type.legal_values[0])
        self.action_values = np.asarray(action_values, dtype=float)

        self.epsilon = 0.0
        self._memory = ObservationStack((len(self.state_channels),), lookback=lookback)

        self.normalizer = StackNormalizer("meanstd")

        # As the action values are directly fed into the network, those as well
        # are normalized.
        self.action_normalizer = StackNormalizer("minmax")
        self.action_normalizer.fit(self.action_values[..., None])
        self.action_values_normalized = self.action_normalizer.transform(
            self.action_values[..., None]
        ).flatten()

        self.action_repeat_max = num_repeat
        self.action_repeat = 0
        self._prev_raw_act_and_meta: Optional[
            Tuple[np.ndarray, Dict[str, np.ndarray]]
        ] = None

        assert len(self.action_channels) == 1, "Only supports single actions."

        # Prepopulate `self.input_channels` and warmup model inference.
        try:
            inputs = self.preprocess_observations(self._memory.stack[None, ...])
            inputs = make_state_action_pairs(inputs, self.action_values_normalized)
            self._model(inputs)
        except Exception:
            LOG.warning("Model warmup failed, might still work as expected tho.")
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
            for channel, value in zip(self.action_channels, values):
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
        if random.random() < self.epsilon or self.action_repeat > 0:
            if self.action_repeat == 0:
                action_indices = np.random.randint(
                    low=0, high=len(self.action_values), size=(stacks.shape[0], 1)
                )
                actions = self.action_values[action_indices]  # shape (N, 1)
                meta = dict(index=action_indices, nodoe=actions)
                # Randomly alter how long actions are held
                self.action_repeat = self.action_repeat_max
                if self.action_repeat_max > 1:
                    self.action_repeat = random.randint(1, self.action_repeat_max)
                self._prev_raw_act_and_meta = (actions, meta)
            else:  # repeat_count > 0
                actions, meta = cast(Tuple, self._prev_raw_act_and_meta)
                self.action_repeat -= 1
        else:
            stacks = self.preprocess_observations(stacks)
            stacks = make_state_action_pairs(stacks, self.action_values_normalized)
            with CM["get-actions-predict"]:
                q_values = self._model(stacks).numpy()
            action_indices = argmin_q(q_values, len(self.action_values))
            action_indices = action_indices.astype(np.int32).ravel()
            actions = self.action_values[action_indices, None]  # shape (N, 1)
            meta = dict(index=action_indices, nodoe=actions)
            self.action_repeat = 0
        if self.idoe:
            actions = self.doe_transform(actions)
        return actions, meta

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
        self.action_repeat = 0
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

        train_model = tf.keras.Model(self._model.inputs, self._model.outputs)
        train_model.compile(
            optimizer=self._optimizer,
            loss="mean_squared_error",
            metrics=[min_q, avg_q, max_q],
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
    def from_config(cls, config):
        raise NotImplementedError("Cannot initialize from config.")

    def _save(self, zipfile: MemoryZipFile) -> MemoryZipFile:
        zipfile.add("config.json", self.get_config())
        zipfile.add("model.h5", self._model)
        zipfile.add_json(
            "Action.json",
            dict(
                class_name=self.action_type.__name__,
                class_module=self.action_type.__module__,
            ),
        )
        self.normalizer.save(zipfile)
        return zipfile

    @classmethod
    def _load(
        cls, zipfile: MemoryZipFile, custom_objects: Optional[List[Type[object]]] = None
    ):
        if custom_objects is None:
            custom_objects = [ExpandDims, Squeeze]
        config = zipfile.get("config.json")
        model = zipfile.get_keras("model.h5", custom_objects)
        action_meta = zipfile.get_json("Action.json")
        assert isinstance(action_meta, dict)
        action_type = cls.load_action_type(action_meta, custom_objects)
        obj = cls(model=model, action=action_type, **config)
        obj.normalizer = StackNormalizer.load(zipfile)
        return obj
