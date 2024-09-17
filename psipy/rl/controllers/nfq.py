import os
import logging
import random
import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from psipy.core.io import MemoryZipFile
from psipy.nn.layers import ExpandDims, Squeeze
from psipy.rl.core.controller import Controller
from psipy.rl.controllers.keras_utils import reset_model
from psipy.rl.controllers.layers import ArgMinLayer, ExtractIndexLayer, MinLayer
from psipy.rl.core.cycle_manager import CM
from psipy.rl.io.batch import Batch
from psipy.rl.core.plant import Action, State
from psipy.rl.preprocessing import StackNormalizer

__all__ = ["NFQ", "ObservationStack", "tanh2"]

LOG = logging.getLogger(__name__)

NET_MAX = 1
NET_MIN = 0

import pandas as pd

CSV = pd.DataFrame()


class DQBatch(Sequence):
    """A batch wrapper which calculates targets for NFQ.

    Args:
        batch: The batch instance with data to be trained on
        model: The min q model for normal NFQ, and the raw q value model for double NFQ
        act_model: The act model, which selects the index of the best action
        train_model: The model to get a q value for a specific action (not necessarily
                     the best). Only used in the double NFQ case.
        A: the slope of the q normalizer
        B: the intercept of the q normalizer
        gamma: Discount parameter
        doubleq: if double q targets should be computed
        prioritized: If the batch is sampled according to transition priority or not.
        scale: Whether or not to linearly scale the q targets into the range [0,1]
               Only do this when training on a single batch, not growing!
    """

    def __init__(
        self,
        batch: Batch,
        model: tf.keras.Model,
        act_model: tf.keras.Model,
        train_model: tf.keras.Model,
        A: float,
        B: float,
        gamma: float = 0.99,
        doubleq: bool = False,
        prioritized: bool = False,
        scale: bool = False,
    ):
        global CSV
        self.batch = batch
        self.act_model = act_model
        self.prioritized = prioritized
        self.doubleq = doubleq

        # Prepare the Batch for doing some computations on all its sorted values.
        minibatch_size = batch.minibatch_size
        self.batch.set_minibatch_size(-1).sort()

        # Depending on whether double q is enabled or not, the model returns a
        # set of q values per state or just a single (minimum) one. Given that
        # output, the target q values are computed in matrix or vector form.
        costs, terminals = self.batch.costs_terminals[0]
        if len(costs.shape) == 1:
            costs = costs[:, None]
        assert len(costs.shape) == 2, "Be aware of broadcasting!"

        # Both of the following are shaped (N_SAMPLES, 1) or (N_SAMPLES, N_ACT).
        # This uses batch's internal cache, such that multiple iterations on the
        # *same* dataset will not require reloading.
        nextstates = self.batch.nextstates.all()
        qs = model(nextstates).numpy()

        if scale:
            # Invert the scaling of the network's Q values in order to calculate
            # the proper TD update.
            qs = self.scale(qs, A, B, invert=True)

        q_target = costs + gamma * qs * (1 - terminals)

        #print(">>> immediate cost", costs)
        print(">>> terminal", terminals)
        #print(">>> q_target before scaling: ", q_target)


        # CSV = CSV.append(pd.DataFrame({"A":[self.A], "B":[self.B], "outmin":[out_min], "outmax":[out_max]}))

        # All terminals are bad terminals, as we are considering infinite
        # horizon regulator problems only. Terminal states are states which
        # the system may never reach, or it might break. Terminal states can
        # not be left anymore, therefore they have no future cost, but only
        # maximum immediate cost.
        q_target[terminals.ravel() == 1] = 1

        print(">>> qtargets after setting terminals", q_target)


        if scale:
            # Update the scaling parameters based on the max and min output of the network, and
            # the max and min possible outputs of a sigmoid (last layer's activation).
            out_min = np.min(q_target)
            out_max = np.max(q_target)
            self.A, self.B = self.update_scaling_parameters(out_min, out_max)

            # Scale the network's Q values in order to remove the effect of the magnitude
            # of costs incurred.
            q_target = self.scale(q_target, self.A, self.B)


        # Clamp down q values given their minimum value and clip values to
        # within sigmoid bounds to prevent saturation (Hafner).
        q_target = np.clip((q_target - q_target.min()) + 0.005, 0.005, 0.995)

        print(">>> qtargets after scaling and setting terminals and clipping", q_target)


        # qs[costs.ravel() == 0] = 0.05

        # Store the q target in the batch; it is altered below if
        # using a prioritized double NFQ in order to reduce dims
        self.batch.set_targets(q_target)

        if self.prioritized:
            start = time.time()
            # If using double q, use the current ("online") network's action
            # when calculating the TD error (PER Alg. 1)
            if self.doubleq:
                states, actions = self.batch.states_actions[0]
                q_pred = train_model([states, actions]).numpy()
                # Here the online network is used to predict what action index should
                # be used in the next state. The targets are then indexed by these
                # values. Note that these targets are static, i.e. they are only
                # computed at batch creation time. The targets used during training
                # will be different since the actual online network is being used
                # mid-training. This could be altered if the batch is allowed to
                # prioritize and resample during fitting.
                online_next_actions = act_model(nextstates).numpy().astype(np.int32)
                # Because q_target[:, online_next_actions] doesn't work...
                q_target = np.array(
                    [q[i] for q, i in zip(q_target, online_next_actions)]
                )
            else:  # normal NFQ
                states = self.batch.states[0]
                q_pred = model(states).numpy()
            # Q_pred does not need to be unscaled; it is compared to q_target which is
            # already scaled and the network is set up to predict scaled q values.
            # Also, q_pred uses the old A, B parameters, which is ok since during
            # training the predicted qs will also be using the same old parameters and
            # fitted to the new A, B. This prioritization behavior mimics that training
            # step here.
            delta = q_target - q_pred
            LOG.debug(
                f"Time taken for prioritization: {round((time.time() - start), 4)}s"
            )
            self.batch.set_delta(delta)

        # Reset the Batch to its original config.
        self.batch.set_minibatch_size(minibatch_size).shuffle()

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx: int) -> Union[Tuple, np.ndarray]:
        sa, t, w = self.batch.statesactions_targets[idx]

        # The targets are either shaped (N_SAMPLES, 1) or (N_SAMPLES, N_ACT).
        # The latter is the case in a Double Q setting, where the q value
        # is not picked by the minimum but by the online network's argmin q
        # value, aka the online network's action.
        if self.doubleq:
            s_ = self.batch.nextstates[idx]
            argmin_a = self.act_model(s_).numpy().ravel()
            t = t[np.arange(len(t)), argmin_a.astype(np.int32), None]
        if self.prioritized:
            return sa, t, w
        
        return sa, t

    @staticmethod
    def scale(qs, A, B, invert: bool = False):
        # return qs
        # if invert:
        #     return qs * A
        # return qs / A
        if invert:
            return (qs - B) / A
        return A * qs + B
        # eps = 10e-2
        # if invert:
        #     return np.sign(qs) * (((np.sqrt(1+4*eps*(np.abs(qs)+1+eps))-1))/(2*eps)-1)
        # return np.sign(qs) * (np.sqrt(np.abs(qs)+1)-1) + eps * qs

    @staticmethod
    def update_scaling_parameters(out_min: float, out_max: float):
        out_diff = out_max - out_min
        net_diff = NET_MAX - NET_MIN
        if out_diff == 0:
            LOG.warning("RESETTING PARAMS")
            print("RESETTING PARAMS")
            ## EXPERIMENTAL ##
            # It seems as if there are max cost episodes only at the beginning
            # of training, the network will collapse to all the same q value,
            # no matter what the input. In this case, the difference in the
            # network becomes 0 and will cause a ZeroDiv error. To prevent this,
            # A and B are reset to "no scaling" and training continues.
            # This is not without precedent: Riedmiller claims one can reset
            # the scaling parameters at the beginning of each training cycle
            # (line 15 in Fig 3), and experiments showed that the network will
            # eventually converge given that it explores a bit more.
            return 1, 0
        A = net_diff / out_diff
        B = -((out_min * net_diff) / out_diff) + NET_MIN
        return A, B

    def get_A_B(self) -> Tuple[float, float]:
        """Return the updated A and B normalization parameters."""
        return self.A, self.B


def expected_discounted_cost(
    max_steps: int, gamma: float, max_step_cost: Optional[float] = None
) -> float:
    """Helper function to find the appropriate step cost.

    With sigmoidal output NFQ, the total discounted costs over the course
    of an episode should not exceed the maximum sigmoid value: 1. This function
    finds the proper max step cost for a given gamma and max episode steps to
    satisfy that requirement. It can also return what the expected discounted
    future costs would be, when given all three parameters.

    expectation = sum[gamma^i * max_step_cost for i in range(max_steps)]
    
    Args:
        max_steps: The number of steps in an episode
        gamma: The discount factor
        max_step_cost: Optional, if None, will compute it. If provided,
                       will return the expected future discounted costs
                       given this value.
    """
    discounts = sum([gamma ** i for i in range(max_steps)])
    if max_step_cost is None:
        return 1 / discounts
    return discounts * max_step_cost


def tanh2(e: np.ndarray, C: float, mu: float) -> np.ndarray:
    """Smooth differentiable cost function according to Hafner 2011.

    .. note::

        Reinforcement learning in feedback control. Challenges and benchmarks
        from technical process control. Roland Hafner, Martin Riedmiller. 2011

    Args:
        e: Vector of control deviations or "errors".
        C: Maximum value to produce, scales the tanh2 function.
        mu: Width of the tanh2 function. The curve will reach 95% of its max
            height when e equals mu.
    """
    w = np.arctanh(np.sqrt(0.95)) / mu
    return np.tanh(np.abs(e) * w) ** 2 * C


class ObservationStack:
    """Stack of observations over time as the last dimension.

    This class is only used for inference, not during training. It only stores
    as many observations as required for a single inference step, **not** the
    complete memory buffer over many episodes.

        >>> mem = ObservationStack((2,), lookback=2, dtype=np.uint8)
        >>> len(mem)
        2
        >>> mem = mem.append(np.array([1, 1])).append(np.array([2, 2]))
        >>> mem.stack[..., 0].tolist()
        [1, 1]
        >>> mem.stack[..., 1].tolist()
        [2, 2]
        >>> mem = mem.append([3, 3])
        >>> mem.stack[..., 0].tolist()
        [2, 2]
        >>> mem.stack[..., 1].tolist()
        [3, 3]

    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        lookback: int = 1,
        dtype: np.dtype = np.float32,
    ) -> None:
        self._stack = np.zeros(observation_shape + (lookback,), dtype=dtype)
        self._setup = False

    def append(self, observation: np.ndarray) -> "ObservationStack":
        observation = np.asarray(observation)
        self._stack = np.roll(self._stack, -1, axis=-1)
        self._stack[..., -1] = observation
        if not self._setup:
            self._stack[..., :] = observation[..., None]
            self._setup = True
        return self

    def clear(self):
        self._stack = np.zeros(self._stack.shape)

    def __len__(self) -> int:
        return self._stack.shape[-1]

    @property
    def stack(self) -> np.ndarray:
        return self._stack.copy()


def enrich_errors(
    stacks: np.ndarray,
    channels: Tuple[str, ...],
    *control_pairs: Tuple[str, str],
    drop_pvs: bool = False,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Enriches the given observation stacks with control deviations.

    The control deviations overwrite the setpoint columns. The process variable
    columns are retained in their original condition.

    .. todo::

        Make this handle changing setpoints well: Only the most recent setpoint
        of LOOKBACK many setpoints should be used for all LOOKBACK many
        observations in the given stack.

    Example:

        >>> stacks = np.reshape(np.arange(6), (1, 3, 2))
        >>> stacks, channels = enrich_errors(stacks, ("CV", "SP", "PV"), ("SP", "PV"))
        >>> stacks.tolist()
        [[[0, 1], [-2, -2], [4, 5]]]
        >>> channels
        ('CV', 'SP_err', 'PV')

    Args:
        stacks: Stack of observations, shape ``(BATCH, CHANNELS, LOOKBACK)``.
        channels: Channel names of the stacks' ``CHANNELS`` dimension.
        control_pairs: ``(SP, PV)`` pairs of channel names.
        drop_pvs: Whether to drop the process variables from the stack after
                  they were used to compute the error value.

    Returns:
        Modified copy of the ``stacks`` and ``channels`` parameters.

    """
    assert isinstance(channels, tuple), "Channels need to be immutable."
    pvcols = []
    sps = []
    for sp, pv in control_pairs:
        sps.append(sp)
        spcol = channels.index(sp)
        pvcol = channels.index(pv)
        assert spcol > -1 and pvcol > -1, "SP and PV need to be in input channels."
        stacks[:, spcol, :] = stacks[:, spcol, :] - stacks[:, pvcol, :]
        pvcols.append(pvcol)
    channels = tuple(
        [
            f"{c}_err" if c in sps else c
            for i, c in enumerate(channels)
            if not drop_pvs or i not in pvcols
        ]
    )
    if drop_pvs:
        stacks = np.delete(stacks, pvcols, axis=1)
    return stacks, channels


class NFQ(Controller):
    """Neural Fitted Q-Iteration.

    Implementation of a Neural Fitted Q controller, which is based on a neural
    model with states as input and multiple q values as output, one per legal
    discrete action.

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
        scale: Whether or not to linearly scale the q targets into the range [0,1]
               Only do this when training on a single batch, not growing!
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
        scale: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            action_values=action_values,
            lookback=lookback,
            control_pairs=control_pairs,
            num_repeat=num_repeat,
            doubleq=doubleq,
            **kwargs,
        )
        self.lookback = lookback
        self.doubleq = doubleq
        self.prioritized = prioritized
        self.scale = scale
        self.idoe = self.action_type.dtype == "continuous"
        if self.idoe:
            LOG.info("Running NFQ in I-DOE mode.")
        self._optimizer = optimizer
        self._model = model
        self.control_pairs = control_pairs
        if action_values is None:
            action_values = tuple(self.action_type.legal_values[0])
        self.action_values = np.asarray(action_values, dtype=float)

        self._memory = ObservationStack((len(self.state_channels),), lookback=lookback)
        self.normalizer = StackNormalizer("meanstd")
        self.epsilon = 0.0

        self.action_repeat_max = num_repeat
        self.action_repeat = 0
        self._prev_raw_act_and_meta: Optional[
            Tuple[np.ndarray, Dict[str, np.ndarray]]
        ] = None

        assert len(self.action_channels) == 1, "Only supports single actions."

        # Q value normalization parameters; init to "no prior knowledge" values
        self.A = 1
        self.B = 0

        # Prepopulate `self.input_channels` and "warmup" neural network prediction.
        stack = self.preprocess_observations(self._memory.stack[None, ...])
        self.act_model(stack)

    def WRITE_CSV(self, path, name):
        global CSV
        CSV.to_csv(os.path.join(path, f"q_scaling-{name}.csv"))
        CSV = pd.DataFrame()
        self.A = 1
        self.B = 0

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
            with CM["get-actions-predict"]:
                action_indices = self.act_model(stacks).numpy()
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

    @property
    def min_model(self):
        self.maybe_make_model()
        return self._min_model

    @property
    def act_model(self):
        self.maybe_make_model()
        return self._act_model

    def maybe_make_model(self) -> None:
        if hasattr(self, "_train_model"):
            return

        inputs = self._model.inputs
        act_in = tf.keras.Input(shape=(1,), dtype=tf.int32, name="actions")

        # Outputs
        qs = self._model.output  # n_action q_values
        q_act = ExtractIndexLayer(name="q_by_act")([qs, act_in])  # select q by index
        q_min = MinLayer(name="q_min")(qs)  # select most promising action's q value
        act_out = ArgMinLayer(name="act")(qs)  # select most promising action

        # Some metrics, proper python functions to have them be named.
        def avg_q(y_true, y_preds):
            return tf.reduce_mean(y_preds)

        def median_q(*args):
            return tf.contrib.distributions.percentile(qs, 50.0)

        def min_q(y_true, y_preds):
            return tf.reduce_min(y_preds)

        def max_q(y_true, y_preds):
            return tf.reduce_max(y_preds)

        # Model used for training. Gradient decent through a single output unit which
        # is picked by `act_in` -- the action previously executed in state.
        train_model = tf.keras.models.Model(inputs=inputs + [act_in], outputs=q_act)
        train_model.compile(
            optimizer=self._optimizer,
            loss="MSE",
            metrics=[min_q, avg_q, max_q],  # median_q
        )
        # Workaround for `AssertionError`s seen during multi-threaded fit.
        # NOTE: No longer needed with tf2.2+?
        # train_model._make_train_function()

        # Model to get the min q value for state.
        min_model = tf.keras.models.Model(inputs=inputs, outputs=q_min)
        min_model.compile(optimizer=self._optimizer, loss="mean_squared_error")

        # Model to greedily evaluate the model, returning an action index.
        act_model = tf.keras.models.Model(inputs=inputs, outputs=act_out)
        act_model.compile(optimizer=self._optimizer, loss="mean_squared_error")

        self._train_model = train_model
        self._min_model = min_model
        self._act_model = act_model

    def fit(
        self,
        batch: Batch,
        costfunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        iterations: int = 1,
        epochs: int = 1,
        minibatch_size: int = -1,
        gamma: float = 0.99,
        callbacks: Optional[List] = None,
        reset_between_iterations: bool = False,
        **kwargs,
    ) -> None:
        """Train the underlying model.

        Args:
            batch:
            costfunc: A function accepting a numpy array of observations to compute a
                      single cost for each.
            **kwargs: arguments going to keras.model.fit()
                      Example: verbose=0 to suppress Keras output
        """
        if callbacks is None:
            callbacks = []

        if costfunc is not None:
            batch.compute_costs(costfunc)

        for iteration in range(1, iterations + 1):
            LOG.info("Fit iteration: %d", iteration)

            # In the original NFQ, the network is reset between iterations, while
            # within each iteration it is trained to convergence.
            if reset_between_iterations:
                reset_model(self.train_model)

            batch.set_minibatch_size(minibatch_size).shuffle()

            # Train network to output new target q values.
            qmodel = self._model if self.doubleq else self.min_model
            sequence = DQBatch(
                batch,
                qmodel,
                self.act_model,
                self.train_model,
                self.A,
                self.B,
                gamma,
                self.doubleq,
                self.prioritized,
                self.scale,
            )
            if self.scale:
                self.A, self.B = sequence.get_A_B()
            self.train_model.fit(
                sequence, epochs=epochs, callbacks=callbacks, **kwargs,
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
            custom_objects = [Squeeze, ExpandDims]
        config = zipfile.get("config.json")
        model = zipfile.get_keras("model.h5", custom_objects)
        action_meta = zipfile.get_json("Action.json")
        assert isinstance(action_meta, dict)
        action_type = cls.load_action_type(action_meta, custom_objects)
        obj = cls(model=model, action=action_type, **config)
        obj.normalizer = StackNormalizer.load(zipfile)
        return obj
