# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Neural Fitted Q-Iteration with Continuous Actions
====================================================

See :mod:`psipy.rl.control` for details.

"""

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras.optimizers import RMSprop

from psipy.core.io import MemoryZipFile
from psipy.nn.optimizers.rprop import Rprop
from psipy.rl.control import Controller
from psipy.rl.control.keras_utils import ActorCriticOptimizerRMSprop
from psipy.rl.control.keras_utils import ActorCriticOptimizerRprop
from psipy.rl.control.keras_utils import clone_model, reset_model
from psipy.rl.control.layers import GaussianNoiseLayer
from psipy.rl.control.nfq import ObservationStack, enrich_errors
from psipy.rl.control.noise import Noise
from psipy.rl.io.batch import Batch
from psipy.rl.plant import State
from psipy.rl.preprocessing import StackNormalizer

__all__ = ["NFQCA"]


class NFQCA(Controller):
    """Neural Fitted Q-Iteration with Continuous Actions.

    Args:
        actor: Actor model, ``state`` input, ``action`` output. Output should
               be ``tanh`` in order for it to be scaled correctly.
        critic: Critic model, ``(state, action)`` input, single Q value output.
                Current implementation is optimized for ``sigmoid`` outputs.
        lookback: Number of history steps from the state to feed into the two
                  models as ``state``.
        td3: Extended :ref:`NFQCA` with ideas from :ref:`TD3`.
        exploration: Exploration :class:`Noise` to use.
        control_pairs: ``(SP, PV)`` tuples of channel names to each merge into a
                       single control deviations channel in the neural network's
                       input.
        disable_terminals: Whether to disable transition ending high terminal
                           state cost.
        optimizer: Optimizer type. Options: "rmsprop" or "rprop". Defaults to
                   "rmsprop".
        kwargs: Keyword arguments to pass up to :class:`Controller`.
    """

    _actor: tf.keras.Model
    _critic: tf.keras.Model
    _chained_critic: tf.keras.Model

    #: :class:`~psipy.rl.preprocessing.normalization.StackNormalizer` instance.
    normalizer: StackNormalizer

    #: Whether :ref:`TD3` tweaks are enabled.
    td3: bool

    #: Standard deviation of gaussian noise added to actor output during
    #: critic training in TD3 mode.
    NOISE_STD: ClassVar[float] = 0.2

    #: Clip value for actor output during critic training in TD3 mode.
    NOISE_CLIP: ClassVar[float] = 0.5

    def __init__(
        self,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        lookback: int = 1,
        td3: bool = False,
        exploration: Optional[Noise] = None,
        control_pairs: Optional[Tuple[Tuple[str, str], ...]] = None,
        drop_pvs: bool = False,
        disable_terminals: bool = False,
        optimizer: str = "rmsprop",
        **kwargs,
    ):
        super().__init__(
            lookback=lookback,
            td3=td3,
            control_pairs=control_pairs,
            drop_pvs=drop_pvs,
            disable_terminals=disable_terminals,
            optimizer=optimizer,
            **kwargs,
        )
        assert self.action_type.dtype == "continuous"
        assert len(self.action_type.legal_values) == 1, "Only single channel actions."
        assert actor.output.shape[1] == 1, "Only supports single action actors atm."
        assert critic.output.shape[1] == 1

        legal_values = self.action_type.get_legal_values(*self.action_channels)

        self.control_pairs = control_pairs
        self.drop_pvs = drop_pvs
        self.disable_terminals = disable_terminals
        self.td3 = td3
        self.optimizer = optimizer
        self.normalizer = StackNormalizer("meanstd")
        self.action_normalizer = StackNormalizer("meanmax").fit(
            np.array(legal_values).T
        )
        self.exploration = exploration

        self._memory = ObservationStack((len(self.state_channels),), lookback)

        self._actor = actor
        self._critic = critic

        # In order to save the models, we create copies of the model structure/weights
        # but not the actual model; thus weights will transfer but compilation does not.
        # TODO: Move this to `save`?
        self.__actor = tf.keras.Model(actor.inputs, actor.outputs)
        self.__critic = tf.keras.Model(critic.inputs, critic.outputs)

        # Prepopulate `self.input_channels` and "warmup" neural network prediction.
        self.get_actions(self._memory.stack[None, ...])

    def get_q(self, state: Optional[State] = None) -> np.ndarray:
        """Produces a single q value given a state.

        Args:
            state: State object from the plant. If it is not provided, the current
                   ``state`` in the memory of the :class:`NFQCA` is used.

        Returns:
            Q values as float in a ndarray
        """
        if state is None:
            stacks = self._memory.stack[None, ...]
        else:
            state_values = state.as_array(*self.state_channels)
            self._memory.append(state_values)
            stacks = self._memory.stack[None, ...]
        stacks = self.preprocess_observations(stacks)
        return self.chained_critic(stacks).numpy().ravel()

    def _get_action(self, observation: np.ndarray) -> np.ndarray:
        """Produce a single action value given one observation from the plant.

        Args:
            observation: Vector of measurements. Currently no images or
                similar supported!

        Returns:
            Single action float, although in a ndarray shaped (1,)
        """
        self._memory.append(observation)
        return self.get_actions(self._memory.stack[None, ...])

    def get_actions(self, stacks: np.ndarray) -> np.ndarray:
        """Computes actions given many state stacks.

        Args:
            stacks: Observation stacks of shape ``(BATCH, CHANNELS, LOOKBACK)``.
        """
        stacks = self.preprocess_observations(stacks)
        actions = self.actor(stacks).numpy()

        if self.exploration:
            actions = self.exploration(actions)

        actions = self.action_normalizer.inverse_transform(actions)

        # Currently only a single action per stack-item is supported.
        actions = actions.ravel()

        # Clip the actions because the network might output a value outside the
        # range of legal values. In gym environments out of bounds actions might
        # cause assertion errors, in other processes even undefined behavior.
        lower, upper = self.action_type.legal_values[0]
        return np.clip(actions, lower, upper)

    def preprocess_observations(self, stacks: np.ndarray) -> np.ndarray:
        """Preprocesses observation stacks before those are passed to the network.

        Employed in both the local :meth:`get_actions` method as well as in the
        :class:`~Batch` during training.

        Args:
          stacks: Observation stacks of shape ``(BATCH, CHANNELS, LOOKBACK)``.
        """
        stacks = self.normalizer.transform(stacks)
        channels = self.state_channels
        if self.control_pairs is not None:
            stacks, channels = enrich_errors(
                stacks, channels, *self.control_pairs, drop_pvs=self.drop_pvs
            )
        self.input_channels = channels
        return stacks

    def notify_episode_starts(self) -> None:
        ...

    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop.

        Raises:
            NotNotifiedOfEpisodeStart
        """
        self._memory.clear()
        if self.exploration:
            self.exploration.reset()

    def get_optimizer(
        self, action_bounds: Optional[Tuple[float, float]] = None, actorcritic=False
    ) -> tf.keras.optimizers.Optimizer:
        """Returns optimizer instance.

        Args:
            action_bounds: Lower and upper bounds for the actions.

        Raises:
            ValueError: Raised when optimizer type is not supported
        """
        if self.optimizer == "rmsprop":
            if actorcritic:
                return ActorCriticOptimizerRMSprop(
                    self._actor.output, action_bounds=action_bounds
                )
            return RMSprop()

        if self.optimizer == "rprop":
            if actorcritic:
                return ActorCriticOptimizerRprop(
                    self._actor.output, action_bounds=action_bounds
                )
            return Rprop()

        raise ValueError(f"Optimizer type {self.optimizer} not supported")

    def maybe_make_model(self) -> None:
        """Create and compile all required models, if not done so yet."""
        if hasattr(self, "_chained_critic"):
            return

        common_state = self._actor.inputs
        action_out = self._actor.output
        action_in = self._critic.inputs[-1]

        if self.td3:
            # When td3 is enabled, add gaussian noise to actor output during
            # critic training. This is only used in "chained" models below, the
            # actor itself is left untouched and stays deterministic.
            noise = GaussianNoiseLayer(self.NOISE_STD, self.NOISE_CLIP, output_clip=1)
            action_out = noise(action_out)

        # Network output might be a outside bounds due to linear output
        # activations. Although the actor is optimized to stick to the
        # output bounds using inverted gradients (see ActorCriticOptimizer),
        # it might still produce values slightly off. Note that clipping here
        # is only applied to the actions fed into the critic, not to the actual
        # actor output. This means that the actual actor output may still be
        # out of bounds, which is required for gradient inversion! The actions
        # are again clipped on the numpy side in get_actions
        # NOTE: Currently disabled for debugging purposes, as clipping hides
        #       when action bounds are not learned properly.
        # action_out = ClipLayer(-1, 1)(action_out)

        critic = self._critic
        q1 = critic(common_state + [action_in])
        chained_q1 = critic(common_state + [action_out])

        # Keep q1 and chained_q1 for when td3 is enabled. q and chained_q is
        # overwritten by joined value if td3 is enabled.
        q = q1
        chained_q = chained_q1

        if self.td3:
            # TD3 uses two q networks instead of one, always taking the "worse"
            # expectation in order to not overestimate state value.
            critic2 = clone_model(self._critic, name="critic2")
            q2 = critic2(common_state + [action_in])
            q = tfkl.maximum([q, q2])
            chained_q2 = critic2([common_state, action_out])
            chained_q = tfkl.maximum([chained_q, chained_q2])  # q target

        self._critic = tf.keras.models.Model(
            inputs=common_state + [action_in], outputs=q
        )
        self._chained_critic = tf.keras.models.Model(
            inputs=common_state, outputs=chained_q, name="chained_critic"
        )

        def min_q(*args):
            return tf.reduce_min(chained_q)

        def avg_q(*args):
            """Average of the critics' qs given the current actor's actions."""
            return tf.reduce_mean(chained_q)

        def avg_q1(*args):
            """Average of the first critic's qs given the current actor's actions."""
            # Even in td3, the policy is updated by only minimizing q1.
            return tf.reduce_mean(chained_q1)

        def max_q(*args):
            return tf.reduce_max(chained_q)

        def min_act(*args):
            return tf.reduce_min(action_out)

        def avg_act(*args):
            return tf.reduce_mean(action_out)

        def max_act(*args):
            return tf.reduce_max(action_out)

        def min_act_in(*args):
            return tf.reduce_min(action_in)

        def avg_act_in(*args):
            return tf.reduce_mean(action_in)

        def max_act_in(*args):
            return tf.reduce_max(action_in)

        def critic_loss(q_targets: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
            """Compute the mse appropriate for td3 or non-td3 mode."""
            Loss = tf.keras.losses.MeanSquaredError
            # Loss = tf.keras.losses.Huber
            mse = Loss()(q_targets, q1)
            if self.td3:
                mse = mse + Loss()(q_targets, q2)
            return mse

        # Compile critic model, collecting trainable variables.
        self._critic.compile(
            optimizer=self.get_optimizer(),
            loss=critic_loss,
            metrics=[
                min_q,
                avg_q,
                max_q,
                min_act_in,
                avg_act_in,
                max_act_in,
                min_act,
                avg_act,
                max_act,
            ],
        )

        # Make the actor only train actor variables, although the gradient is
        # computed through both the actor and the critic models. Note that this
        # has the effect of any layers which are shared between the critic and
        # the actor to only be trained by the actor!
        # critic_vars = self._critic.trainable_variables.copy()
        for layer in self._critic.layers:
            layer.trainable = False

        action_bounds = None
        if self._actor.layers[-1].activation is tf.keras.activations.linear:
            action_bounds = (-1, 1)

        self._actor.compile(
            optimizer=self.get_optimizer(action_bounds=action_bounds, actorcritic=True),
            loss=lambda t, p: chained_q1,  # NOTE: Using output of q1 only, even in td3!
            metrics=[min_q, max_q, avg_q, min_act, avg_act, max_act],
        )

        # Make sure the critic's trainable variables are still the same as
        # before and reset the critic layers' trainable states to suppress
        # Keras' "trainable variable mismatch" warning.
        # NOTE: Incompatible with tf2
        # assert self._critic._collected_trainable_weights == critic_vars
        for layer in self._critic.layers:
            layer.trainable = True

    @property
    def chained_critic(self) -> tf.keras.Model:
        self.maybe_make_model()
        return self._chained_critic

    @property
    def critic(self) -> tf.keras.Model:
        self.maybe_make_model()
        return self._critic

    @property
    def actor(self) -> tf.keras.Model:
        self.maybe_make_model()
        return self._actor

    def fit_critic(
        self,
        batch: Batch,
        costfunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        iterations: int = 1,
        epochs: int = 1,
        minibatch_size: int = -1,
        gamma: float = 0.99,
        # autostop: bool = False,
        callbacks: Optional[List] = None,
        reset_between_iterations: bool = False,
        **kwargs,
    ) -> None:
        """Fits the critic model.

        Args:
            **kwargs: Arguments going to ``keras.model.fit()``.
                      Example: ``verbose=0`` to suppress keras output
        """
        if callbacks is None:
            callbacks = []
        if costfunc is not None:
            batch.compute_costs(costfunc)

        for _iteration in range(1, iterations + 1):
            # Compute target qs using the next states' predicted q values and bellman.
            batch.set_minibatch_size(-1).sort()
            qs = self.chained_critic(batch.nextstates).numpy().ravel()
            costs, terminals = batch.costs_terminals[0]
            target_qs = costs.ravel() + gamma * qs
            if self.disable_terminals:
                # The following should use np.max(qs) when the model uses `relu`
                # output activations instead of `sigmoid`.
                target_qs[terminals.ravel() == 1] = 1
            assert np.all(target_qs >= 0)
            target_qs = target_qs - np.min(target_qs)
            target_qs = np.clip(target_qs + 0.05, 0.05, 0.95)
            batch.set_targets(target_qs)

            # In the original NFQ(CA), the network is reset between iterations,
            # while within each iteration it is trained to convergence from
            # scratch.
            if reset_between_iterations:
                self.reset_critic()

            # Train network to output new target q values.
            batch.set_minibatch_size(minibatch_size).shuffle()
            self.critic.fit(
                batch.statesactions_targets,
                epochs=epochs,
                callbacks=callbacks,
                **kwargs
                # initial_epoch=(iteration - 1) * epochs,
            )

    def fit_actor(
        self,
        batch: Batch,
        minibatch_size: int = -1,
        epochs: int = 1,
        # autostop: bool = False,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        reset_between_iterations: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Fit the actor model

        Args:
            **kwargs: arguments going to keras.model.fit()
                      Example: verbose=0 to suppress keras output
        """
        if callbacks is None:
            callbacks = []

        # In the original NFQ(CA), the network is reset between iterations,
        # while within each iteration it is trained to convergence from
        # scratch.
        if reset_between_iterations:
            self.reset_actor()

        batch.set_minibatch_size(minibatch_size).shuffle()
        history = self.actor.fit(
            batch.states, epochs=epochs, callbacks=callbacks, **kwargs
        )
        return history.history

    def fit_normalizer(
        self, observations: np.ndarray, method: Optional[str] = None
    ) -> None:
        """Fit the :class:`~psipy.rl.preprocessing.normalization.StackNormalizer`."""
        if method and method != self.normalizer.method:
            self.normalizer = StackNormalizer(method)
        self.normalizer.fit(observations)

    def reset_actor(self) -> None:
        """Resets the actor model."""
        reset_model(self.actor)

    def reset_critic(self) -> None:
        """Resets the critic model."""
        reset_model(self.critic)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NFQCA":
        """Initializes an NFQCA instance from a config dict.

        Raises:
            NotImplementedError: Currently method is not implemented. TODO as
                this is a requirement to adhere to the
                :mod:`~psipy.core.io.Saveable` interface.
        """
        raise NotImplementedError("Cannot initialize from config.")

    def _save(self, zipfile: MemoryZipFile) -> MemoryZipFile:
        zipfile.add("config.json", self.get_config())
        zipfile.add("actor_model.h5", self.__actor)
        zipfile.add("critic_model.h5", self.__critic)
        zipfile.add_json(
            "Action.json",
            dict(
                class_name=self.action_type.__name__,
                class_module=self.action_type.__module__,
            ),
        )
        self.normalizer.save(zipfile)
        if self.exploration:
            self.exploration.save(zipfile)
        return zipfile

    @classmethod
    def _load(
        cls, zipfile: MemoryZipFile, custom_objects: Optional[List[Type[object]]] = None
    ) -> "NFQCA":
        config = zipfile.get("config.json")
        actor_model = zipfile.get_keras("actor_model.h5", custom_objects)
        critic_model = zipfile.get_keras("critic_model.h5", custom_objects)
        action_meta = zipfile.get_json("Action.json")
        assert isinstance(action_meta, dict)
        action_type = cls.load_action_type(action_meta, custom_objects)
        obj = cls(actor=actor_model, critic=critic_model, action=action_type, **config)
        obj.normalizer = StackNormalizer.load(zipfile)
        try:
            obj.exploration = Noise.load_impl(zipfile)
        except FileNotFoundError:  # Exploration noise is optional.
            pass
        return obj
