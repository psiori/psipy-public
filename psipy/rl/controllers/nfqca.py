# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Neural Fitted Q-Iteration with Continuous Actions
====================================================

See :mod:`psipy.rl.controllers` for details.

"""

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers as tfkl
from tensorflow.keras.optimizers import RMSprop

from psipy.core.io import MemoryZipFile
from psipy.nn.optimizers.rprop import Rprop
from psipy.rl.core.controller import Controller
from psipy.rl.controllers.keras_utils import ActorCriticOptimizerRMSprop
from psipy.rl.controllers.keras_utils import ActorCriticOptimizerRprop
from psipy.rl.controllers.keras_utils import clone_model, reset_model
from psipy.rl.controllers.layers import GaussianNoiseLayer
from psipy.rl.controllers.nfq import ObservationStack, enrich_errors
from psipy.rl.controllers.noise import Noise
from psipy.rl.io.batch import Batch
from psipy.rl.core.plant import State
from psipy.rl.preprocessing import StackNormalizer

from psipy.nn.optimizers.rprop import Rprop


__all__ = ["NFQCA", "NFQCA_Actor"]


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
        print(f"NFQCA __init__: {kwargs}")
        print(f"NFQCA self.id: {self.id}")
        print(f"NFQCA self._config: {self._config}")
        print(f"NFQCA self.get_config(): {self.get_config()}")

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
        breakpoint() # TODO implement this method using tape

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
        #chained_q1 = critic(common_state + [action_out])

        # Keep q1 and chained_q1 for when td3 is enabled. q and chained_q is
        # overwritten by joined value if td3 is enabled.
        q = q1
        #chained_q = chained_q1

        if self.td3:
            #breakpoint() # TODO implement this method using tape

            # TD3 uses two q networks instead of one, always taking the "worse"
            # expectation in order to not overestimate state value.
            critic2 = clone_model(self._critic, name="critic2")
            q2 = critic2(common_state + [action_in])
            q = tfkl.maximum([q, q2])
            chained_q2 = critic2([common_state, action_out])
            chained_q = tfkl.maximum([chained_q, chained_q2])  # q target

        #self._critic = tf.keras.models.Model(
        #    inputs=common_state + [action_in], outputs=q
        #)
        #self._chained_critic = tf.keras.models.Model(
        #    inputs=common_state, outputs=chained_q, name="chained_critic"
        #)

        self._chained_critic = True 

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

        #def critic_loss(q_targets: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
        #    """Compute the mse appropriate for td3 or non-td3 mode."""
            #Loss = tf.keras.losses.MeanSquaredError
            # Loss = tf.keras.losses.Huber
         #   mse = Loss()(q_targets, q1)
         #   if self.td3:
         #       mse = mse + Loss()(q_targets, q2)
         #   return mse
        
        # Compile critic model, collecting trainable variables.
        #self._critic.compile(
        #    optimizer=self.get_optimizer(),
        #    loss=CriticLoss(), # critic_loss,
            # metrics=[
            #     min_q,
            #     avg_q,
            #     max_q,
            #     min_act_in,
            #     avg_act_in,
            #     max_act_in,
            #     min_act,
            #     avg_act,
            #     max_act,
            # ],
        #)

        #self.critic_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        #self.actor_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        #self.critic_opt = tf.keras.optimizers.SGD(learning_rate=0.0001) 
        #self.actor_opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=0.0001) # 0.0001 
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


        #self.critic_opt = Rprop()
        #self.actor_opt = Rprop()

        # Make the actor only train actor variables, although the gradient is
        # computed through both the actor and the critic models. Note that this
        # has the effect of any layers which are shared between the critic and
        # the actor to only be trained by the actor!
        # critic_vars = self._critic.trainable_variables.copy()
        #for layer in self._critic.layers:
        #    layer.trainable = False

        #action_bounds = None
        #if self._actor.layers[-1].activation is tf.keras.activations.linear:
        #    action_bounds = (-1, 1)

        #self._actor.compile(
        #    optimizer=self.get_optimizer(action_bounds=action_bounds, actorcritic=True),
        #    loss=lambda t, p: chained_q1,  # NOTE: Using output of q1 only, even in td3!
        #    #metrics=[min_q, max_q, avg_q, min_act, avg_act, max_act],
        #)

        # Make sure the critic's trainable variables are still the same as
        # before and reset the critic layers' trainable states to suppress
        # Keras' "trainable variable mismatch" warning.
        # NOTE: Incompatible with tf2
        # assert self._critic._collected_trainable_weights == critic_vars
        #for layer in self._critic.layers:
        #    layer.trainable = True

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

    @classmethod
    def default_critic_model(
        cls,
        state_dim: int,
        action_dim: int = 1,
        lookback: int = 1,
        hidden_dim: int = 256,
        feature_dim: int = 100,
    ) -> tf.keras.Model:
        """Create a default critic model.

        Args:
            state_dim: Dimension of the state.
            action_dim: Dimension of the action. Presently only 1 is supported.
            lookback: Number of history steps from the state to feed into the critic.
        """
        inp = tfkl.Input((state_dim, lookback), name="state_critic")
        act = tfkl.Input((action_dim,), name="act_in")
        net = tfkl.Concatenate()([tfkl.Flatten()(inp), act])
        net = tfkl.Dense(hidden_dim, activation="relu")(net)
        net = tfkl.Dense(hidden_dim, activation="relu")(net)
        net = tfkl.Dense(feature_dim, activation="tanh")(net)
        net = tfkl.Dense(1, activation="sigmoid")(net)
        model = tf.keras.Model([inp, act], net, name="critic")
        return model

    @classmethod
    def default_actor_model(
        cls,
        state_dim: int,
        action_dim: int = 1,
        lookback: int = 1,
        hidden_dim: int = 256,
        feature_dim: int = 100,
    ) -> tf.keras.Model:
        """Create a default actor model.

        Args:
            state_dim: Dimension of the state.
            action_dim: Dimension of the action. Presently only 1 is supported.
            lookback: Number of history steps from the state to feed into the actor.
        """
        inp = tfkl.Input((state_dim, lookback), name="state_actor")
        net = tfkl.Flatten()(inp)
        net = tfkl.Dense(hidden_dim, activation="relu")(net)
        net = tfkl.Dense(hidden_dim, activation="relu")(net)
        net = tfkl.Dense(feature_dim, activation="tanh")(net)
        net = tfkl.Dense(action_dim, activation="tanh")(net)
        model = tf.keras.Model(inp, net, name="actor")
        return model

    def fit_critic(
        self,
        batch: Batch,
        costfunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        iterations: int = 1,
        epochs: int = 1,
        minibatch_size: int = -1,
        gamma: float = 0.98,
        callbacks: Optional[List] = None,
        reset_between_iterations: bool = False,
        clip_targets_top: bool = True,
        clip_targets_bottom: bool = False,
        use_qmin_trick: bool = False,
        **kwargs,
    ) -> None:
        """Fits the critic model.

        Args:
            gamma: Discount factor. Contradicting the belief of others in the
                field, we believe in pragamaticaly using a gamma well below 1,
                although this may lead to sub-optimal control policies, e.g. 
                in shortest path to goal settings. In our experience, a gamma < 
                1.0 helps a lot regarding stability and robustness over a 
                variaty of tasks and parameter settings, especially in the 
                beginning of the training process, where q-values in the goal 
                area are most likely to be overestimated. We choose a value of 
                0.98 as standard, which in the limit results in a maximum of 50 
                x the default_step costs as expected future costs. This should 
                be kept in mind as a guidance what length of lookahead can be 
                expected and also when chosing the default step cost in such a 
                such a way the future cost estimate is bounded below 1.0. We 
                have good experience with increasing gamma towards 1.0 later in 
                the training process for achieving longer lookaheads and 
                assuring optimal control policies. If necessary for stability, 
                this can be done in several steps distributed over the training 
                procedure. In our experience, this approach does not lead to 
                any negative effects in the quality of the final control policy.
            clip_targets_top: Whether to clip targets at 0.95 to prevent the   
                optimizer from pushing the weights too far into approximating a 
                1.
                This method was sugested by R. Hafner in "Dateneffiziente 
                selbstlernende neuronale Regler", 2009. In our experience, this 
                can be switched on without negative effects, positive effects 
                in this more modern implemenation are presently unclear.
            clip_targets_bottom: Whether to clip targets at 0.05 to prevent the 
                optimizer from pushing the weights too far into approximating a 
                0. This seems to have negative effects, with our standard 
                modelling of zero-costs in the goal region. Do not use without 
                good reason and comparison with the default setting.
            qmin: Whether to use the q-min trick also suugested by R. Hafner in 
                "Dateneffiziente selbstlernende neuronale Regler" 2009 to 
                improve stability. This will deduct the minimum q-target value 
                from all q-targets to "pull" the targets towards zero. From our 
                preliminary experience this is not necessary on systems like    
                the cartpole, but it sometimes quickly helps to "pull" a 
                network with random initial high estimates of the goal states 
                to zero. Where the algorithm might need dozens to hundreds of 
                iterations to pull the network down to zero, switching this on 
                for just one or two iterations can help immediately improve 
                estimates in the critic. Whether this has really a positive 
                effect on the actor and the overall learning time needed is 
                unclear for now. Please note, that we have never seen a network 
                to "diverge" (q-target values to run away, by constantly 
                increasing everywhere) in practice with modern weight 
                initialization and optimizers, iff using a gamma well below 1.

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

            #breakpoint()

            ns = tf.convert_to_tensor(batch.nextstates[0], dtype=tf.float32)
            a = self.actor(ns)
            qs = self.critic([ns, a])

            #qs = self.chained_critic(batch.nextstates[0]).numpy().ravel()
            costs, terminals = batch.costs_terminals[0]
            target_qs = costs.ravel() + gamma * tf.squeeze(qs, axis=-1)  # TODO: move both to tf (or stay in numpy?? --> copy is slow), * (1 - terminals)
            if self.disable_terminals:
                # The following should use np.max(qs) when the model uses `relu`
                # output activations instead of `sigmoid`.
                target_qs[terminals.ravel() == 1] = 1
            assert np.all(target_qs >= 0)

            print(f"min target_qs: {np.min(target_qs)}")
            print(f"max target_qs: {np.max(target_qs)}")
            print(f"mean target_qs: {np.mean(target_qs)}")
            print(f"std target_qs: {np.std(target_qs)}")

            #breakpoint()

            if use_qmin_trick:
                target_qs = target_qs - np.min(target_qs)
            if clip_targets_top and clip_targets_bottom:
                target_qs = np.clip(target_qs + 0.05, 0.05, 0.95)
            if clip_targets_top:
                target_qs = np.clip(target_qs, 0.0, 0.995)  
            if clip_targets_bottom:
                target_qs = np.clip(target_qs + 0.05, 0.05, 1.0)

            batch.set_targets(target_qs)


            # In the original NFQ(CA), the network is reset between iterations,
            # while within each iteration it is trained to convergence from
            # scratch.
            if reset_between_iterations:
                self.reset_critic()

            # Train network to output new target q values.
            batch.set_minibatch_size(minibatch_size).shuffle()

            #breakpoint()
            #self.critic.fit(
            #    batch.statesactions_targets,
            #    epochs=epochs,
            #    callbacks=callbacks,
            #    **kwargs
                # initial_epoch=(iteration - 1) * epochs,
            #)


        # Target actions and target Q
        #na = self.actor_targ(ns)
        #tq = self.critic_targ([ns, na])
        # Bellman backup: y = r + gamma * (1 - done) * tq
        #y = r + self.gamma * (1.0 - d) * tf.squeeze(tq, axis=-1)

        #with tf.GradientTape() as tape:
        #    q = tf.squeeze(self.critic([s, a]), axis=-1)
        #    loss = self.mse(y, q)
        #grads = tape.gradient(loss, self.critic.trainable_variables)
        #self.critic_opt.apply_gradients(zip(grads, self.critic.#trainable_variables))
        #return loss

            self.mse = tf.keras.losses.MeanSquaredError()

            for epoch in range(epochs):
                for statesactions, targets, weights in batch.statesactions_targets:

                    print (f"num samples: {len(statesactions[1])}")

                    st = tf.convert_to_tensor(statesactions[0], dtype=tf.float32)
                    at = tf.convert_to_tensor(statesactions[1], dtype=tf.float32)
                    tt = tf.convert_to_tensor(targets, dtype=tf.float32)

                    #breakpoint()
                    with tf.GradientTape() as tape:
                        #breakpoint()
                        q = self.critic([st, at])
                        assert len(q) == len(tt)
                        assert len(q[0]) == 1
                        assert len(tt[0]) == 1
                        assert tt.shape == q.shape

                        loss = self.mse(tt, q)  # this assumes, both are of dims Nx1, not just N, thus, targets are warpped in 1d arrays

                    #breakpoint()

                    grads = tape.gradient(loss, self.critic.trainable_variables)

                    self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))    
                    
                    print(f"epoch {epoch} critic update with loss: {loss.numpy()}")

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

        self.increment_generation()

        # In the original NFQ(CA), the network is reset between iterations,
        # while within each iteration it is trained to convergence from
        # scratch.
        if reset_between_iterations:
            self.reset_actor()

        batch.set_minibatch_size(minibatch_size).shuffle()

        for epoch in range(epochs):
            for s in batch.states:
                st = tf.convert_to_tensor(s, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    a = self.actor(st)
                    q = self.critic([st, a])
                    # Minimize(!) Q ⇒ minimize mean Q, because we have costs
                    loss = tf.reduce_mean(q)
                grads = tape.gradient(loss, self.actor.trainable_variables)
                self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

                print(f"epoch {epoch} actor update with mean q: {loss.numpy()}")


#        history = self.actor.fit(
#            batch.states, epochs=epochs, callbacks=callbacks, **kwargs
#        )
#        return history.history

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

    def get_default_basename(self) -> str:
        return f"nfqca-{ self.id_strand }"

    def get_default_filename(self) -> str:
        return f"nfqca-{ self.id }"

    def _save(self, zipfile: MemoryZipFile) -> MemoryZipFile:
        zipfile.add("config.json", self.get_config())
        zipfile.add("actor_model.keras", self.__actor)
        zipfile.add("critic_model.keras", self.__critic)
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
        actor_model = zipfile.get_keras("actor_model.keras", custom_objects, support_legacy=True)
        critic_model = zipfile.get_keras("critic_model.keras", custom_objects, support_legacy=True)
        action_meta = zipfile.get_json("Action.json")
        assert isinstance(action_meta, dict)

        # KERAS 3 presently (2025-08-22) does NOT preserve the models input and # output layer types; it wraps tensors in a list, if they were not 
        # already.
        # this seems to be "by intetntion" or at least not likely to change
        # quickly: https://github.com/keras-team/keras/issues/19999
        # so we need to unwrap the tensors here.
        # so annoying that the new distribution violates basic contracts 
        # like model == model.save().load() :(
        # Unfortunately, this will loose all internal states. 
        actor_model = tf.keras.Model(
            inputs = actor_model.inputs[0] if isinstance(actor_model.inputs, (list, tuple)) else actor_model.inputs, 
            outputs = actor_model.outputs[0] if isinstance(actor_model.outputs, (list, tuple)) else actor_model.outputs)
        
        critic_model = tf.keras.Model(
            inputs = critic_model.inputs, # do not unwrap, its a list by intention already before saving and loading 
            outputs = critic_model.outputs[0] if isinstance(critic_model.outputs, (list, tuple)) else critic_model.outputs)

        action_type = cls.load_action_type(action_meta, custom_objects)
        obj = cls(actor=actor_model, critic=critic_model, action=action_type, **config)
        obj.normalizer = StackNormalizer.load(zipfile)
        try:
            obj.exploration = Noise.load_impl(zipfile)
        except FileNotFoundError:  # Exploration noise is optional.
            pass
        return obj



class NFQCA_Actor(Controller):
    """Pure-actor compatible with the NFQCA class.
    
    This class loads only the actor model and relevant settings from NFQCA 
    saved files, ignoring the critic. It provides a mechanism for random 
    exploration and can reload the neural network and input transformations 
    from specified file paths.

    This actor class is prepared to be used in the Collect & Infer paradigm, where the training is separated from the collection of additional data.

    Args:
        actor: Actor model, ``state`` input, ``action`` output. Output should
               be ``tanh`` in order for it to be scaled correctly.
        lookback: Number of history steps from the state to feed into the model.
        exploration: Exploration :class:`Noise` to use.
        control_pairs: ``(SP, PV)`` tuples of channel names to each merge into a
                       single control deviations channel in the neural network's
                       input.
        drop_pvs: Whether to drop PV values when creating control pairs.
        **kwargs: Keyword arguments to pass up to :class:`Controller`.
    """

    def __init__(
        self,
        actor: tf.keras.Model,
        lookback: int = 1,
        exploration: Optional[Noise] = None,
        control_pairs: Optional[Tuple[Tuple[str, str], ...]] = None,
        drop_pvs: bool = False,
        **kwargs,
    ):
        super().__init__(
            lookback=lookback,
            control_pairs=control_pairs,
            drop_pvs=drop_pvs,
            **kwargs,
        )

        assert self.action_type.dtype == "continuous"
        assert len(self.action_type.legal_values) == 1, "Only single channel actions."
        assert actor.output.shape[1] == 1, "Only supports single action actors atm."

        legal_values = self.action_type.get_legal_values(*self.action_channels)

        self.control_pairs = control_pairs
        self.drop_pvs = drop_pvs
        self.normalizer = StackNormalizer("meanstd")
        self.action_normalizer = StackNormalizer("meanmax").fit(
            np.array(legal_values).T
        )
        self.exploration = exploration

        self._memory = ObservationStack((len(self.state_channels),), lookback)
        self._actor = actor

        # Prepopulate `self.input_channels` and "warmup" neural network prediction.
        self.get_actions(self._memory.stack[None, ...])

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
        actions = self._actor(stacks).numpy()

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
        """Handles episode start, called from the loop."""
        pass

    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop."""
        self._memory.clear()
        if self.exploration:
            self.exploration.reset()

    def reload(
            self,
            file_path: str,
            load_all_settings: bool = False
    ) -> None:
        """Reload the neural network and transformations from a specified file path from within a NFQCA file.
        
        Args:
            file_path: Path to the NFQCA saved file.
            load_all_settings: If True, read all settings from the file including
                              exploration settings. If False (default), only reload
                              the actor model and transformations, preserving current
                              exploration settings.
        """
        
        # Load the saved file
        zipfile = MemoryZipFile(file_path)

        top_level = [
            d for d in zipfile.ls(include_directories=True) if d.endswith("/")
        ]
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
        zipfile.cd("NFQCA")

        self._reload(zipfile, load_all_settings)
    
    def _reload(
            self,
            zipfile: MemoryZipFile,
            load_all_settings: bool = False
    ) -> None:
        
        # Load actor model
        actor_model = zipfile.get_keras("actor_model.keras", support_legacy=True)
        
        # Handle Keras 3 tensor wrapping issue
        actor_model = tf.keras.Model(
            inputs=actor_model.inputs[0] if isinstance(actor_model.inputs, (list, tuple)) else actor_model.inputs,
            outputs=actor_model.outputs[0] if isinstance(actor_model.outputs, (list, tuple)) else actor_model.outputs
        )
        
        # Update the actor model
        self._actor = actor_model
        
        # Load normalizer
        self.normalizer = StackNormalizer.load(zipfile)
                
        if load_all_settings:
            # Load all settings from config
            config = zipfile.get("config.json")
            self.control_pairs = config.get("control_pairs")
            self.drop_pvs = config.get("drop_pvs", False)
            
            # Load exploration if it exists
            try:
                self.exploration = Noise.load_impl(zipfile)
            except FileNotFoundError:
                self.exploration = None

    def _save(
        self,
        zipfile: MemoryZipFile
    ) -> MemoryZipFile:
        """Save the actor model and settings to a zipfile."""
        zipfile.add("config.json", self.get_config())
        zipfile.add("actor_model.keras", self._actor)
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
        cls,
        zipfile: MemoryZipFile,
        custom_objects: Optional[List[Type[object]]] = None
    ) -> "NFQCA_Actor":
        """Load an NFQCA_Actor instance from a zipfile."""
        config = zipfile.get("config.json")
        actor_model = zipfile.get_keras("actor_model.keras", custom_objects, support_legacy=True)
        action_meta = zipfile.get_json("Action.json")
        assert isinstance(action_meta, dict)

        # Handle Keras 3 tensor wrapping issue
        actor_model = tf.keras.Model(
            inputs=actor_model.inputs[0] if isinstance(actor_model.inputs, (list, tuple)) else actor_model.inputs,
            outputs=actor_model.outputs[0] if isinstance(actor_model.outputs, (list, tuple)) else actor_model.outputs
        )

        action_type = cls.load_action_type(action_meta, custom_objects)
        obj = cls(actor=actor_model, action=action_type, **config)
        obj.normalizer = StackNormalizer.load(zipfile)
        
        # Reconstruct action_normalizer from action type (NFQCA doesn't save it)
        legal_values = obj.action_type.get_legal_values(*obj.action_channels)
        obj.action_normalizer = StackNormalizer("meanmax").fit(np.array(legal_values).T)
        try:
            obj.exploration = Noise.load_impl(zipfile)
        except FileNotFoundError:  # Exploration noise is optional.
            pass
        return obj 


