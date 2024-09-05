# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Value Network
==========================================

A value network to represent state -> expected cost mappings, not explicitly
considering state-transition actions.

.. autosummary::

    VNet

"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf

from psipy.core.io import MemoryZipFile
from psipy.nn.layers import ExpandDims, Squeeze
from psipy.rl.core.controller import Controller
from psipy.rl.controllers.nfq import ObservationStack
from psipy.rl.io.batch import Batch
from psipy.rl.core.plant import Action, State
from psipy.rl.preprocessing import StackNormalizer

__all__ = ["VNet"]


LOG = logging.getLogger(__name__)


class VNet(Controller):
    """A value network for learning state -> expected cost mappings.

    Args:
        model: Neural Value-Function approximator. Expected to be single input,
               sigmoid output.
        lookback: Number of observation timesteps to stack into a single input.
        disable_terminals: Whether to disable transition ending high terminal
                           state cost.
        **kwargs: Further keyword arguments for :class:`Controller`.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        lookback: int = 1,
        disable_terminals: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            lookback=lookback,
            disable_terminals=disable_terminals,
            **kwargs,
        )
        self.disable_terminals = disable_terminals
        self._model = model

        self._memory = ObservationStack((len(self.state_channels),), lookback)

        self.normalizer = StackNormalizer("meanstd")

    def notify_episode_starts(self) -> None:
        raise NotImplementedError("VNet cannot be used in a loop.")

    def get_action(self, state: State) -> Action:
        raise NotImplementedError("VNet provides no actions.")

    def get_actions(
        self, stacks: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        raise NotImplementedError("VNet provides no actions.")

    def preprocess_observations(self, stacks: np.ndarray) -> np.ndarray:
        """Preprocesses observation stacks before those are passed to the network.

        Employed in both the local :meth:`get_actions` method as well as in the
        :class:`~Batch` during training.

        Args:
          stacks: Observation stacks of shape ``(BATCH, CHANNELS, LOOKBACK)``.
        """
        stacks = self.normalizer.transform(stacks)
        return stacks

    def notify_episode_stops(self) -> None:
        raise NotImplementedError("VNet cannot be used in a loop.")

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
            optimizer="rmsprop",
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
        reset_between_iterations: bool = False,
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
            batch.set_minibatch_size(-1).sort()

            # only one batch containing all data shifted by 1, i.e. states[1:]
            inputs = batch.nextstates[0]  # (BATCH, state, dim, lookback)

            qs = self.train_model.predict(inputs)  # (BATCH * N_ACT, 1)
            costs, terminals = batch.costs_terminals[0]

            # update qs with shifted states
            qs = costs + gamma * qs

            if not self.disable_terminals:
                qs[terminals.ravel() == 1] = 1
            # qs[costs.ravel() == 0] = 0
            qs = np.clip((qs - qs.min()) + 0.05, 0.05, 0.95)

            # set targets for training
            batch.set_targets(qs)

            LOG.info(f"Fit iteration: {iteration}")
            batch.set_minibatch_size(minibatch_size).shuffle()
            self.train_model.fit(
                batch.states_targets,
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
