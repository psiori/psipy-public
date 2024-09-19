# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Keras utility functions.

.. autosummary::

    ActorCriticOptimizer
    ActorCriticOptimizerRMSprop
    ActorCriticOptimizerRprop
    clone_model
    reset_model
    reset_session

"""

import logging
from typing import Optional, Tuple

import tensorflow as tf

from psipy.nn.optimizers.rprop import Rprop

__all__ = [
    "ActorCriticOptimizer",
    "ActorCriticOptimizerRMSprop",
    "ActorCriticOptimizerRprop",
    "clone_model",
    "reset_model",
    "reset_session",
]


LOG = logging.getLogger(__name__)


class ActorCriticOptimizer:
    """Optimizer for neural network Actor/Critic RL algorithms.

    Optimizes the parameters of an actor according to a critic network,
    minimizing the critic's output.

    Example::

        optimizer = ActorCriticOptimizer(self._actor.output, action_bounds=(-1, 1))
        self._actor.compile(
            optimizer=optimizer,
            loss=lambda t, p: chained_q,
            metrics=[min_q, max_q, avg_q, min_act, avg_act, max_act],
        )

    Args:
        actions: Output tensor of the actor.
        action_bounds: Lower and upper bounds for the actions.
    """

    def get_gradients(self, loss, params):
        """Calculates the policy gradients through the critic network.

        Args:
            loss: The chained critic's output.
            params: Actor network's parameters.
        """
        qs = loss
        actions = self.actions
        grads = tf.gradients(ys=qs, xs=actions)[0]

        if self.action_bounds is not None:
            # Follows Hausknecht et al 2016, two alternative implementations.
            lo, hi = self.action_bounds

            ## a)
            # Invert gradients if action outside bounds, additionally scale
            # gradients by closeness to bounds. Using this approach smoothes
            # the actions around their central (not average) value.
            # a = tf.cast(grads < 0, tf.float32) * ((lo - actions) / (hi - lo))
            # b = tf.cast(grads > 0, tf.float32) * ((actions - lo) / (hi - lo))
            # grads = grads * (a + b)

            ## b)
            # Invert gradients if pushing action further outside bounds.
            invert = tf.logical_or(
                tf.logical_and(actions > hi, grads < 0),
                tf.logical_and(actions < lo, grads > 0),
            )
            grads = tf.compat.v1.where(invert, -grads, grads)

        return tf.gradients(ys=actions, xs=params, grad_ys=grads)

    def get_config(self):
        return {}


class ActorCriticOptimizerRMSprop(tf.keras.optimizers.RMSprop, ActorCriticOptimizer):
    """Combines RMSProp with ActorCriticOptimizer.

    See :class:`ActorCriticOptimizer` for details.

    Args:
        actions: Output tensor of the actor.
        action_bounds: Lower and upper bounds for the actions.
        **kwargs: Argument passed to Rprop
    """

    def __init__(
        self,
        actions: tf.Tensor,
        action_bounds: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        super().__init__(name=self.__class__.__name__, **kwargs)
        self.actions = actions
        self.action_bounds = action_bounds


class ActorCriticOptimizerRprop(Rprop, ActorCriticOptimizer):
    """Combines Rprop Optimizer with the ActorCriticOptimizer

    See :class:`psipy.nn.optimizers.rprop.Rprop` and  :class:`ActorCriticOptimizer` for
    details.

    Args:
        actions: Output tensor of the actor.
        action_bounds: Lower and upper bounds for the actions.
        **kwargs: Argument passed to RMSprop
    """

    def __init__(
        self,
        actions: tf.Tensor,
        action_bounds: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        super().__init__(name=self.__class__.__name__, **kwargs)
        self.actions = actions
        self.action_bounds = action_bounds


def clone_model(model: tf.keras.Model, name: str) -> tf.keras.Model:
    """Clone a given :mod:`~tensorflow.keras.Model` with a new name.

    This creates new variables but keeps the same values. Training the original
    (or cloned) model will not affect the other.
    """
    model = tf.keras.models.clone_model(model)
    model._name = name
    return model


def reset_model(model: tf.keras.Model):
    """Runs all variable initializers in model's variables to reset them.

    WARNING: There is no official way to reset a model. The following code is
    claimed to work for Dense, Convolutional, and Recurrent layers ONLY.

    See https://github.com/keras-team/keras/issues/341#issuecomment-547833394 for
    more details.

    Args:
        model: Keras model to reset in-place.
    """
    for layer in model.layers:
        # If using a model as a layer, apply the functio recursively
        if isinstance(layer, tf.keras.Model):
            reset_model(layer)
            continue

        # Find initializers for the current layer
        if hasattr(layer, "cell"):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            # If not an initializer, skip it
            if "initializer" not in key:
                continue

            # Find the corresponding variable, like the kernel or the bias
            if key == "recurrent_initializer":  # special case check
                var = getattr(init_container, "recurrent_kernel")
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            # Assign the initializer, which reinitializes the layer
            var.assign(initializer(var.shape, var.dtype))

    ## Old Tensorflow 1.x code:
    # sess = tf.compat.v1.keras.backend.get_session()
    # sess.run([v.initializer for v in model.variables])
    # if hasattr(model, "optimizer"):
    #     sess.run([v.initializer for v in model.optimizer.weights])
    # return model


def reset_session() -> tf.compat.v1.Session:
    """Reset the global default Keras and Tensorflow session."""
    config = tf.compat.v1.ConfigProto(log_device_placement=False)
    # on1 = tf.compat.v1.OptimizerOptions.ON_1
    # config.graph_options.optimizer_options.global_jit_level = on1
    try:
        tf.compat.v1.get_default_session().close()
    except AttributeError:
        pass
    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        LOG.error(e)
    sess = tf.compat.v1.Session(config=config)
    sess.as_default()
    tf.compat.v1.keras.backend.set_session(sess)
    return sess
