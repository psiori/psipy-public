# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
#
# Authors:
#   Saskia Bruhn <saskia.bruhn@psiori.com>, March 2020

"""RPROP, an optimizer for full batch learning
==============================================

RPROP was introduced in the paper "A Direct Adaptive Method for Faster Backpropagation
Learning: The RPROP Algorithm" (Riedmiller, Braun, 1993). In the following the variant
from this paper is referred to as :math:`RPROP^+`.

There are 4 different variants of the RPROP algorithm:
:math:`RPROP^+`, :math:`RPROP^-`, :math:`iRPROP^-` and :math:`iRPROP^+`

:math:`RPROP^-` is described in the paper "Advanced Supervised Learning in Multi-layer
Perceptrons - From Backpropagation to Adaptive Learning Algorithms" (Riedmiller, 1994)

The variants :math:`iRPROP^-` and  :math:`iRPROP^+` are introduced in the paper
"Improving The Rrpop Learning Algorithm" (Igel, Hüsken, 2000)

The paper "Empirical evaluation of the improved RPROP learning algorithms"
(Igel, Hüsken, 2003). gives a nice overview over all the four variants of the algorithm.

The class :class:`Rprop` implements the two variants :math:`RPROP^+` and
:math:`iRPROP^+`.


Module overview
---------------

.. autosummary::

    Rprop
    RpropPlus
    iRpropPlus


:math:`RPROP^-`
---------------

:math:`RPROP^-` optimizes the weights based on the directions of
their gradients, but not their magnitudes. In each iteration every weight is
updated in the opposite direction of its gradient.

.. math::
    w_{ij}^{(t+1)} = w_{ij}^{(t)} + \\Delta w_{ij}^{(t)}

with

.. math::
    \\Delta w_{ij}^{(t)} = - sign(\\frac{\\delta E}{\\delta w_{ij}}^{(t)}) *
    \\Delta_{ij}^{(t)}

The step size of this update is small in the beginning. As long as the gradient does
not change its direction, the step size is increased in each iteration by the factor
:math:`\\eta^+`. If the gradient changes its direction, which means the minimum has been
crossed, the step size is decreased by the factor :math:`\\eta^-`. If the gradient
equals 0, the step size stays the same as before.
To make sure that the step size does not become too big or too small, a minimum
(:math:`\\Delta_{min}`) and a maximum (:math:`\\Delta_{max}`) step size is used.

.. math::
    \\Delta_{ij}^{(t)} = \\begin{cases}
                            min(\\eta^+ *&\\Delta_{ij}^{(t-1)}, \\Delta_{max}) &,&
                            \\text{if } \\frac{\\delta E}{\\delta w_{ij}}^{(t-1)} *
                            \\frac{\\delta E}{\\delta w_{ij}}^{(t)} > 0 \\\\
                            max(\\eta^- *&\\Delta_{ij}^{(t-1)}, \\Delta_{min}) &,&
                            \\text{if } \\frac{\\delta E}{\\delta w_{ij}}^{(t-1)} *
                            \\frac{\\delta E}{\\delta w_{ij}}^{(t)} < 0 \\\\
                            &\\Delta_{ij}^{(t-1)} &,& \\text{else}
                         \\end{cases}

This is also the basis for the other RPROP variants. All variants build up on the
:math:`RPROP^-` algorithm. What distinguishes them is the way they deal with the change
of direction of the gradient.


:math:`RPROP^+`
---------------

:math:`RPROP^+` extends the :math:`RPROP^-` algorithm with weight-backtracking. When the
minimum has been crossed, it goes back the last step size and sets the gradient to zero
to avoid the change of direction being punished again in the next step.

.. math::
    \\Delta w_{ij}^{(t)} = - \\Delta w_{ij}^{(t-1)}

.. math::
    \\frac{\\delta E}{\\delta w_{ij}}^{(t-1)} := 0


:math:`iRPROP^+`
----------------

:math:`iRPROP^+` extends the :math:`RPROP^-` algorithm with error dependent
weight-backtracking. It takes into account the fact that the change of direction of the
gradient does not indicate whether the the error has increased by the last weight
update. Therefore it is reasonable to make the step reversal dependent on the
development of the error. This means:

.. math::
    \\Delta w_{ij}^{(t)} = \\begin{cases}
                                - \\Delta w_{ij}^{(t-1)} &,&
                                \\text{if } E^{(t)} > E^{(t-1)} \\\\
                                0 &,& \\text{else}
                           \\end{cases}

.. math::
    \\frac{\\delta E}{\\delta w_{ij}}^{(t-1)} := 0


:math:`iRPROP^-`
----------------

The last variant :math:`iRPROP^-` extends the :math:`RPROP^-` algorithm only by
setting the gradient to zero if the minimum has been crossed.

.. math::
    \\frac{\\delta E}{\\delta w_{ij}}^{(t-1)} := 0


"""

import numpy
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer


class Rprop(Optimizer):
    """RPROP is an optimizer for full batch learning.

    This class implements the two variants :math:`RPROP^+` and :math:`iRPROP^+`.
    They can be called by their respective subclass.

    Example::

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ])
        model.compile(
            optimizer=RpropPlus(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

    Args:
        init_stepsize: Initial step size in direction of gradient.
        scale_up: Factor by which the step size is multiplied if it is increased.
        scale_down: Factor by which the step size is multiplied if it is decreased.
        min_stepsize: Minimal step size, used when step size decreased by scale_down is
            smaller.
        max_stepsize: Maximal step size, used when step size increased by scale_up is
            bigger.
        error_dependent:

            - ``True`` :math:`\\rightarrow` iRPROP+, see :class:`iRpropPlus`
            - ``False`` :math:`\\rightarrow` RPROP+, see :class:`iRpropPlus`

    """

    def __init__(
        self,
        init_stepsize: float = 1e-3,
        scale_up: float = 1.2,
        scale_down: float = 0.5,
        min_stepsize: float = 1e-6,
        max_stepsize: float = 50.0,
        error_dependent: bool = False,
        name: str = "Rprop",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.init_stepsize = init_stepsize
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_stepsize = min_stepsize
        self.max_stepsize = max_stepsize
        self.error_dependent = error_dependent

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        stepsizes = [
            K.variable(numpy.ones(shape) * self.init_stepsize) for shape in shapes
        ]
        old_grads = [K.zeros(shape) for shape in shapes]
        old_loss = K.zeros(tf.shape(loss))
        prev_weight_deltas = [K.zeros(shape) for shape in shapes]
        self._weights = stepsizes + old_grads
        updates = []

        for param, grad, old_grad, prev_weight_delta, stepsize in zip(
            params, grads, old_grads, prev_weight_deltas, stepsizes
        ):
            # If gradient did not switch sign, increase stepsize or use max_stepsize,
            # elif gradient switched sign, decrease stepsize or use min_stepsize, else
            # old_grad*grad is zero, keep stepsize
            new_stepsize = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(stepsize * self.scale_up, self.max_stepsize),
                K.switch(
                    K.less(grad * old_grad, 0),
                    K.maximum(stepsize * self.scale_down, self.min_stepsize),
                    stepsize,
                ),
            )

            # new_delta makes it go the step_size in the opposite direction of the
            # gradient if the gradient is grater or less than 0. If the gradient equals
            # 0 it is set to 0.
            new_delta = K.switch(
                K.greater(grad, 0),
                -new_stepsize,
                K.switch(K.less(grad, 0), new_stepsize, K.zeros_like(new_stepsize)),
            )

            # iRPROP+: If gradient did not change sign, take new_delta
            #          elif gradient changed sign, check if loss increased
            #               If loss increased, take -prev_weight_delta
            #               elif loss did not increase take zeros_like(new_delta)
            #          elif grad * old_grad = 0 take take zeros_like(new_delta)
            if self.error_dependent:
                # result of loss comparison in case gradient changed its direction
                loss_comparison = K.switch(
                    K.greater(loss, old_loss),
                    -prev_weight_delta,
                    K.zeros_like(new_delta),
                )

                weight_delta = K.switch(
                    K.greater(grad * old_grad, 0),
                    new_delta,
                    K.switch(
                        K.less(grad * old_grad, 0),
                        loss_comparison,
                        K.zeros_like(new_delta),
                    ),
                )

            # RPROP+: If sign of gradient changed, take previous weight_delta and go
            # back, else take new_delta
            else:
                weight_delta = K.switch(
                    K.less(grad * old_grad, 0), -prev_weight_delta, new_delta
                )

            # add weight_delta to param
            new_param = param + weight_delta

            # reset gradient to 0 if sign of gradient has changed to avoid double
            # punishment
            grad = K.switch(K.less(grad * old_grad, 0), K.zeros_like(grad), grad)

            updates.extend(
                [
                    K.update(param, new_param),
                    K.update(stepsize, new_stepsize),
                    K.update(old_grad, grad),
                    K.update(old_loss, loss),
                    K.update(prev_weight_delta, weight_delta),
                ]
            )

        return updates

    def get_config(self):
        config = {
            "init_step_size": self.init_step_size,
            "scale_up": self.scale_up,
            "scale_down": self.scale_down,
            "min_step_size": self.min_step_size,
            "max_step_size": self.max_step_size,
            "error_dependent": self.error_dependent,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class RpropPlus(Rprop):
    """:math:`RPROP^+` optimizer, :class:`Rprop` without ``error_dependent``."""

    def __init__(
        self,
        init_stepsize: float = 1e-3,
        scale_up: float = 1.2,
        scale_down: float = 0.5,
        min_stepsize: float = 1e-6,
        max_stepsize: float = 50.0,
        name: str = "RpropPlus",
    ):
        super().__init__(
            init_stepsize=init_stepsize,
            scale_up=scale_up,
            scale_down=scale_down,
            min_stepsize=min_stepsize,
            max_stepsize=max_stepsize,
            error_dependent=False,
            name=name,
        )


class iRpropPlus(Rprop):
    """:math:`iRPROP^+` optimizer, :class:`Rprop` with ``error_dependent``."""

    def __init__(
        self,
        init_stepsize: float = 1e-3,
        scale_up: float = 1.2,
        scale_down: float = 0.5,
        min_stepsize: float = 1e-6,
        max_stepsize: float = 50.0,
        name: str = "iRpropPlus",
    ):
        super().__init__(
            init_stepsize=init_stepsize,
            scale_up=scale_up,
            scale_down=scale_down,
            min_stepsize=min_stepsize,
            max_stepsize=max_stepsize,
            error_dependent=True,
            name=name,
        )
