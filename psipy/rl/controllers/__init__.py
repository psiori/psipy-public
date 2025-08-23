# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""PSIPY RL Controllers
====================================================

The following is an introduction to the included neural network based
controllers.

Algorithms
----------

NFQ
``````````

**Neural Fitted Q-Iteration**

:mod:`psipy.rl.controllers.nfq`

In NFQ, a neural q-function approximator is learned by iterativley fitting it
to cached Q-targets. The cached targets are updated over *iterations*, updating
the q-function in between over *epochs*. The q-network maps state-action pairs to
a single q-value::

    Q(state, action) -> q

The loss is the TD-Error, approximating the expected cost (or, in popular
literatur, reward) for any given ``(state, action)`` pair. The expected cost for
the next state is computed by greedily evaluating the q-network itself, using
the best action (expectedly producing minimal cost) given the next state::

    L = (Q(state, action) - (c(state) + γ * min_aQ(next_state, a)))^2

.. note::
    Riedmiller, Martin.
    `"Neural fitted Q iteration–first experiences with a data efficient neural
    reinforcement learning method."
    <https://link.springer.com/chapter/10.1007/11564096_32>`_
    European Conference on Machine Learning.
    Springer, Berlin, Heidelberg, 2005.


NFQCA
``````````

**Neural Fitted Q-Iteration with Continuous Actions**

:mod:`psipy.rl.control.nfqca`

NFQCA is an extension to :ref:`NFQ`. In contrast to :ref:`NFQ`, an explicit
policy network ``Pi`` (called *actor*) is used for producing the actions::

    π(state) -> u

That actor is fitted to maximize the expected value (minimize the expected cost)
of the action it generates according to the current q-network, also called
*critic* ::

    L = Q(state, π(state))

Instead of greedily evaluating the critic itself when training it, the actor's
output for the next state is used::

    L = (Q(state, action) - (c(state) + γ * Q(next_state, π(next_state))))^2

.. note::
    Hafner, Roland, and Martin Riedmiller.
    `"Reinforcement learning in feedback control."
    <https://link.springer.com/article/10.1007/s10994-011-5235-x>`_
    Machine learning 84.1-2 (2011): 137-169.

DDPG
``````````

**Deep Deterministic Policy Gradient**

DDPG is an extension to :ref:`NFQCA`, introducing training in a semi-online
fashion, performing minibatch stochastic gradient descent steps on both the
critic and actor in alternation to steps in the environment. In order to
support training on the continuously varying data distribution (as data is
continuously generated using the latest policy), the original paper also
adds *target networks*  for both critic and actor.

Generally, the changes introduced by DDPG to :ref:`NFQCA` are similar as the
changes introduced by DQN to :ref:`NFQ`.

As psipy-rl focuses on separation of data acquisition and training, and since we want to fully adapt the collect & infer paradigm, we have not integrated these 'inter-episode' updates into our variant of NFQ-CA / DDPG. We also will not adapt and do not believe in the target network idea. Instead, we calculate new targets in the data-space before sweeping over the full batch (in mini-batches), thus, not needing a 'frozen' network.

.. note::
    Lillicrap, Timothy P., et al.
    "Continuous control with deep reinforcement learning."
    arXiv preprint `arXiv:1509.02971 <https://arxiv.org/abs/1509.02971>`_
    (2015).

TD3
``````````

**Twin Delayed Deep Deterministic Policy Gradient**

TD3 is an extension to :ref:`DDPG`, introducing further adjustments in order to
make the *semi-online* training more stable.

1. Reduce overestimation bias by employing a second critic, training them
   concurrently but employing the less positive estimate for computing target
   values.

2. Add noise to the actions used for the expected q-values to produce a smoother
   q-estimate harder to be exploited by the policy updates.

3. Update critic more frequently than actor.

Adjustments 1. and 2. are implemented as extension to :ref:`NFQCA`.
Adjustment 3. has no relevance for :ref:`NFQCA` as both critic and actor
are trained to convergence independently from data collection.

.. note::
    Fujimoto, Scott, Herke van Hoof, and David Meger.
    "Addressing function approximation error in actor-critic methods."
    arXiv preprint `arXiv:1802.09477 <https://arxiv.org/abs/1802.09477>`_
    (2018).


Costs vs. Rewards
-----------------

**The following is currently conceptual only and not yet implemented in
:mod:`psipy.rl`!**

In control theory it is common to talk about costs, which often equal the
process's control deviation. In reinforcement learning research on the other
hand, there has been a switch to discussing rewards. That switch has been
somewhat timely correlated with researchers focusing on using computer game
simulations.

:mod:`psipy.rl` combines the two ideas, allowing the application of the
implemented solutions to both raw control problems where one optimizes a
control deviation and to computer games where no instantaneous cost is
available but only sparse rewards can be found. To enable this, cost functions
return values in ``[-1, 1]``. In regulator tasks, the best situation will be
to achieve 0 at all times -- there won't ever be any positive values or rewards,
just no costs.


Activations
-----------

.. todo::
    Discussion of what activations to use when? Primarily relu, tanh and
    sigmoid.

Exploration
-----------
The tradeoff between exploration and exploitation is at the core of reinforcement
learning. Do you take the action the algorithm thinks is best, or do something
else which may lead you to new states that you have no yet seen?  There are countless
methods to resolve this dilemma, most involving mixing in random actions in some way.
The way random actions are involved are different based on the action space.

For discrete action spaces, the most common form of exploration is called
epsilon-greedy. The algorithm starts with some epsilon value
:math:`\\epsilon \\in [0,1]`, which corresponds to the percentage of the actions
taken that will be randomly sampled from the action space.  Epsilon is then decreased
over time, either linearly or through other means, e.g. exponentially.

For continuous action spaces, one could generate random numbers within the action
range, however, there are smarter methods. Commonly, the algorithm is evaluated and
some noise is added to the action to change its "course". For example, DDPG uses
the Ornstein Uhlenbeck process to alter its actions when exploring. This process
is a mean reverting brownian motion.  Here, the epsilon greedy strategy would also
be used, but instead of randomly sampling, the noise is added.

"""


from psipy.rl.core.controller import ContinuousRandomActionController
from psipy.rl.core.controller import Controller, DiscreteRandomActionController
from psipy.rl.controllers.nfq import NFQ
from psipy.rl.controllers.nfqca import NFQCA
from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.controllers.noise import OrnsteinUhlenbeckActionNoise, RandomNormalNoise

__all__ = [
    "ContinuousRandomActionController",
    "Controller",
    "DiscreteRandomActionController",
    "NFQ",
    "NFQs",
    "NFQCA",
    "OrnsteinUhlenbeckActionNoise",
    "RandomNormalNoise",
]
