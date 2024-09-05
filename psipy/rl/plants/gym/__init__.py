# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""

.. todo::

    Module description. How and why do we wrap openai gym?

"""

from psipy.rl.plants.gym.cartpole_plants import CartPolePlant
from psipy.rl.plants.gym.cartpole_plants import CartPoleUnboundedContinuousPlant
from psipy.rl.plants.gym.cartpole_plants import CartPoleUnboundedPlant
from psipy.rl.plants.gym.cartpole_plants import CartPoleAssistedBalancePlant
from psipy.rl.plants.gym.envs import *  # noqa
from psipy.rl.plants.gym.gym_plant import GymPlant

__all__ = [
    "GymPlant",
    "CartPolePlant",
    "CartPoleUnboundedPlant",
    "CartPoleUnboundedContinuousPlant",
    "CartPoleAssistedBalancePlant",
]
