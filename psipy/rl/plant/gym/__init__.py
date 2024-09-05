# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""

.. todo::

    Module description. How and why do we wrap openai gym?

"""

from psipy.rl.plant.gym.cartpole_plants import CartPolePlant
from psipy.rl.plant.gym.cartpole_plants import CartPoleUnboundedContinuousPlant
from psipy.rl.plant.gym.cartpole_plants import CartPoleUnboundedPlant
from psipy.rl.plant.gym.cartpole_plants import CartPoleAssistedBalancePlant
from psipy.rl.plant.gym.envs import *  # noqa
from psipy.rl.plant.gym.gym_plant import GymPlant

__all__ = [
    "GymPlant",
    "CartPolePlant",
    "CartPoleUnboundedPlant",
    "CartPoleUnboundedContinuousPlant",
    "CartPoleAssistedBalancePlant",
]
