# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Custom OpenAI Gym environments, using their API.

Custom environments are useful for testing out controllers in simulation on problems which
are similar to already created gym environments. For example, cart pole swing up is a simple
alteration to the CartPole-v0 environment.

All custom environments must be registered with the OpenAI Gym environment registry in order
to be loaded. This is done once on module initialization.

These environments are not to be used directly but wrapped by the plants
in :mod:`psipy.rl.plant.gym`.
"""

import logging

import gymnasium as gym
from gymnasium.envs.registration import register

LOG = logging.getLogger(__name__)

try:
    register(
        id="CartPole-v2",
        entry_point="psipy.rl.plant.gym.envs.cartpole:CartPoleV2Env",
        max_episode_steps=1000,  # mock, because of gym setting done=True after these
        reward_threshold=195.0,
    )
    register(
        id="CartPoleUnbounded-v0",
        entry_point="psipy.rl.plant.gym.envs.cartpole:CartPoleUnboundedEnv",
        max_episode_steps=1000,  # mock, because of gym setting done=True after these
        reward_threshold=195.0,
    )
    register(
        id="CartPoleUnboundedContinuous-v0",
        entry_point="psipy.rl.plant.gym.envs.cartpole:CartPoleUnboundedContinuousEnv",
        max_episode_steps=1000,  # mock, because of gym setting done=True after these
        reward_threshold=195.0,
    )
    register(
        id="CartPoleAssistedBalance-v0",
        entry_point="psipy.rl.plant.gym.envs.cartpole:CartPoleAssistedBalanceEnv",
        max_episode_steps=1000,
        reward_threshold=195.0,
    )
except gym.error.Error as e:
    LOG.warning(e)
