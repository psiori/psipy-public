# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Example script that learns cartpole with NFQs."""

import sys

import tensorflow as tf
from tensorflow.keras import layers as tfkl

from typing import Callable
import numpy as np

from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop
from psipy.rl.plants.simulated.cartpole import (
    CartPoleBangAction,
    CartPoleState,
    CartPole)
from psipy.rl.visualization.plotting_callback import PlottingCallback

# Parameters
RENDER = True           # whether or not to render the plant during training

NUM_EPISODES = 200
NUM_EPISODE_STEPS = 400
GAMMA = 0.98
STACKING = 1            # history length. 1 = no stacking, just the current state.
EPSILON = 0.05          # epsilon-greedy exploration

DEFAULT_STEP_COST = 0.01
TERMINAL_COST = 1.0     # this is the cost for leaving the track.

STATE_CHANNELS = [
    "cart_position",
    "cart_velocity",
    # "pole_angle",     # use this instead of sine/cosine if you want to use the angle directly. Remeber to adapt the cost function...
    "pole_sine",
    "pole_cosine",
    "pole_velocity",
    # "move_ACT",       # use, if stacking > 1
]
ACTION_CHANNELS = [
    "move",
]

SART_FOLDER = "sart-cartpole-train"  # Define where we want to save our SART files
X_THRESHOLD = 3.6                    # (half) cart track length. Provide more space than balancing standard for swingup.

def make_model(n_inputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((1,), name="actions")
    net = tfkl.Flatten()(inp)
    net = tfkl.concatenate([act, net])
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(1, activation="sigmoid")(net)
    return tf.keras.Model([inp, act], net)

def make_cost_function(x_threshold: float = 3.6,
                       position_idx=None,
                       cosine_idx=None) -> Callable[[np.ndarray], np.ndarray]:
    def cost_function(state: np.ndarray) -> np.ndarray:

        position = state[:, position_idx]
        cosine = state[:, cosine_idx]

        # ZERO COSTS in center of track with pole pointing upwards.
        # Within the center region of the track, costs are SHAPED
        # from DEFAULT_STEP_COST (in center, but pole pointing downwards),
        # to 0.0 (in center, but pole pointing upwards) using the cosine
        # of the pole angle.
        costs = (1.0-(cosine+1.0)/2.0) * DEFAULT_STEP_COST

        # DEFAULT COSTS for positions outside the center region of the track.
        costs[abs(position) >= 0.2 * x_threshold] = DEFAULT_STEP_COST

        # VERY HIGH TERMINAL COSTS for leaving the track.
        costs[abs(position) >= x_threshold]       = TERMINAL_COST

        return costs

    return cost_function

cost_function = make_cost_function(x_threshold=X_THRESHOLD,
                                   position_idx=STATE_CHANNELS.index("cart_position"),
                                   cosine_idx=STATE_CHANNELS.index("pole_cosine"))

ActionType = CartPoleBangAction
StateType = CartPoleState

Plant = CartPole(x_threshold=X_THRESHOLD,
                 cost_function=CartPole.cost_func_wrapper(
                                     cost_function,
                                     STATE_CHANNELS))  

# Make the NFQ model
model = make_model(len(STATE_CHANNELS), STACKING)
nfqs = NFQs(
    model=model,
    state_channels=STATE_CHANNELS,
    action=ActionType,
    action_values=ActionType.legal_values[0],
    optimizer=tf.keras.optimizers.Adam(),
    lookback=STACKING,
    num_repeat=1
)

callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="ME", is_ax2=lambda x: x.endswith("qdelta")
)

batch = None
episode = 0

try:
    batch = Batch.from_hdf5(
        SART_FOLDER,
        state_channels=STATE_CHANNELS,
        action_channels=ACTION_CHANNELS,
        lookback=STACKING,
        control=nfqs,
    )
    print(f"SUCCESSFULLY LOADED episodes from {SART_FOLDER}: {len(batch._episodes)}. I will use this data.")

    nfqs.fit_normalizer(batch.observations, method="meanstd")
    episode = len(batch._episodes)
except FileNotFoundError:
    print(f"No episodes found in {SART_FOLDER}. I will start from scratch.")

loop = Loop(Plant, nfqs, f"CartPole", SART_FOLDER, render=RENDER)

nfqs.epsilon = EPSILON

while episode < NUM_EPISODES:
    print(f"Episode {episode} of {NUM_EPISODES}")

    loop.run_episode(episode, max_steps=NUM_EPISODE_STEPS)

    batch = Batch.from_hdf5(
        SART_FOLDER,
        state_channels=STATE_CHANNELS,
        action_channels=ACTION_CHANNELS,
        lookback=STACKING,
        control=nfqs,
    )

    nfqs.fit_normalizer(batch.observations, method="meanstd")  
    # refit the input scaling to the newly collected data after each episode.
    # although riedmiller argues in his tips and tricks book chapter, that, 
    # for his case, doing this is unproblematic, this is a little bit risky, 
    # in our case, as we do not reset the neural network weights
    # after each epsiode but continue with the previous network. Thus, this
    # does "move" the data below the current network slightly. In practice, it 
    # becomes usually negligible after a number of episodes. If you want to be safe,
    # stop refitting after some time, but not too early, to make sure you
    # have collected data from all parts of the state space (pole upward, high
    # velocities).

    try:
        nfqs.fit(
            batch,
            costfunc=cost_function,
            iterations=4,
            epochs=8,
            minibatch_size=2048,
            gamma=GAMMA,
            callbacks=[callback],
            verbose=1,
        )
    except KeyboardInterrupt:
        pass

    episode += 1
