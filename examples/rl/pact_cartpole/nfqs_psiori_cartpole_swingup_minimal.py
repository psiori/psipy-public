# Copyright (C) PSIORI GmbH, Germany
# Authors: Sascha Lange

"""Minimal example script that learns to swingup and balance the cartpole with NFQ using three discrete actions (-max, 0, max) to chose from. Does not expect any parameters from the command line but uses the hyper paramters described in the NFQ 2.0 paper."""

import sys

import tensorflow as tf
from tensorflow.keras import layers as tfkl

from typing import Callable
import numpy as np

from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop
from psipy.rl.plants.real.pact_cartpole.cartpole import (
    SwingupContinuousDiscreteAction,
    SwingupPlant,
    SwingupState,
)
from psipy.rl.visualization.plotting_callback import PlottingCallback

# Parameters
RENDER = True           # whether or not to render the plant during training

NUM_EPISODES = 400
NUM_EPISODE_STEPS = 400
GAMMA = 0.98
STACKING = 6            # history length. 1 = no stacking, just the current state.
EPSILON = 0.1           # epsilon-greedy exploration

DEFAULT_STEP_COST = 0.01
TERMINAL_COST = 1.0     # this is the cost for leaving the track. ATTENTION: it needs to be high enough to prevent creating a "shortcut" for the agent; it needs to be higher than the accumulated discounted step costs for the steps going to infinity. For a gamma of 0.98, the geometric series converges to 50x the step cost (as lim(t to infinity) of sum(gamma^t) = 50). We choose the terminal costs to be 100 times the step cost to be on the safe side and make it easy for the agent to understand there is no benefit in leaving the track.

STATE_CHANNELS = [
    "cart_position",
    "cart_velocity",
    # "pole_angle",     # use this instead of sine/cosine if you want to use the angle directly. Remeber to adapt the cost function...
    "pole_sine",
    "pole_cosine",
    "pole_velocity",
    "direction_ACT",       # use, if stacking > 1
]
ACTION_CHANNELS = [
    "direction",
]

SART_FOLDER = "psidata-nfqs-sart-cartpole-train"  # Define where we want to save our SART files

def make_model(n_inputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((1,), name="actions")
    net = tfkl.Flatten()(inp)
    net = tfkl.concatenate([act, net])
    # net = tfkl.Dense(n_inputs * lookback * 20, activation="relu")(net) # add this layer if you remove velocities from the state
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(1, activation="sigmoid")(net)
    model = tf.keras.Model([inp, act], net)
    model.summary()
    return model

def make_sparse_cost_func(position_idx: int=0,
                          cosine_idx: int=3,
                          step_cost: float=0.01,
                          use_cosine: bool=True,
                          use_upright_margin: bool=False,
                          upright_margin: float=0.3,
                          xminus: bool=True) -> Callable[[np.ndarray], np.ndarray]:
    # Define a custom cost function to change the inbuilt costs
    def sparse_costfunc(states: np.ndarray) -> np.ndarray:
        center = (SwingupPlant.LEFT_SIDE + SwingupPlant.RIGHT_SIDE) / 2.0
        margin = abs(SwingupPlant.RIGHT_SIDE - SwingupPlant.LEFT_SIDE) / 2.0 * 0.3  # 30% of distance from center to hard endstop

        position = states[:, position_idx] 
        cosine = states[:, cosine_idx]       

        if isinstance(cosine, np.ndarray):
            costs = np.zeros(cosine.shape)
        else:
            costs = 0.0

        if use_cosine:
            costs = (1.0-(cosine+1.0)/2.0) * step_cost  # shaping of costs in goal area to reward low pole angle deviations from upright position

        if use_upright_margin:
            costs[1.0-(cosine+1.0)/2.0 > upright_margin] = step_cost    

        costs[abs(position - center) >= margin] = step_cost 

        if xminus:  # non-terminal bad area
            costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET * 2] = step_cost * 5
            costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET * 2] = step_cost * 5

        # terminal bad area
        costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET] = 1.0
        costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET] = 1.0

        return costs

    return sparse_costfunc


cost_function = make_sparse_cost_func(
    position_idx=STATE_CHANNELS.index("cart_position"),
    cosine_idx=STATE_CHANNELS.index("pole_cosine"),
    step_cost=DEFAULT_STEP_COST,
    use_cosine=True,
    use_upright_margin=False,
    upright_margin=0.3,
    xminus=True)

ActionType = SwingupContinuousDiscreteAction
StateType = SwingupState

SwingupPlant(
        hostname="127.0.0.1",   # if you run the script on your computer, not the blue box, change this to the IP of the blue box (e.g. 192.168.177.145)
        hilscher_port="5555",
        sway_start=False,
        cost_function=SwingupPlant.cost_func_wrapper(
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
    
    if episode % 10 == 0 and episode < NUM_EPISODES / 2: # read NFQ 2.0 paper for more details, rationale and (bad, but negligible) effects of refitting the normalizer "continuously"
        nfqs.fit_normalizer(batch.observations, method="meanstd")
        print("Refit the normalizer again using meanstd.")

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
