

"""Example script that tests multi dimensinonal actions of NFQs on a simple wumpus world."""

import sys

import tensorflow as tf
from tensorflow.keras import layers as tfkl

from typing import Callable, Type
import numpy as np

from psipy.rl.core.controller import Controller, DiscreteRandomActionController
from psipy.rl.core.plant import Plant, State, Action
from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop
from psipy.rl.plants.simulated.wumpus import (
    WumpusAction,
    WumpusState,
    WumpusPlant)
from psipy.rl.visualization.plotting_callback import PlottingCallback

# Parameters
RENDER = True           # whether or not to render the plant during training

NUM_EPISODES = 200
NUM_EPISODE_STEPS = 30  # Increased steps since we have pits to avoid
GAMMA = 0.98
STACKING = 1            # history length. 1 = no stacking, just the current state.
EPSILON = 0.20          # epsilon-greedy exploration

STATE_CHANNELS = [
    "x",
    "y",
]

ACTION_CHANNELS = [
    "move_x",
    "move_y",
]

SART_FOLDER = "psidata-nfqs-sart-wumpus-train"  # Define where we want to save our SART files

def make_model(n_inputs, n_action_dims, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((n_action_dims,), name="actions")  # Changed from (1,) to (n_action_dims,)
    net = tfkl.Flatten()(inp)
    net = tfkl.concatenate([act, net])
    print("ACT SHAPE", act.shape)
    print("INP SHAPE", inp.shape)
    print("NET INPUT SHAPE", net.shape)
    # net = tfkl.Dense(n_inputs * lookback * 20, activation="relu")(net) # add this layer if you remove velocities from the state
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(1, activation="sigmoid")(net)
    model = tf.keras.Model([inp, act], net)
    model.summary()
    return model


#class RandomControl(Controller):
#    def __init__(self, action_type: Type[Action]):
#        self.action_type = action_type
#        self.action_channels = action_type.channels
    
#    def get_action(self, state: State) -> Action:
#        return self.action_type(
#        )
    
#    def notify_episode_starts(self) -> None:
#        pass
    
#    def notify_episode_stops(self) -> None:
#        pass


ActionType = WumpusAction
StateType = WumpusState

Plant = WumpusPlant(pit_positions=())


# Make the NFQ model - now with correct action dimensions
model = make_model(len(STATE_CHANNELS), len(ACTION_CHANNELS), STACKING)

#loop = Loop(Plant, random_control, f"Wumpus", SART_FOLDER, render=RENDER)
#loop.run(episodes=NUM_EPISODES, max_episode_steps=NUM_EPISODE_STEPS)


nfqs = NFQs(
    model=model,
    state_channels=STATE_CHANNELS,
    action=ActionType,
#    action_values=ActionType.legal_values,
    optimizer=tf.keras.optimizers.Adam(),
    lookback=STACKING,
    num_repeat=1,
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

loop = Loop(Plant, nfqs, f"Wumpus", SART_FOLDER, render=RENDER)

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

