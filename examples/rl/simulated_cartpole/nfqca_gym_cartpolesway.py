# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
"""Learn to prevent swaying of a pole using NFQCA.

This example shows how to use NFQCA to continuously control a cart to maintain
a steady downwards facing pole.
"""
import glob
import os.path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl

from psipy.rl.control import ContinuousRandomActionController
from psipy.rl.control.nfq import tanh2
from psipy.rl.control.nfqca import NFQCA
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop
from psipy.rl.plant.gym.cartpole_plants import CartPoleContAction
from psipy.rl.plant.gym.cartpole_plants import CartPoleState
from psipy.rl.plant.gym.cartpole_plants import CartPoleSwayContinuousPlant
from psipy.rl.visualization.plotting_callback import PlottingCallback

# Define where we want to save our SART files
sart_folder = "./sart-cartpolesway-nfqca"


# Create a model based on state, action shapes and lookback
def make_actor(inputs, lookback):
    inp = tfkl.Input((inputs, lookback), name="state_actor")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(20, activation="tanh")(net)
    net = tfkl.Dense(20, activation="tanh")(net)
    net = tfkl.Dense(1, activation="tanh")(net)
    return tf.keras.Model(inp, net, name="actor")


def make_critic(inputs, lookback):
    inp = tfkl.Input((inputs, lookback), name="state_critic")
    act = tfkl.Input((1,), name="act_in")
    net = tfkl.Concatenate()([tfkl.Flatten()(inp), act])
    net = tfkl.Dense(20, activation="tanh")(net)
    net = tfkl.Dense(20, activation="tanh")(net)
    net = tfkl.Dense(1, activation="sigmoid")(net)
    return tf.keras.Model([inp, act], net, name="critic")


# Define a custom cost function to change the inbuilt costs
def costfunc(states: np.ndarray) -> np.ndarray:
    angles = states[:, 2]
    angle_cost = tanh2(angles, C=0.01, mu=(np.pi / 360) * 15)
    costs = angle_cost

    # Prevent exploiting swing direction changes
    # angle_speed = states[:, 3]
    # costs += tanh2(angle_speed, C=0.01, mu=5)

    distance = states[:, 0]
    costs += tanh2(distance, C=0.04, mu=0.05)

    return costs


# Create a function to make new episodes (if desired)
def create_fake_episodes(folder, lookback):
    # The goal state is arbitrary when exploring randomly
    # We can just set it wherever and by that produce fake trajectories
    more_episodes = []

    kwargs = dict(lookback=lookback)
    for path in glob.glob(f"{folder}/*.h5"):
        with SARTReader(path) as reader:
            o, a, t, c = reader.load_full_episode()

        master_o = o.copy()

        more_episodes.append(Episode(o, a, t, c, **kwargs))

        o = master_o.copy()
        o[:, 0, ...] = np.mean(o[:, 0, ...]) - o[:, 0, ...]
        more_episodes.append(Episode(o, a, t, c, **kwargs))

        o = master_o.copy()
        o[:, 0, ...] = np.mean(o[:, 0, ...]) - o[:, 0, ...] + 0.5
        more_episodes.append(Episode(o, a, t, c, **kwargs))

        o = master_o.copy()
        o[:, 0, ...] = np.mean(o[:, 0, ...]) - o[:, 0, ...] - 0.5
        more_episodes.append(Episode(o, a, t, c, **kwargs))

        o = master_o.copy()
        o[:, 0, ...] = np.mean(o[:, 0, ...]) - o[:, 0, ...] - 1
        more_episodes.append(Episode(o, a, t, c, **kwargs))

        o = master_o.copy()
        o[:, 0, ...] = np.mean(o[:, 0, ...]) - o[:, 0, ...] + 1
        more_episodes.append(Episode(o, a, t, c, **kwargs))

        o = master_o.copy()
        o[:, 0, ...] = c
        more_episodes.append(Episode(o, a, t, c, **kwargs))

        # # Add episode full of goal states
        # o = o.copy()
        # a = a.copy()
        # o[:, 0] = 0
        # o[:, 1] = 0
        # o[:, 2] = 180
        # o[:, 3] = 0
        # a[:] = 0
        # more_episodes.append(Episode(o, a, t, c, **kwargs))

        return more_episodes


# Create some placeholders so lines don't get too long
plant = CartPoleSwayContinuousPlant()
ActionType = CartPoleContAction
StateType = CartPoleState
lookback = 3

# Collect initial data with a continuous random action controller
explorer = ContinuousRandomActionController(StateType.channels(), ActionType)
loop = Loop(plant, explorer, "CartPoleSway", sart_folder)
loop.run(50)

# Make the NFQCA model
actor = make_actor(len(StateType.channels()), lookback)
critic = make_critic(len(StateType.channels()), lookback)
nfqca = NFQCA(
    actor=actor,
    critic=critic,
    state_channels=StateType.channels(),
    action=ActionType,
    lookback=lookback,
    td3=True,
)

# Load the collected data
batch = Batch.from_hdf5(sart_folder, lookback=lookback, control=nfqca)
# Create fake episodes and append them to the batch
fakes = create_fake_episodes(sart_folder, lookback)
batch.append(fakes)
# Fit the normalizer
nfqca.fit_normalizer(batch.observations, method="std")

callbacks1 = [
    PlottingCallback(
        ax1="q",
        is_ax1=lambda x: x.endswith("q") or x == "loss",
        ax2="mse",
        is_ax2=lambda x: "act" in x,
        title="Critic",
    )
]

callbacks2 = [
    PlottingCallback(
        ax1="act",
        is_ax1=lambda x: "act" in x,
        ax2="mse",
        is_ax2=lambda x: x == "loss",
        title="Actor",
    )
]

try:
    for cycle in range(120):
        print(f"At iteration {cycle+1}!")
        nfqca.fit_critic(
            batch,
            costfunc=costfunc,
            iterations=5,
            epochs=1,
            minibatch_size=500,
            gamma=0.999,
            callbacks=callbacks1,
        )
        nfqca.fit_actor(batch, epochs=1, minibatch_size=500, callbacks=callbacks2)
except KeyboardInterrupt:
    pass

nfqca.save("cartpolesway-nfqca-model.zip")

loaded = NFQCA.load("cartpolesway-nfqca-model.zip", custom_objects=[CartPoleContAction])
loop = Loop(
    plant, nfqca, "CartPoleSwayEval", os.path.join("live", sart_folder), render=True
)
loop.run(episodes=100, max_episode_steps=500)
