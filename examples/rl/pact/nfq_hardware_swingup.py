# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Example script that learns Cartpole Swingup with NFQ on the Swingup Hardware.

In order to connect, you need to follow the instructions in :class:`SwingupPlant`.
"""

import glob
import time

import numpy as np
import tensorflow as tf
from psipy.rl.control.nfq import NFQ, tanh2
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop, LoopPrettyPrinter
from psipy.rl.visualization.plotting_callback import PlottingCallback
from tensorflow.keras import layers as tfkl

from cartpole_control.plant.cartpole_plant import (
    SwingupDiscretizedAction,
    SwingupPlant,
    SwingupState,
)

# Define where we want to save our SART files
sart_folder = "swingup4-retrain"
net_name = sart_folder + "/latest_net.zip"

STATE_CHANNELS = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_velocity",
    "direction_ACT",
)
THETA_CHANNEL_IDX = STATE_CHANNELS.index("pole_angle")
EPS_STEPS = 200


# Create a model based on state, action shapes and lookback
def make_model_nfqs(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((1,), name="action")
    net = tfkl.Flatten()(inp)
    net = tfkl.Concatenate()([net, act])
    net = tfkl.Dense(20, activation="tanh")(net)
    net = tfkl.Dense(20, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
    return tf.keras.Model([inp, act], net)


# Create a model based on state, action shapes and lookback
def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="state")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(30, activation="tanh")(net)
    net = tfkl.Dense(30, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
    return tf.keras.Model(inp, net)


# Define a custom cost function to change the inbuilt costs
def costfunc(states: np.ndarray) -> np.ndarray:
    position = states[:, 0]
    theta = states[:, THETA_CHANNEL_IDX]
    theta_speed = states[:, THETA_CHANNEL_IDX + 1]

    costs = tanh2(theta, C=0.01, mu=0.5)
    costs += tanh2(theta_speed, C=0.01, mu=2.5)
    costs[position < SwingupPlant.LEFT_SIDE + 20] = 1
    costs[position > SwingupPlant.RIGHT_SIDE - 20] = 1

    return costs


num = 1000
rando_positions = np.random.randint(
    SwingupPlant.LEFT_SIDE + 200,  # Do not go into terminal area
    SwingupPlant.RIGHT_SIDE - 200,  # Do not go into terminal area
    num,
)


# Create a function to make new episodes (if desired)
def create_fake_episodes(folder: str, lookback: int):
    kwargs = dict(lookback=lookback)
    for path in glob.glob(f"{folder}/*.h5"):
        try:
            with SARTReader(path) as reader:
                o, a, t, c = reader.load_full_episode(
                    state_channels=STATE_CHANNELS, action_channels=("direction_index",),
                )
            a_ = a.copy()
            a_[a == 1] = 0
            a_[a == 0] = 1
            o[:, 0] -= SwingupPlant.CENTER
            o = -1 * o
            o[:, 0] += SwingupPlant.CENTER
            yield Episode(o, a_, t, c, **kwargs)
        except (KeyError, OSError):
            continue

    kwargs = dict(lookback=lookback)

    # According to NFQ Tricks, add goal states with all 3 actions
    # These are added 200 times
    print(f"Creating {num} hints to goal.")
    print("These are only calculated for 3-act!")
    o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    o[:, 0] = rando_positions
    t = np.zeros(num)
    a = np.ones(num)
    o[:, -1] = -1  # Do opposite of current action
    c = np.zeros(num)
    yield Episode(o, a, t, c, **kwargs)
    o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    o[:, 0] = rando_positions
    t = np.zeros(num)
    a = np.zeros(num)
    o[:, -1] = 1  # Do opposite of current action
    c = np.zeros(num)
    yield Episode(o, a, t, c, **kwargs)
    o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    o[:, 0] = rando_positions
    t = np.zeros(num)
    a = np.ones(num) * 2  # 0 is the 2nd index
    c = np.zeros(num)
    yield Episode(o, a, t, c, **kwargs)
    # o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    # t = np.zeros(num)
    # a = np.ones(num) * 2 # 0 is the 2nd index
    # c = np.zeros(num)
    # yield Episode(o, a, t, c, **kwargs)


plant = SwingupPlant("5555", speed_value=800, speed_value2=400)
ActionType = SwingupDiscretizedAction
StateType = SwingupState
lookback = 2
gamma = 0.999
max_episode_length = EPS_STEPS

load_network = False
initial_fit = False


# Make the NFQ model
if not load_network:
    model = make_model(len(STATE_CHANNELS), len(ActionType.legal_values[0]), lookback,)
    nfq = NFQ(
        model=model,
        state_channels=STATE_CHANNELS,
        action_channels=("direction",),
        action=ActionType,
        action_values=ActionType.legal_values[0],
        lookback=lookback,
        num_repeat=5,
    )
else:
    nfq = NFQ.load(net_name, custom_objects=[ActionType])

nfq.epsilon = 0.0
nfq.epsilon = 0.5

callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "loss"
)

loop = Loop(plant, nfq, "Hardware Swingup", sart_folder)


# FIT BEFORE INTERACT >
if initial_fit:
    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        state_channels=STATE_CHANNELS,
        action_channels=("direction_index",),
        lookback=lookback,
        control=nfq,
    )
    # fakes = create_fake_episodes(sart_folder, lookback, batch.num_samples)
    # batch.append(fakes)

    # Fit the normalizer
    nfq.fit_normalizer(batch.observations, method="meanstd")

    batch_size = max(512, batch.num_samples // 10)
    print(f"Batch size: {batch_size}/{batch.num_samples}")

    # Fit the controller
    try:
        nfq.fit(
            batch,
            costfunc=costfunc,
            iterations=50,
            epochs=20,
            minibatch_size=batch_size,
            gamma=gamma,
            callbacks=[callback],
            verbose=0,
        )
    except KeyboardInterrupt:
        pass
    nfq.save(net_name)


start_time = time.time()
cycles = 500
epsilon = nfq.epsilon

pp = LoopPrettyPrinter(costfunc)

for cycle in range(0, cycles):
    print("Cycle:", cycle + 1)
    if cycle % 10 == 0 and cycle != 0:
        nfq.epsilon = 0.0
    else:
        epsilon = max(0.01, epsilon - 0.05)
        nfq.epsilon = epsilon
    print("NFQ Epsilon:", nfq.epsilon)
    time.sleep(1)

    # Collect data
    for _ in range(1):
        loop.run_episode(cycle + 1, max_steps=max_episode_length, pretty_printer=pp)
    # If the last episode ended in the goal state, save it for later viewing
    if plant.is_solved:
        save_name = f"POTENTIAL-{cycle}-swingup-hardware-nfq-model.zip"
    nfq.save(net_name)

    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        state_channels=STATE_CHANNELS,
        action_channels=("direction_index",),
        lookback=lookback,
        control=nfq,
    )
    # fakes = create_fake_episodes(sart_folder, lookback, batch.num_samples)
    # batch.append(fakes)

    # Fit the normalizer
    nfq.fit_normalizer(batch.observations, method="meanstd")

    batch_size = max(
        256 * ((cycle // 20) + 1), batch.num_samples // 10
    )  # max(batch.num_samples // 1, 512)
    print(f"Batch size: {batch_size}/{batch.num_samples}")

    # Fit the controller
    nfq.fit(
        batch,
        costfunc=costfunc,
        iterations=5,
        epochs=5,  # 12,
        minibatch_size=batch_size,
        gamma=gamma,
        callbacks=[callback],
        verbose=0,
    )

print("Elapsed time:", time.time() - start_time)
