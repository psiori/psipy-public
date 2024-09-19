# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Example script that learns cartpole with NFQs."""

import sys

import tensorflow as tf
from tensorflow.keras import layers as tfkl

from psipy.rl.control.nfqs import NFQs
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleGymAction,
    CartPoleState,
    CartPolePlant,
    CartPoleAssistedBalancePlant)
from psipy.rl.visualization.plotting_callback import PlottingCallback

# Define where we want to save our SART files
sart_folder = "sart-cartpole"


def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((1,), name="actions")
    net = tfkl.Flatten()(inp)
    net = tfkl.concatenate([act, net])
    net = tfkl.Dense(40, activation="tanh")(net)
    net = tfkl.Dense(40, activation="tanh")(net)
    net = tfkl.Dense(1, activation="sigmoid")(net)
    return tf.keras.Model([inp, act], net)


# Create some placeholders so lines don't get too long
plant = CartPoleAssistedBalancePlant()  # note: this is instantiated!
ActionType = CartPoleGymAction
StateType = CartPoleState
lookback = 1


# Make the NFQ model
model = make_model(len(StateType.channels()), len(ActionType.legal_values[0]), lookback)
nfqs = NFQs(
    model=model,
    state_channels=StateType.channels(),
    action=ActionType,
    action_values=(0, 1),
    lookback=lookback,
    num_repeat=10
)

# Collect initial data with a discrete random action controller
# if "--collect" in sys.argv:
nfqs.epsilon = 1
loop = Loop(plant, nfqs, "CartPolev0", sart_folder, render=False)
loop.run(200)
nfqs.epsilon = 0

# Load the collected data
batch = Batch.from_hdf5(
    sart_folder,
    action_channels=(f"{ActionType.channels[0]}",),
    lookback=lookback,
    control=nfqs,
)

# Fit the normalizer
nfqs.fit_normalizer(batch.observations, method="max")

# Fit the controller
callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "loss"
)
try:
    nfqs.fit(
        batch,
        iterations=30,
        epochs=10,
        minibatch_size=1024,
        gamma=1.0,
        # callbacks=[callback],
    )
except KeyboardInterrupt:
    pass

# Eval the controller with rendering on.  Enjoy!
loop = Loop(plant, nfqs, "CartPolev0Eval", f"live-{sart_folder}", render=True)
loop.run(100)

print(f"Solved at: {batch.num_episodes if plant.is_solved else 'Not solved!'}")
cycle_time = CartPolePlant().calculate_cycle_time(len(batch.observations))
print(f"Cycle time: {cycle_time:.2f} seconds")
