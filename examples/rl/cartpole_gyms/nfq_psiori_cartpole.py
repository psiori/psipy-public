"""Example script that learns to swing up PSIORI's version of the cartpole.
"""

import sys

import tensorflow as tf
from tensorflow.keras import layers as tfkl

from psipy.rl.controllers.nfq import NFQ
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop
from psipy.rl.plants.simulated.cartpole import (
    CartPoleBangAction,
    CartPole,
    CartPoleState,
)
#from psipy.rl.visualization.plotting_callback import PlottingCallback

# Define where we want to save our SART files
sart_folder = "psidata-sart-cartpole-swingup"


# Create a model based on state, action shapes and lookback
def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
    return tf.keras.Model(inp, net)


# Create some placeholders so lines don't get too long
plant = CartPole()  # note: this is instantiated!
ActionType = CartPoleBangAction
StateType = CartPoleState

state_channels = [
    "cart_position",
    "cart_velocity",
    "pole_sine",
    "pole_cosine",
    "pole_velocity",
#   "move_ACT",  # add, if lookback > 1
]
lookback = 1


# Make the NFQ model
model = make_model(len(StateType.channels()), len(ActionType.legal_values[0]), lookback)
nfq = NFQ(
    model=model,
    state_channels=StateType.channels(),
    action=ActionType,
    action_values=(-10, 10),
    lookback=lookback,
    doubleq=False,
    prioritized=False,
    cost_function=plant._cost_function,
)

# Collect initial data with a discrete random action controller

loop = Loop(plant, nfq, "simulated.cartpole.CartPole", sart_folder, render=True)

for i in range(200):
    nfq.epsilon = 0.2
    loop.run(1)


    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        lookback=lookback,
        control=nfq,
    #    prioritization="proportional",
    )

    # Fit the normalizer
    nfq.fit_normalizer(batch.observations, method="max")

    # Fit the controller
#callback = PlottingCallback(
#    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "loss"
#)
    try:
        nfq.fit(
            batch,
            iterations=2,
            epochs=5,
            minibatch_size=256,
            gamma=0.99,
#        callbacks=[callback],
        )
    except KeyboardInterrupt:
        pass

# Eval the controller with rendering on.  Enjoy!
#loop = Loop(plant, nfq, "simulated.cartpole.CartPole", f"{sart_folder}-eval", render=True)
#loop.run(2)
