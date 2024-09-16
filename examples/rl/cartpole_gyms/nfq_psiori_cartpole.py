"""Example script that learns to swing up PSIORI's version of the cartpole.
"""

import sys
from typing import Optional

from matplotlib import pyplot as plt
from numpy import cast
import tensorflow as tf
from tensorflow.keras import layers as tfkl

from psipy.rl.controllers.nfq import NFQ
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
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
    inp = tfkl.Input((n_inputs, lookback), name="state")
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



def plot_swingup_state_history(
    episode: Optional[Episode],
    plant: Optional[CartPole] = None,
    filename: Optional[str] = None,
) -> None:
    """Creates a plot that details the controller behavior.

    The plot contains 3 subplots:

    1. Cart position
    2. Pole angle, with green background denoting the motor being active
    3. Action from the controller.  Actions that fall within the red
        band are too small to change velocity, and so the cart does not
        move in this zone.

    Args:
        plant: The plant currently being evaluated, will plot after
                the episode is finished.
        sart_path: If given, will load a sart file instead of requiring
                the plant to run

    """
    cost = None
    #plant = cast(CartPole, plant)

    x = episode.observations[:, 0]
    x_s = episode.observations[:, 1]
    t = episode.observations[:, 2]
    pole_cosine = episode.observations[:, 3]
    pole_sine = episode.observations[:, 4]
    td = episode.observations[:, 5]
    a = episode._actions[:, 0]
    cost = episode.costs
        

    fig, axs = plt.subplots(5, figsize=(10, 8))
    axs[0].plot(x, label="cart_position")
    axs[0].set_title("cart_position")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(pole_cosine, label="pole_angle")
    axs[1].axhline(0, color="grey", linestyle=":", label="target")
    axs[1].set_title("Pole Cosine")
#    axs[1].set_ylim((-1.0, 1,0))
    #axs[1].set_ylabel("Angle")
    axs[1].legend()

    axs[2].plot(td, label="pole_velocity")
    axs[2].set_title("pole_velocity")
    axs[2].set_ylabel("Angular Vel")
    axs[2].legend()

    axs[3].plot(a, label="Action")
    axs[3].axhline(0, color="grey", linestyle=":")
    axs[3].set_title("Control")
    axs[3].set_ylabel("Velocity")
    axs[3].legend(loc="upper left")
 #   axs2b = axs[3].twinx()
 #   axs2b.plot(x_s, color="black", alpha=0.4, label="True Velocity")
 #   axs2b.set_ylabel("Steps/s")
 #   axs2b.legend(loc="upper right")

    if cost is not None:
        axs[4].plot(cost, label="cost")
        axs[4].set_title("cost")
        axs[4].set_ylabel("cost")
        axs[4].legend()

    plt.suptitle("NFQ Controller on Physical Swingup Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

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
    action_channels=("move",),
    state_channels=StateType.channels(),
    action=ActionType,
    action_values=ActionType.legal_values[0],
    lookback=lookback,
    scale=True,
)
nfq.epsilon = 0.2

# Collect initial data with a discrete random action controller


loop = Loop(plant, nfq, "simulated.cartpole.CartPole", sart_folder, render=True)

old_epsilon = nfq.epsilon
nfq.epsilon = 1.0

loop.run(1, max_episode_steps=200)

nfq.epsilon = old_epsilon

# Load the collected data
batch = Batch.from_hdf5(
    sart_folder,
    action_channels=["move_index",],
    lookback=lookback,
    control=nfq,
)


for i in range(200):
    loop.run(1, max_episode_steps=200)

    batch.append_from_hdf5(sart_folder,
                           action_channels=["move_index",])
    
    plot_swingup_state_history(batch._episodes[-1], filename=f"swingup_latest_episode.png")

    # Fit the normalizer
    nfq.fit_normalizer(batch.observations) # , method="max")

    # Fit the controller
#callback = PlottingCallback(
#    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "loss"
#)
    try:
        nfq.fit(
            batch,
            iterations=5,
            epochs=10,
            minibatch_size=256,
            gamma=0.98,
            verbose=1,
 
#        callbacks=[callback],
        )
    except KeyboardInterrupt:
        pass

# Eval the controller with rendering on.  Enjoy!
#loop = Loop(plant, nfq, "simulated.cartpole.CartPole", f"{sart_folder}-eval", render=True)
#loop.run(2)

