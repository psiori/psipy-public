"""Example script that learns to swing up PSIORI's version of the cartpole.
"""

import sys
from typing import Optional

from matplotlib import pyplot as plt
from numpy import cast
import tensorflow as tf
from tensorflow.keras import layers as tfkl
import numpy as np

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
plant = CartPole(x_threshold=3.6)  # note: this is instantiated!
ActionType = CartPoleBangAction
StateType = CartPoleState



def plot_swingup_state_history(
    episode: Optional[Episode],
    plant: Optional[CartPole] = None,
    filename: Optional[str] = None,
    episode_num: Optional[int] = None,
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
        
    figure = plt.figure(0,  figsize=(10, 8))
    figure.clear()

    axes = figure.subplots(5)

    axes[0].plot(x, label="cart_position")
    axes[0].set_title("cart_position")
    axes[0].set_ylabel("Position")
    axes[0].legend()

    axes[1].plot(pole_cosine, label="cos")
    axes[1].plot(pole_sine, label="sin")
    axes[1].axhline(0, color="grey", linestyle=":", label="target")
    axes[1].set_title("Angle")
#   axes[1].set_ylim((-1.0, 1,0))
    #axes[1].set_ylabel("Angle")
    axes[1].legend()

    axes[2].plot(td, label="pole_velocity")
    axes[2].set_title("pole_velocity")
    axes[2].set_ylabel("Angular Vel")
    axes[2].legend()

    axes[3].plot(a, label="Action")
    axes[3].axhline(0, color="grey", linestyle=":")
    axes[3].set_title("Control")
    axes[3].set_ylabel("Velocity")
    axes[3].legend(loc="upper left")
 #   axes2b = axs[3].twinx()
 #   axes2b.plot(x_s, color="black", alpha=0.4, label="True Velocity")
 #   axes2b.set_ylabel("Steps/s")
 #   axes2b.legend(loc="upper right")

    if cost is not None:
        axes[4].plot(cost, label="cost")
        axes[4].set_title("cost")
        axes[4].set_ylabel("cost")
        axes[4].legend()

    if episode_num is None:
        figure.suptitle("NFQ Controller on Physical Swingup Model")
    else:
        figure.suptitle(f"NFQ Controller on Physical Swingup Model, Episode {episode_num}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        figure.savefig(filename)
        plt.close(figure)
    else:
        figure.show()

import numpy as np

def plot_metrics(metrics, fig=None):
    if fig is None:
        fig = plt.figure(1,  figsize=(10, 8))
    else:
        fig.clear()

    axs = fig.subplots(2)

    window_size = 7

    if window_size > len(metrics["avg_cost"]):
        return
    
    # Calculate moving average and variance
    avg_cost = np.array(metrics["avg_cost"])
    moving_avg = np.convolve(avg_cost, np.ones(window_size)/window_size, mode='same')
    
    # Calculate moving variance
    moving_var = np.convolve(avg_cost**2, np.ones(window_size)/window_size, mode='same') - moving_avg**2
    moving_std = np.sqrt(moving_var)
    
    # Plot original data, moving average, and variance
    x = range(len(avg_cost))
    x_valid = x # range(window_size-1, len(avg_cost))
    
    axs[0].plot(x_valid, avg_cost, label="avg_cost", alpha=0.3, color='gray')
    axs[0].plot(x_valid, moving_avg, label="moving average", color='blue')
    axs[0].fill_between(x_valid, moving_avg - moving_std, moving_avg + moving_std, alpha=0.2, color='blue', label='±1 std dev')
    
    axs[0].set_title("Average Cost")
    axs[0].set_ylabel("Cost per step")
    axs[0].legend()

    fig.show()

    return fig
    


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
eval_loop = Loop(plant, nfq, "simulated.cartpole.CartPole", f"{sart_folder}-eval", render=True)

old_epsilon = nfq.epsilon
nfq.epsilon = 1.0

loop.run(1, max_episode_steps=300)

nfq.epsilon = old_epsilon

metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }

# Load the collected data
batch = Batch.from_hdf5(
    sart_folder,
    action_channels=["move_index",],
    lookback=lookback,
    control=nfq,
)

fig = None
do_eval = True

for i in range(200):
    loop.run(1, max_episode_steps=300)

    batch.append_from_hdf5(sart_folder,
                           action_channels=["move_index",])
    plot_swingup_state_history(batch._episodes[len(batch._episodes)-1], filename=f"swingup_latest_episode.png",
                               episode_num=len(batch._episodes))
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if i % 10 == 0:
        nfq.fit_normalizer(batch.observations) # , method="max")

    if do_eval:
        old_epsilon = nfq.epsilon
        nfq.epsilon = 0.0
        eval_loop.run(1, max_episode_steps=300)
        nfq.epsilon = old_epsilon

        episode_metrics = eval_loop.metrics[1] # only one episode was run

        metrics["total_cost"].append(episode_metrics["total_cost"])
        metrics["cycles_run"].append(episode_metrics["cycles_run"])
        metrics["wall_time_s"].append(episode_metrics["wall_time_s"])
        metrics["avg_cost"].append(episode_metrics["total_cost"] / episode_metrics["cycles_run"])

        fig = plot_metrics(metrics, fig=fig)


    # Fit the controller
#callback = PlottingCallback(
#    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "loss"
#)
    try:
        nfq.fit(
            batch,
            iterations=3,
            epochs=8,
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

