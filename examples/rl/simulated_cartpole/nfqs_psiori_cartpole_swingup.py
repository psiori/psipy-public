# Copyright (C) PSIORI GmbH, Germany
# Authors: Sascha Lange

"""Example script that learns to swing up PSIORI's version of the cartpole.

This script trains a neural network to swing up and balance a cartpole using Neural Fitted Q-learning.

The cartpole starts hanging down and needs to learn to swing up and balance in an upright position. The state consists of the cart position, velocities, and pole angle (represented as sine/cosine). Actions are discrete forces applied to the cart.

Training data is stored in SART files under the 'psidata-nfs-sart-cartpole-swingup' directory. On subsequent runs without deleting this data, the script will:

    1. Load the existing training data
    2. Continue training the last saved model
    4. Append new episodes to the existing dataset

The script uses epsilon-greedy exploration and shaped rewards to guide learning, with costs approaching zero as the pole moves upward and the cart stays centered.

It stores the model in the 'model-latest' file and it generates a plot of the cost function over time as well as plot of the trajectories of evaluation episodes.

"""
import os
import sys
from typing import Callable, Optional, List

from matplotlib import pyplot as plt
from numpy import cast
import tensorflow as tf
from tensorflow.keras import layers as tfkl
import numpy as np

from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop
from psipy.rl.plants.simulated.cartpole import (
    CartPoleBangAction,
    CartPole,
    CartPoleState,
)
from psipy.rl.visualization.metrics import RLMetricsPlot
from psipy.rl.visualization.plotting_callback import PlottingCallback

# Define where we want to save our SART files
EXPERIMENT_FOLDER = "experiment-nfqs-cartpole"
SART_FOLDER = f"{EXPERIMENT_FOLDER}/psidata-sart-cartpole"
PLOT_FOLDER = f"{EXPERIMENT_FOLDER}/plots"

RENDER = True
EVAL = True

NUM_EPISODES = 400
NUM_EPISODE_STEPS = 400
GAMMA = 0.98
STACKING = 1            # history length. 1 = no stacking, just the current state.
EPSILON = 0.05          # epsilon-greedy exploration


DEFAULT_STEP_COST = 0.01
TERMINAL_COST = 1.0     # this is the cost for leaving the track. ATTENTION: it needs to be high enough to prevent creating a "shortcut" for the agent; it needs to be higher than the accumulated discounted step costs for the steps going to infinity. For a gamma of 0.98, the geometric series converges to 50x the step cost (as lim(t to infinity) of sum(gamma^t) = 50). We choose the terminal costs to be 100 times the step cost to be on the safe side and make it easy for the agent to understand there is no benefit in leaving the track.

X_THRESHOLD = 3.6

# Create a model based on state, action shapes and lookback
def make_model(n_inputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((1,), name="actions")
    net = tfkl.Flatten()(inp)
    net = tfkl.concatenate([act, net])
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(1, activation="sigmoid")(net)
    model = tf.keras.Model([inp, act], net)
    model.summary()
    return model

state_channels = [
    "cart_position",
    "cart_velocity",
    "pole_sine",
    "pole_cosine",
    "pole_velocity",
#   "move_ACT",  # add, if lookback > 1
]

CART_POSITION_CHANNEL_IDX = state_channels.index("cart_position")
COSINE_CHANNEL_IDX = state_channels.index("pole_cosine")

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
                                   position_idx=CART_POSITION_CHANNEL_IDX,
                                   cosine_idx=COSINE_CHANNEL_IDX)


print(">>> ATTENTION: chosen cost function: ", cost_function)

ActionType = CartPoleBangAction
StateType = CartPoleState

plant = CartPole(x_threshold=X_THRESHOLD,
                 cost_function=CartPole.cost_func_wrapper(
                     cost_function,
                     state_channels))


def plot_swingup_state_history(
    episode: Optional[Episode],
    state_channels: List[str],
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

    x = episode.observations[:, state_channels.index("cart_position")]
    td = episode.observations[:, state_channels.index("cart_velocity")]
    pole_sine = episode.observations[:, state_channels.index("pole_sine")]
    pole_cosine = episode.observations[:, state_channels.index("pole_cosine")]
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

metrics_plot = RLMetricsPlot(filename=f"{PLOT_FOLDER}/metrics-latest.png")


# contruct folder structure as needed
if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)

if not os.path.exists(SART_FOLDER):
    os.makedirs(SART_FOLDER)

if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)


# Load the latest model or create a new one
lookback = STACKING
nfq = None

try:
    nfq = NFQs.load(f"{EXPERIMENT_FOLDER}/model-latest.zip",
                    custom_objects=[ActionType])
        
    nfq.optimizer = tf.keras.optimizers.Adam() # if you want a specific learning rate, provide it here as an argument with learning_rate=... -- the optimizer and the learning rate given to NFQs on construction is not stored in the model files.

    print(">>> MODEL LOADED from ", f"{EXPERIMENT_FOLDER}/model-latest.zip")

except Exception as e:
    # Make the NFQ model
    model = make_model(len(state_channels), lookback=lookback)
    nfq = NFQs(
        model=model,
        state_channels=state_channels,
        action=ActionType,
        action_values=ActionType.legal_values[0],
        optimizer=tf.keras.optimizers.Adam(),
        lookback=lookback,
    )
    print(">>> MODEL could not be loaded, CREATED a new one")

nfq.epsilon = 0.1

# Collect initial data with a discrete random action controller
loop = Loop(plant, nfq, "simulated.cartpole.CartPole", SART_FOLDER, render=RENDER)
eval_loop = Loop(plant, nfq, "simulated.cartpole.CartPole", f"{SART_FOLDER}-eval", render=RENDER)

metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }
min_avg_step_cost = 0.01  # if avg costs of an episode are less than 105% of this, we save the model

fig = None
do_eval = EVAL

callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="ME", is_ax2=lambda x: x.endswith("qdelta")
)

for i in range(NUM_EPISODES):
    loop.run(1, max_episode_steps=NUM_EPISODE_STEPS)

    action_channels = (f"{ActionType.channels[0]}",)
    print(">>> action_channels: ", action_channels)

    batch = Batch.from_hdf5(
        SART_FOLDER,
        action_channels=action_channels,
        state_channels=state_channels,
        lookback=lookback,
        control=nfq,
    )

    filename = f"{PLOT_FOLDER}/swingup_latest_episode-{len(batch._episodes)}.png"

    plot_swingup_state_history(batch._episodes[len(batch._episodes)-1],
                               state_channels=state_channels,
                               filename=filename,
                               episode_num=len(batch._episodes))
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if (i < 10 or i % 10 == 0) and i < NUM_EPISODES / 2:
        nfq.fit_normalizer(batch.observations) # , method="max")


    # Fit the controller
    try:
        nfq.fit(
            batch,
            costfunc=cost_function,
            iterations=4,
            epochs=8,
            minibatch_size=2048,
            gamma=GAMMA,
            callbacks=[callback],
            verbose=1,
        )


        nfq.save(f"{EXPERIMENT_FOLDER}/model-latest-saving")  # this is always saved to allow to continue training after interrupting (and potentially changing) the script

        # delete the old model
        if os.path.exists(f"{EXPERIMENT_FOLDER}/model-latest.zip"):
            os.remove(f"{EXPERIMENT_FOLDER}/model-latest.zip")

        # rename the new model to the old model
        os.rename(f"{EXPERIMENT_FOLDER}/model-latest-saving.zip", 
                  f"{EXPERIMENT_FOLDER}/model-latest.zip")

    except KeyboardInterrupt:
        pass

    if do_eval:
        old_epsilon = nfq.epsilon
        nfq.epsilon = 0.0
        eval_loop.run(1, max_episode_steps=600)
        nfq.epsilon = old_epsilon

        episode_metrics = eval_loop.metrics[1] # only one episode was run

        metrics["total_cost"].append(episode_metrics["total_cost"])
        metrics["cycles_run"].append(episode_metrics["cycles_run"])
        metrics["wall_time_s"].append(episode_metrics["wall_time_s"])
        metrics["avg_cost"].append(episode_metrics["total_cost"] / episode_metrics["cycles_run"])


        metrics_plot.update(metrics)
        metrics_plot.plot()

        if metrics_plot.filename is not None:
            print(">>>>>>>> SAVING PLOT <<<<<<<<<<<")
            metrics_plot.save()

        avg_step_cost = episode_metrics["total_cost"] / episode_metrics["cycles_run"]

        if avg_step_cost < min_avg_step_cost * 1.05:
            filename = f"{EXPERIMENT_FOLDER}/model-candidate-{len(batch._episodes)}"
            print("Saving candidate model: ", filename)
            nfq.save(filename)

        if avg_step_cost < min_avg_step_cost:
            min_avg_step_cost = avg_step_cost
            nfq.save(f"{EXPERIMENT_FOLDER}/model-very_best")


