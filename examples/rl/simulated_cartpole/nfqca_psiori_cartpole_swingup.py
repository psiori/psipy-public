# Copyright (C) PSIORI GmbH, Germany
# Authors: Alexander Höreth, Sascha Lange

"""Example script that learns to swing up PSIORI's version of the cartpole using NFQ-CA with continuous actions.
"""
import os
import sys
from typing import Callable, Optional, List

from matplotlib import pyplot as plt
from numpy import cast
import tensorflow as tf
from tensorflow.keras import layers as tfkl
import numpy as np

from psipy.rl.controllers.nfqca import NFQCA
from psipy.rl.controllers.noise import RandomNormalNoise
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop
from psipy.rl.plants.simulated.cartpole import (
    CartPoleContinuousAction,
    CartPole,
    CartPoleState,
)
from psipy.rl.visualization.metrics import RLMetricsPlot
from psipy.rl.visualization.plotting_callback import PlottingCallback

# Define where we want to save our SART files
EXPERIMENT_FOLDER = "experiment-nfqca-cartpole"
SART_FOLDER = f"{EXPERIMENT_FOLDER}/psidata-sart-cartpole"
PLOT_FOLDER = f"{EXPERIMENT_FOLDER}/plots"

RENDER = True
EVAL = True

NUM_EPISODES = 400
NUM_EPISODE_STEPS = 400
GAMMA = 0.98
STACKING = 1            # history length. 1 = no stacking, just the current state.
EPSILON = 0.1           # epsilon-greedy exploration
EPSILON_SCALE = 0.5     # std of the normal distribution to be added to explorative actions


DEFAULT_STEP_COST = 0.01
TERMINAL_COST = 1.0     # this is the cost for leaving the track. ATTENTION: it needs to be high enough to prevent creating a "shortcut" for the agent; it needs to be higher than the accumulated discounted step costs for the steps going to infinity. For a gamma of 0.98, the geometric series converges to 50x the step cost (as lim(t to infinity) of sum(gamma^t) = 50). We choose the terminal costs to be 100 times the step cost to be on the safe side and make it easy for the agent to understand there is no benefit in leaving the track.

X_THRESHOLD = 3.6

# Create actor and critic models based on state, action shapes and lookback

def make_actor(inputs, lookback):
    inp = tfkl.Input((inputs, lookback), name="state_actor")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(100, activation="relu")(net)
    net = tfkl.Dense(100, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(1, activation="tanh")(net)
    model = tf.keras.Model(inp, net, name="actor")
    model.summary()
    return model


def make_critic(inputs, lookback):
    inp = tfkl.Input((inputs, lookback), name="state_critic")
    act = tfkl.Input((1,), name="act_in")
    net = tfkl.Concatenate()([tfkl.Flatten()(inp), act])
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(1, activation="sigmoid")(net)
    model = tf.keras.Model([inp, act], net, name="critic")
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
                       valid_angle: float = None,
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

        if valid_angle is not None:
            costs[np.arccos(cosine) > valid_angle] = DEFAULT_STEP_COST * 10

        return costs

    return cost_function

cost_function = make_cost_function(x_threshold=X_THRESHOLD,
                                   valid_angle=None, # np.pi/5.0,
                                   position_idx=CART_POSITION_CHANNEL_IDX,
                                   cosine_idx=COSINE_CHANNEL_IDX)


print(">>> ATTENTION: chosen cost function: ", cost_function)

ActionType = CartPoleContinuousAction
StateType = CartPoleState

plant = CartPole(x_threshold=X_THRESHOLD,
                 cost_function=CartPole.cost_func_wrapper(
                     cost_function,
                     state_channels),
            #     start_angle=0.0,
                 valid_angle=None) # np.pi/4.0)


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
nfqca = None

try:
    nfqca = NFQCA.load(f"{EXPERIMENT_FOLDER}/model-latest.zip",
                       custom_objects=[ActionType])
    
    print(">>> MODEL LOADED from ", f"{EXPERIMENT_FOLDER}/model-latest.zip")

    nfqca.exploration = RandomNormalNoise(size=1, std=EPSILON_SCALE)

except Exception as e:
    # Make the NFQ model
    actor = make_actor(len(state_channels), lookback=lookback)
    critic = make_critic(len(state_channels), lookback=lookback)

    nfqca = NFQCA(
        actor=actor,
        critic=critic,
        state_channels=state_channels,
        action=ActionType,
        lookback=lookback,
        exploration=RandomNormalNoise(size=1, std=EPSILON_SCALE),
        td3=False,  # TODO: double check if we want this
    )
    print(">>> MODEL could not be loaded, CREATED a new one")


loop = Loop(plant, nfqca, "simulated.cartpole.CartPole", SART_FOLDER, render=RENDER)
eval_loop = Loop(plant, nfqca, "simulated.cartpole.CartPole", f"{SART_FOLDER}-eval", render=RENDER)

metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }
min_avg_step_cost = 0.01  # if avg costs of an episode are less than 105% of this, we save the model

fig = None
do_eval = EVAL

RAND_START_EPS = 0


for i in range(NUM_EPISODES):
    loop.run(1, max_episode_steps=NUM_EPISODE_STEPS)

    if i < RAND_START_EPS:
        continue

    action_channels = (f"{ActionType.channels[0]}",)
    print(">>> action_channels: ", action_channels)

    batch = Batch.from_hdf5(
        SART_FOLDER,
        action_channels=action_channels,
        state_channels=state_channels,
        lookback=lookback,
        control=nfqca,
    )

    filename = f"{PLOT_FOLDER}/swingup_latest_episode-{len(batch._episodes)}.png"

    plot_swingup_state_history(batch._episodes[len(batch._episodes)-1],
                               state_channels=state_channels,
                               filename=filename,
                               episode_num=len(batch._episodes))
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if (i < 10 or i % 10 == 0) and i < NUM_EPISODES / 2:
        nfqca.fit_normalizer(batch.observations, method="std")


    try:
        for iterations in range(1):
            # Fit the controller
            nfqca.fit_critic(
                batch,
                costfunc=cost_function,
                iterations=2,
                epochs=8,
                minibatch_size=8192,
                gamma=GAMMA,
                verbose=1,
            )
            nfqca.fit_actor(batch, 
                            epochs=1, 
                            minibatch_size=2000,
                            )

        nfqca.save(f"{EXPERIMENT_FOLDER}/model-latest-saving")  # this is always saved to allow to continue training after interrupting (and potentially changing) the script

        # delete the old model
        if os.path.exists(f"{EXPERIMENT_FOLDER}/model-latest.zip"):
            os.remove(f"{EXPERIMENT_FOLDER}/model-latest.zip")

        # rename the new model to the old model
        os.rename(f"{EXPERIMENT_FOLDER}/model-latest-saving.zip", 
                  f"{EXPERIMENT_FOLDER}/model-latest.zip")

    except KeyboardInterrupt:
        pass

    if do_eval:
        old_exploration = nfqca.exploration
        nfqca.exploration = None
        eval_loop.run(1, max_episode_steps=NUM_EPISODE_STEPS)
        nfqca.exploration = old_exploration

        # Create evaluation batch and plot
        eval_batch = Batch.from_hdf5(
            f"{SART_FOLDER}-eval",
            action_channels=action_channels,
            state_channels=state_channels,
            lookback=lookback,
            control=nfqca,
        )

        eval_filename = f"{PLOT_FOLDER}/swingup_eval_episode-{len(batch._episodes)}.png"

        plot_swingup_state_history(eval_batch._episodes[len(eval_batch._episodes)-1],
                                   state_channels=state_channels,
                                   filename=eval_filename,
                                   episode_num=len(batch._episodes))

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
            nfqca.save(filename)

        if avg_step_cost < min_avg_step_cost:
            min_avg_step_cost = avg_step_cost
            nfqca.save(f"{EXPERIMENT_FOLDER}/model-very_best")


