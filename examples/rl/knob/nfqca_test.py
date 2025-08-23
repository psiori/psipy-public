# Copyright (C) PSIORI GmbH, Germany
# Authors: Alexander Höreth, Sascha Lange

"""Example script that tests the NFQ-CA implementation on the knob.
"""
import os
import sys
from typing import Callable, Optional, List

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers as tfkl
import numpy as np

from psipy.rl.controllers.nfqca import NFQCA
from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.controllers.noise import RandomNormalNoise
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop
from psipy.rl.plants.simulated.knob import (
    ContinuousKnobAction,
    Knob,
    KnobState,
    make_default_cost_function,
)
from psipy.rl.visualization.metrics import RLMetricsPlot
from psipy.rl.visualization.plotting_callback import PlottingCallback

def plot_knob_state_history(
    episode: Optional[Episode],
    state_channels: List[str],
    filename: Optional[str] = None,
    episode_num: Optional[int] = None,
) -> None:
    """Creates a plot that details the controller behavior for the knob.

    The plot contains 3 subplots:

    1. Knob position in degrees [-180, 180]
    2. Action from the controller (turn value)
    3. Immediate costs over time

    Args:
        episode: The episode to plot
        state_channels: List of state channel names
        filename: If given, will save the plot to this file
        episode_num: Episode number for the title

    """
    if episode is None:
        return

    position = episode.observations[:, state_channels.index("position")]
    actions = episode._actions[:, 0]
    costs = episode.costs
        
    figure = plt.figure(0, figsize=(10, 8))
    figure.clear()

    axes = figure.subplots(3)

    # Plot 1: Knob position
    axes[0].plot(position, label="knob_position", color='blue')
    axes[0].axhline(0, color="red", linestyle="--", label="target (0°)")
    axes[0].set_title("Knob Position")
    axes[0].set_ylabel("Position (degrees)")
    axes[0].set_ylim((-180, 180))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Actions
    axes[1].plot(actions, label="turn_action", color='green')
    axes[1].axhline(0, color="grey", linestyle=":", label="no action")
    axes[1].set_title("Control Actions")
    axes[1].set_ylabel("Turn Value")
    axes[1].set_ylim((-2.0, 2.0))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Costs
    axes[2].plot(costs, label="immediate_cost", color='red')
    axes[2].set_title("Immediate Costs")
    axes[2].set_ylabel("Cost")
    axes[2].set_xlabel("Time Steps")
    axes[2].set_ylim((0.0, 0.01))
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if episode_num is None:
        figure.suptitle("NFQ-CA Controller on Knob")
    else:
        figure.suptitle(f"NFQ-CA Controller on Knob, Episode {episode_num}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        figure.savefig(filename)
        plt.close(figure)
    else:
        figure.show()


# Define where we want to save our SART files
EXPERIMENT_FOLDER = "experiment-nfqca-knob"
SART_FOLDER = f"{EXPERIMENT_FOLDER}/psidata-sart-knob"
PLOT_FOLDER = f"{EXPERIMENT_FOLDER}/plots"

RENDER = True
EVAL = True

NUM_EPISODES = 400
NUM_EPISODE_STEPS = 400
GAMMA = 0.98
STACKING = 1            # history length. 1 = no stacking, just the current state.
EPSILON = 0.1           # epsilon-greedy exploration
EPSILON_SCALE = 0.2     # std of the normal distribution to be added to explorative actions

LOAD_NFQ_CRITIC = False


# Create actor and critic models based on state, action shapes and lookback

def make_actor(inputs, lookback):
    inp = tfkl.Input((inputs, lookback), name="state_actor")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(100, activation="relu")(net)
    net = tfkl.Dense(100, activation="relu")(net)
    net = tfkl.Dense(10, activation="tanh")(net)
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
    "position",
]

ActionType = ContinuousKnobAction
StateType = KnobState
plant = Knob(cost_function=make_default_cost_function())

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


if LOAD_NFQ_CRITIC:
    model_file = f"{EXPERIMENT_FOLDER}/model-nfq-critic.zip"
    nfq = NFQs.load(model_file)

    nfqca._critic = nfq._model
    nfqca.normalizer = nfq.normalizer



loop = Loop(plant, nfqca, "simulated.knob.Knob", SART_FOLDER, render=RENDER)
eval_loop = Loop(plant, nfqca, "simulated.knob.Knob", f"{SART_FOLDER}-eval", render=RENDER)

metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }
min_avg_step_cost = 0.01  # if avg costs of an episode are less than 105% of this, we save the model

fig = None
do_eval = EVAL

for i in range(NUM_EPISODES):
    loop.run(1, max_episode_steps=NUM_EPISODE_STEPS)

    action_channels = (f"{ActionType.channels[0]}",)
    print(">>> action_channels: ", action_channels)

    batch = Batch.from_hdf5(
        SART_FOLDER,
        action_channels=action_channels,
        state_channels=state_channels,
        lookback=lookback,
        control=nfqca,
    )

    filename = f"{PLOT_FOLDER}/knob_latest_episode-{len(batch._episodes)}.png"

    plot_knob_state_history(batch._episodes[len(batch._episodes)-1],
                           state_channels=state_channels,
                           filename=filename,
                           episode_num=len(batch._episodes))
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if (i < 10 or i % 10 == 0) and i < NUM_EPISODES / 2 and not LOAD_NFQ_CRITIC:
        nfqca.fit_normalizer(batch.observations, method="std")

    try:
        for iterations in range(1):
            # Fit the controller

            if True or not LOAD_NFQ_CRITIC:
                nfqca.fit_critic(
                    batch,
                    iterations=2,
                    epochs=8,
                    minibatch_size=4000,
                    gamma=GAMMA,
                    verbose=1,
                )
        

            # nfqca.reset_actor() # might not work anymore because of the new tf version. thus, we just re-create the actor model from scratch: 
            # nfqca._actor = make_actor(len(state_channels), lookback=lookback)

            nfqca.fit_actor(batch, 
                            epochs=1, 
                            minibatch_size=500
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

        eval_filename = f"{PLOT_FOLDER}/knob_eval_episode-{len(batch._episodes)}.png"

        plot_knob_state_history(eval_batch._episodes[len(eval_batch._episodes)-1],
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


