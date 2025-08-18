# Copyright (C) PSIORI GmbH, Germany
# Authors: Alexander Höreth, Sascha Lange

"""Example script that tests the NFQ-CA implementation on the knob.
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
from psipy.rl.controllers.noise import RandomNormalNoise
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop
from psipy.rl.plants.simulated.knob import (
    DiscreteKnobAction,
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
        figure.suptitle("NFQ-S Controller on Knob")
    else:
        figure.suptitle(f"NFQ-S Controller on Knob, Episode {episode_num}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        figure.savefig(filename)
        plt.close(figure)
    else:
        figure.show()

# Define where we want to save our SART files
EXPERIMENT_FOLDER = "experiment-nfqs-knob"
SART_FOLDER = f"{EXPERIMENT_FOLDER}/psidata-sart-knob"
PLOT_FOLDER = f"{EXPERIMENT_FOLDER}/plots"

RENDER = True
EVAL = True

NUM_EPISODES = 400
NUM_EPISODE_STEPS = 200
GAMMA = 0.98
STACKING = 1            # history length. 1 = no stacking, just the current state.
EPSILON = 0.2           # epsilon-greedy exploration


# Create actor and critic models based on state, action shapes and lookback



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
    "position",
]

ActionType = DiscreteKnobAction
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
nfq = None

try:
    nfq = NFQs.load(f"{EXPERIMENT_FOLDER}/model-latest.zip",
                    custom_objects=[ActionType])
    
    nfq.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    print(">>> MODEL LOADED from ", f"{EXPERIMENT_FOLDER}/model-latest.zip")

except Exception as e:
    # Make the NFQ model
    model = make_model(len(state_channels), lookback)

    nfq = NFQs(
        model=model,
        state_channels=state_channels,
        action=ActionType,
        action_values=ActionType.legal_values[0],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        lookback=lookback,
    )
    print(">>> MODEL could not be loaded, CREATED a new one")


nfq.epsilon = EPSILON

loop = Loop(plant, nfq, "simulated.knob.Knob", SART_FOLDER, render=RENDER)
eval_loop = Loop(plant, nfq, "simulated.knob.Knob", f"{SART_FOLDER}-eval", render=RENDER)

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
        control=nfq,
    )

    filename = f"{PLOT_FOLDER}/knob_latest_episode-{len(batch._episodes)}.png"

    plot_knob_state_history(batch._episodes[len(batch._episodes)-1],
                           state_channels=state_channels,
                           filename=filename,
                           episode_num=len(batch._episodes))
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if (i < 10 or i % 10 == 0) and i < NUM_EPISODES / 2:
        nfq.fit_normalizer(batch.observations, method="std")

    try:
        for iterations in range(2):
            # Fit the controller
            nfq.fit(
                batch,
                iterations=4,
                epochs=8,
                minibatch_size=2048,
                gamma=GAMMA,
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
        eval_loop.run(1, max_episode_steps=NUM_EPISODE_STEPS)
        nfq.epsilon = old_epsilon

        # Create evaluation batch and plot
        eval_batch = Batch.from_hdf5(
            f"{SART_FOLDER}-eval",
            action_channels=action_channels,
            state_channels=state_channels,
            lookback=lookback,
            control=nfq,
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
            nfq.save(filename)

        if avg_step_cost < min_avg_step_cost:
            min_avg_step_cost = avg_step_cost
            nfq.save(f"{EXPERIMENT_FOLDER}/model-very_best")


