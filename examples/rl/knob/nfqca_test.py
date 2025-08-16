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

from psipy.rl.controllers.nfqca import NFQCA
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
EPSILON_SCALE = 0.5     # std of the normal distribution to be added to explorative actions


# Create actor and critic models based on state, action shapes and lookback

def make_actor(inputs, lookback):
    inp = tfkl.Input((inputs, lookback), name="state_actor")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(10, activation="tanh")(net)
    net = tfkl.Dense(10, activation="tanh")(net)
    net = tfkl.Dense(1, activation="tanh")(net)
    model = tf.keras.Model(inp, net, name="actor")
    model.summary()
    return model


def make_critic(inputs, lookback):
    inp = tfkl.Input((inputs, lookback), name="state_critic")
    act = tfkl.Input((1,), name="act_in")
    net = tfkl.Concatenate()([tfkl.Flatten()(inp), act])
    net = tfkl.Dense(25, activation="tanh")(net)
    net = tfkl.Dense(25, activation="tanh")(net)
    net = tfkl.Dense(10, activation="tanh")(net)
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

    nfqca.exploration = RandomNormalNoise(size=1, std=0.5)

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
        exploration=RandomNormalNoise(size=1, std=0.5),
        td3=False,  # TODO: double check if we want this
    )
    print(">>> MODEL could not be loaded, CREATED a new one")


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
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if (i < 10 or i % 10 == 0) and i < NUM_EPISODES / 2:
        nfqca.fit_normalizer(batch.observations, method="std")

    try:
        for iterations in range(1):
            # Fit the controller
            nfqca.fit_critic(
                batch,
                iterations=2,
                epochs=8,
                minibatch_size=8192,
                gamma=GAMMA,
                verbose=1,
            )

            nfqca.fit_actor(batch, 
                            epochs=10, 
                            minibatch_size=2048
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


