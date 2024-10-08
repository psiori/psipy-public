"""Example script that learns to swing up PSIORI's version of the cartpole.
"""

import sys
from typing import Callable, Optional

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
    plot_swingup_state_history
)
from psipy.rl.visualization.plotting_callback import PlottingCallback
from psipy.rl.visualization.metrics import RLMetricsPlot

# Define where we want to save our SART files
sart_folder = "psidata-cartpole-swingup"


# Create a model based on state, action shapes and lookback
def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="state")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
    return tf.keras.Model(inp, net)


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

x_threshold=3.6

def make_cosine_cost_func(x_boundary: float=2.4) -> Callable[[np.ndarray], np.ndarray]:
    def cosine_costfunc(states: np.ndarray) -> np.ndarray:

        if np.ndim(states) == 1:
            #print("WARNING: states is a 1D list. This should not happen.")
            position = states[CART_POSITION_CHANNEL_IDX]
            cosine = states[COSINE_CHANNEL_IDX]

            if abs(position) >= x_boundary:
                cost = 1.0
            elif abs(position) >= x_boundary*0.9: # close to x_boundary
                cost = 0.1
            else: 
                cost = (1.0-(cosine+1.0)/2.0) / 100.0
            return cost
        
        position = states[:, CART_POSITION_CHANNEL_IDX]
        cosine = states[:, COSINE_CHANNEL_IDX]

        costs = (1.0-(cosine+1.0)/2.0) / 100.0  
        costs[abs(position) >= x_boundary*0.9] = 0.1
        costs[abs(position) >= x_boundary] = 1.0

        #print(costs)

        return costs
    return cosine_costfunc

def make_sparse_cost_func(x_boundary: float=2.4,
                          step_cost: float=0.01,
                          use_cosine: bool=True,
                          upright_margin: float=0.2) -> Callable[[np.ndarray], np.ndarray]:
    def sparse_costfunc(states: np.ndarray) -> np.ndarray:
        # unfortunately, we need to provide a vectorized version (for the batch
        # processing in the controller) as well as a single state verison (for the
        # plants).

        if np.ndim(states) == 1:  # this is the version for a single state
            #print("WARNING: states is a 1D list. This should not happen.")

            position = states[CART_POSITION_CHANNEL_IDX]
            cosine = states[COSINE_CHANNEL_IDX]

            if abs(position) >= x_boundary:
                cost = 1.0  # failing terminal state
            elif abs(position) >= x_boundary*0.9:
                cost = step_cost * 10
            elif abs(position) <= x_boundary*0.2:
                if use_cosine:
                    cost = (1.0-(cosine+1.0)/2.0) * step_cost
                else:
                    cost = (abs(1.0-cosine) > upright_margin) * step_cost
            else:
                cost = step_cost
            #print(cost)
            return cost
        
        position = states[:, CART_POSITION_CHANNEL_IDX]
        cosine = states[:, COSINE_CHANNEL_IDX]

        if use_cosine:
            costs = (1.0-(cosine+1.0)/2.0) * step_cost  # can only get lower costs in center of x axis
        else:
            costs = (abs(1.0-cosine) > upright_margin) * step_cost

        costs[abs(position) >= x_boundary*0.2] = step_cost       # standard step costs 
        costs[abs(position) >= x_boundary*0.9] = step_cost * 10  # 10x step costs close to x_boundary
        costs[abs(position) >= x_boundary] = 1.0                 # 100x step costs in terminal states

        # ATTENTION, a word regarding the choice of terminal costs and "step costs":
        # the relation of terminal costs to step costs depends on the gamma value.
        # with gamma=0.98, the geometric sequence  sum(0.98^n) converges to 50 with n
        # going to infinity (infinite lookahead), thus 100x times the cost of an indiviual
        # step seems reasonable and twice as much, as the discounted future step costs can 
        # cause (50x the step cost). for higher gammas closer to one, the terminal costs should
        # be higher, to prevent a terminal state's costs being lower than continuing to acting
        # within the "bounds" (aka non-terminal states). If you see your agent learning to 
        # leave the bounds as quickly as possible, its likely that your terminal costs are too low
        # or your treatment of the terminal transition is not correct (e.g. not doing a TD update
        # on these transitions at all, wrong scaling, etc.). We have seen both types of errors
        # (terminal costs to low, wrong handling of terminal transitions) in our own code as
        # well as our students code, but also in "prominent" projects and papers. So, make sure to check this twice

        return costs
    return sparse_costfunc


cosine_cost_func = make_cosine_cost_func(x_boundary=x_threshold)
sparse_cost_func = make_sparse_cost_func(x_boundary=x_threshold)

used_cost_func = sparse_cost_func

print(">>> ATTENTION: chosen cost function: ", used_cost_func)


lookback = 1

plant = CartPole(x_threshold=x_threshold, cost_function=CartPole.cost_func_wrapper(used_cost_func, state_channels))  # note: this is instantiated!
ActionType = CartPoleBangAction
StateType = CartPoleState


import numpy as np




# Make the NFQ model
model = make_model(len(state_channels), len(ActionType.legal_values[0]), lookback)
nfq = NFQ(
    model=model,
    action_channels=("move",),
    state_channels=state_channels,
    action=ActionType,
    action_values=ActionType.legal_values[0],
    lookback=lookback,
    scale=True,
)
nfq.epsilon = 0.1

# Collect initial data with a discrete random action controller


loop = Loop(plant, nfq, "simulated.cartpole.CartPole", sart_folder, render=False)
eval_loop = Loop(plant, nfq, "simulated.cartpole.CartPole", f"{sart_folder}-eval", render=True)

old_epsilon = nfq.epsilon
nfq.epsilon = 1.0

loop.run(1, max_episode_steps=300)

nfq.epsilon = old_epsilon

metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }
min_avg_step_cost = 0.01  # if avg costs of an episode are less than 105% of this, we save the model

# Load the collected data
#batch = Batch.from_hdf5(
#    sart_folder,
#    action_channels=["move_index",],
#    lookback=lookback,
#    control=nfq,
#)

fig = None
do_eval = True


callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="ME", is_ax2=lambda x: x.endswith("qdelta")
)

metrics_plot = RLMetricsPlot(filename="metrics-latest.png")
episode_plot = None


for i in range(200):
    loop.run(1, max_episode_steps=500)

    #batch.append_from_hdf5(sart_folder,
    #                       action_channels=["move_index",])

    batch = Batch.from_hdf5(
        sart_folder,
        action_channels=["move_index",],
        state_channels=state_channels,
        lookback=lookback,
        control=nfq,
    )

    last_episode_internal = Batch.from_hdf5(
        # load the last episode with full plant-internal state representation
        sart_folder,
        action_channels=["move_index",],
        lookback=lookback,
        only_newest=1,
    )

    episode_plot = plot_swingup_state_history(
        episode=last_episode_internal._episodes[0],
        filename=f"swingup_latest_episode-{len(batch._episodes)}.png",
        episode_num=len(batch._episodes),
        figure=episode_plot,
    )
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if (i < 10 or i % 10 == 0) and i < 100:
        nfq.fit_normalizer(batch.observations) # , method="max")


    # Fit the controller


    try:
        nfq.fit(
            batch,
            costfunc=used_cost_func,
            iterations=4,
            epochs=8,
            minibatch_size=2048,
            gamma=0.98,
            verbose=1,
            callbacks=[callback],
        )
        nfq.save(f"model-latest")  # this is always saved to allow to continue training after interrupting (and potentially changing) the script
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
        metrics_plot.save()

        avg_step_cost = episode_metrics["total_cost"] / episode_metrics["cycles_run"]

        if avg_step_cost < min_avg_step_cost * 1.05:
            filename = f"model-candidate-{len(batch._episodes)}"
            print("Saving candidate model: ", filename)
            nfq.save(filename)

        if avg_step_cost < min_avg_step_cost:
            min_avg_step_cost = avg_step_cost
            nfq.save("model-very_best")


# Eval the controller with rendering on.  Enjoy!
#loop = Loop(plant, nfq, "simulated.cartpole.CartPole", f"{sart_folder}-eval", render=True)
#loop.run(2)

