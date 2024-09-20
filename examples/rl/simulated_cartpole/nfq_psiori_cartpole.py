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


CART_POSITION_CHANNEL_IDX = 0
COSINE_CHANNEL_IDX = 4

x_threshold=3.6

def make_cosine_cost_func(x_boundary: float=2.4) -> Callable[[np.ndarray], np.ndarray]:
    def cosine_costfunc(states: np.ndarray) -> np.ndarray:

        if np.ndim(states) == 1:
            #print("WARNING: states is a 1D list. This should not happen.")
            position = states[CART_POSITION_CHANNEL_IDX]
            cosine = states[COSINE_CHANNEL_IDX]

            if abs(position) >= x_boundary:
                cost = 1.0
            elif abs(position) >= x_boundary*0.9:
                cost = 0.1
            else:
                cost = (1.0-(cosine+1.0)/2.0) / 100.0
            #print(cost)
            return cost
        
        position = states[:, CART_POSITION_CHANNEL_IDX]
        cosine = states[:, COSINE_CHANNEL_IDX]

        costs = (1.0-(cosine+1.0)/2.0) / 100.0  
        costs[abs(position) >= x_boundary*0.9] = 0.1
        costs[abs(position) >= x_boundary] = 1.0

        #print(costs)

        return costs
    return cosine_costfunc

def make_sparse_cost_func(x_boundary: float=2.4) -> Callable[[np.ndarray], np.ndarray]:
    def sparse_costfunc(states: np.ndarray) -> np.ndarray:
        # unfortunately, we need to provide a vectorized version (for the batch
        # processing in the controller) as well as a single state verison (for the
        # plants).

        if np.ndim(states) == 1:  # this is the version for a single state
            #print("WARNING: states is a 1D list. This should not happen.")
            position = states[CART_POSITION_CHANNEL_IDX]
            cosine = states[COSINE_CHANNEL_IDX]

            if abs(position) >= x_boundary:
                cost = 1.0
            elif abs(position) >= x_boundary*0.9:
                cost = 0.1
            elif abs(position) <= x_boundary*0.2:
                cost = (1.0-(cosine+1.0)/2.0) / 100.0
            else:
                cost = 0.01
            #print(cost)
            return cost
        
        position = states[:, CART_POSITION_CHANNEL_IDX]
        cosine = states[:, COSINE_CHANNEL_IDX]

        costs = (1.0-(cosine+1.0)/2.0) / 100.0  # can only get lower costs in center of x axis
        costs[abs(position) >= x_boundary*0.2] = 0.01  # standard step costs 
        costs[abs(position) >= x_boundary*0.9] = 0.1   # 10x step costs close to x_boundary
        costs[abs(position) >= x_boundary] = 1.0       # 100x step costs in terminal states

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
        # well as our students code, but also in "prominent" projects and papers.

        #print(costs)

        return costs
    return sparse_costfunc


cosine_cost_func = make_cosine_cost_func(x_boundary=x_threshold)
sparse_cost_func = make_sparse_cost_func(x_boundary=x_threshold)

used_cost_func = sparse_cost_func

print(">>> ATTENTION: chosen cost function: ", used_cost_func)

plant = CartPole(x_threshold=x_threshold, cost_function=used_cost_func)  # note: this is instantiated!
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
    pole_sine = episode.observations[:, 3]
    pole_cosine = episode.observations[:, 4]
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

def plot_metrics(metrics, fig=None, filename=None):
    if fig is not None:
        fig.clear()
    else:
        fig = plt.figure(1,  figsize=(10, 8))

    axs = fig.subplots(2)

    window_size = 7

    if window_size > len(metrics["avg_cost"]):
        return
    
    #print(">>> metrics['avg_cost']", metrics["avg_cost"])
    
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

    fig.canvas.draw()

    if filename is not None:
        fig.savefig(filename)

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


for i in range(200):
    loop.run(1, max_episode_steps=500)

    #batch.append_from_hdf5(sart_folder,
    #                       action_channels=["move_index",])

    batch = Batch.from_hdf5(
        sart_folder,
        action_channels=["move_index",],
        lookback=lookback,
        control=nfq,
    )

    plot_swingup_state_history(batch._episodes[len(batch._episodes)-1],
                               filename=f"swingup_latest_episode-{len(batch._episodes)}.png",
                               episode_num=len(batch._episodes))
    
    print(">>> num episodes in batch: ", len(batch._episodes))
    
    # Fit the normalizer
    if (i < 10 or i % 10 == 0) and i < 100:
        nfq.fit_normalizer(batch.observations) # , method="max")


    # Fit the controller
#callback = PlottingCallback(
#    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "loss"
#)
    try:
        nfq.fit(
            batch,
            costfunc=used_cost_func,
            iterations=5,
            epochs=10,
            minibatch_size=256,
            gamma=0.98,
            verbose=1,
 
#        callbacks=[callback],
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

        fig = plot_metrics(metrics, fig=fig, filename=f"metrics-latest.png")
        if fig is not None:
            fig.show()

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

