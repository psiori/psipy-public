# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Example script that learns Cartpole Swingup with NFQ on the Swingup Hardware.

In order to connect, you need to follow the instructions in :class:`SwingupPlant`.
"""


"""
==========
NOTES (SL)
==========

Things to check:
+ immediate costs:  ---> fixed (see comments SL in costfunction)
++ "0" when up
++ "1" when down
++ large enough multiple in negative terminal state (e.g. 1000) (fixed)
(+) zero action is real zero ---> presently 150 is neutral action! (SL)
+ correct handling of normalization on all axes including immediate reward and q target (looks fine)
(+/-) immediate reward of terminal state is used (yes&no, see below)
(+/-) update in NFQ on terminal states is correct (its somehow wrong and does not work properly, reason unclear (assumed scaling issue, its not the (only) cause), but worked around with non-terminal more expesive states surrounding terminal state)
+ goal state is NOT a terminal state
+ state information to controller is correct (correct channels, plausible values)
+ lookahead does work properly (include n last states PLUS actions)
+ end state of transition at t is the exact same as start state of transtion t+1
(*) cycle time works properly and does not jitter (much)
- we cause no delay of actions in busy, control and zmq pipes (crane OI learning to better check...)
+ repeat estimate of overall delay
+ if using mini batches, sample order is randomized
+ terminal due to bad angle??? --> what is this?  (checked and fixed)
- plot q, close plot


Improve:
- busy has proper logging
- busy gets (optional) more useful terminal output (like robotcontrol?)
- control and plant really check all values for the assumption and complain (optional: stop?) if violated
- check everything back into the public repo and decide about pact / busy (extract, public?)
- discuss repos, enforce merging, deviation of public repo & actions / control / plant issues with Alex, collect opinion before changing
- handle overflow correctly (see SL comments in RL plant)
"""


import glob
import time
import sys
import os
from getopt import getopt

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from psipy.rl.controllers.nfq import NFQ, tanh2
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop, LoopPrettyPrinter
from psipy.rl.visualization.plotting_callback import PlottingCallback
from tensorflow.keras import layers as tfkl

from psipy.rl.plants.real.pact_cartpole.cartpole import (
#SL   SwingupDiscretizedAction,
    SwingupContinuousDiscreteAction,
    SwingupPlant,
    SwingupState,
    plot_swingup_state_history
)

# Define where we want to save our SART files
sart_folder = "SL-swingup"
net_name = sart_folder + "/latest_net.zip"

STATE_CHANNELS = (
    "cart_position",
    "cart_velocity",
#SL    "pole_angle",
    "pole_sine",
    "pole_cosine",
    "pole_velocity",
    "direction_ACT",
)
THETA_CHANNEL_IDX = STATE_CHANNELS.index("pole_cosine") #SL pole_angle
CART_POSITION_CHANNEL_IDX = STATE_CHANNELS.index("cart_position") #SL pole_angle
EPS_STEPS = 400


# Create a model based on state, action shapes and lookback
def make_model_nfqs(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((1,), name="action")
    net = tfkl.Flatten()(inp)
    net = tfkl.Concatenate()([net, act])
    net = tfkl.Dense(40, activation="tanh")(net)
    net = tfkl.Dense(40, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
    return tf.keras.Model([inp, act], net)


# Create a model based on state, action shapes and lookback
def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="state")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(256, activation="relu")(net)
    net = tfkl.Dense(100, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)  # sigmoid
    return tf.keras.Model(inp, net)


# Define a custom cost function to change the inbuilt costs
def costfunc(states: np.ndarray) -> np.ndarray:
    position = states[:, CART_POSITION_CHANNEL_IDX] # SL TODO: change, don't assume index 0 for position
    theta = states[:, THETA_CHANNEL_IDX]             # SL TODO: if using cosine, needs to change 
    theta_speed = states[:, THETA_CHANNEL_IDX + 1]   

    to_fast = abs(theta_speed) > 0.45

    costs = (1.0-(theta+1.0)/2.0) / 100.0 + (abs(theta_speed) > 0.42) * (abs(theta_speed) - 0.42) / 5.0 #SL orig: tanh2(theta, C=0.01, mu=0.5)
                              # why this: gives 1 when standing up and 0 when hanging down (bc theta..)  -- probably divided to make sure its smaller than terminal costs of failure
    #costs += tanh2(theta_speed, C=0.01, mu=2.5)

    #print(f"#### theta: { theta } costs before bounds: { costs }")
    #print(f"--------------- size costs, position { costs.size }, { position.size }")

    # TERMINAL COSTS FOR NFQ: ARE BASICALLY IGNORED; STATE NEEDS TO HAVE "TERMINAL=True" SET, IN WHICH CASE NFQ IMPLEMENTATION WILL SET EXPECTED COST TO 1 :(

    # PROBLEM: for terminals, we need to check raw, unmoved position (no moved zero), but we can't compute here and lack information about the zero shift. Thus, use the lEFT_SIDE, because we know, in default param setup thats the zero. Also problem: if we move zero, MDP is not markov, because we cant derive positions of bounds from state :(

    center = (SwingupPlant.LEFT_SIDE + SwingupPlant.RIGHT_SIDE) / 2.0
    margin = abs(SwingupPlant.RIGHT_SIDE - SwingupPlant.LEFT_SIDE) / 2.0 * 0.3  # 25% of distance from center to hard endstop

    if position.size > 1:  # SL: original version did not work if passed single states
        costs[abs(position - center) > margin] = 0.011
        costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET * 2] = 0.1
        costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET * 2] = 0.1
        costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET] = 1.0
        costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET] = 1.0

    elif position.size == 1:
        if (abs(position[0] - center) > margin):
            costs[0] = 0.011
        #print (f"true_position { position[0] + SwingupPlant.LEFT_SIDE }")
        if (position[0] + SwingupPlant.LEFT_SIDE<= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET or position[0] >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET):
            costs[0] = 1.0
        elif (position[0] + SwingupPlant.LEFT_SIDE<= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET * 2 or position[0] >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET * 2):
            costs[0] = 0.1
    #print(costs)

    return costs


num = 1000
rando_positions = np.random.randint(
    SwingupPlant.LEFT_SIDE + 500,  # Do not go into terminal area
    SwingupPlant.RIGHT_SIDE - 500,  # Do not go into terminal area
    num,
)


# Create a function to make new episodes (if desired)
def create_fake_episodes(folder: str, lookback: int):
    kwargs = dict(lookback=lookback)
    for path in glob.glob(f"{folder}/*.h5"):
        try:
            with SARTReader(path) as reader:
                o, a, t, c = reader.load_full_episode(
                    state_channels=STATE_CHANNELS, action_channels=("direction_index",),
                )
            a_ = a.copy()
            a_[a == 1] = 0
            a_[a == 0] = 1
            o[:, 0] -= SwingupPlant.CENTER
            o = -1 * o
            o[:, 0] += SwingupPlant.CENTER
            yield Episode(o, a_, t, c, **kwargs)
        except (KeyError, OSError):
            continue

    kwargs = dict(lookback=lookback)

    # According to NFQ Tricks, add goal states with all 3 actions
    # These are added 200 times
    print(f"Creating {num} hints to goal.")
    print("These are only calculated for 3-act!")
    o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    o[:, 0] = rando_positions
    t = np.zeros(num)
    a = np.ones(num)
    o[:, -1] = -1  # Do opposite of current action
    c = np.zeros(num)
    yield Episode(o, a, t, c, **kwargs)
    o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    o[:, 0] = rando_positions
    t = np.zeros(num)
    a = np.zeros(num)
    o[:, -1] = 1  # Do opposite of current action
    c = np.zeros(num)
    yield Episode(o, a, t, c, **kwargs)
    o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    o[:, 0] = rando_positions
    t = np.zeros(num)
    a = np.ones(num) * 2  # 0 is the 2nd index
    c = np.zeros(num)
    yield Episode(o, a, t, c, **kwargs)
    # o = np.zeros((num, len(STATE_CHANNELS)))  # 5th channel is 0-action enrichment
    # t = np.zeros(num)
    # a = np.ones(num) * 2 # 0 is the 2nd index
    # c = np.zeros(num)
    # yield Episode(o, a, t, c, **kwargs)


plant = SwingupPlant(hilscher_port="5555", cost_func=costfunc, sway_start=False)#, 
		   #  speed_values=(800, 400)) #SL speed_value1, speed_value2
        
ActionType = SwingupContinuousDiscreteAction # SL
StateType = SwingupState
lookback = 6
gamma = 0.98
max_episode_length = EPS_STEPS

load_network = False
initial_fit = False

try:
    opts, args = getopt(sys.argv[1:], "hp:", ["help", "play="])
except getopt.GetoptError as err:
    print("Usage: python nfq_hardware_swingup.py [--play <model.zip>]")
    sys.exit(2)

play_model = None

for opt, arg in opts:
    if opt == "-h":
        print("Usage: python nfq_hardware_swingup.py [--play <model.zip>]")
        sys.exit()
    elif opt in ("-p", "--play"):
        play_model = NFQ.load(arg, custom_objects=[ActionType])

if play_model is not None:
    loop = Loop(plant, play_model, "Hardware Swingup", f"{sart_folder}-play")
    loop.run_episode(1, max_steps=-1)
    sys.exit()

# Make the NFQ model
if not load_network:
    model = make_model(len(STATE_CHANNELS), len(ActionType.legal_values[0]), lookback,)
    nfq = NFQ(
        model=model,
        state_channels=STATE_CHANNELS,
        action_channels=("direction",),
        action=ActionType,
        action_values=ActionType.legal_values[0],
        lookback=lookback,
     #   num_repeat=5,
        scale=True,
    )
else:
    nfq = NFQ.load(net_name, custom_objects=[ActionType])

nfq.epsilon = 0.5

callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "avg_qdelta"
)

loop = Loop(plant, nfq, "Hardware Swingup", sart_folder)
eval_loop = Loop(plant, nfq, "Hardware Swingup", f"{sart_folder}-eval")


import numpy as np

def plot_metrics(metrics, fig=None, filename=None):
    if fig is not None:
        fig.clear()
    else:
        fig = plt.figure(1,  figsize=(10, 8))

    axs = fig.subplots(1)

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
    
    axs.plot(x_valid, avg_cost, label="avg_cost", alpha=0.3, color='gray')
    axs.plot(x_valid, moving_avg, label="moving average", color='blue')
    axs.fill_between(x_valid, moving_avg - moving_std, moving_avg + moving_std, alpha=0.2, color='blue', label='±1 std dev')
    
    axs.set_title("Average Cost")
    axs.set_ylabel("Cost per step")
    axs.legend()

    fig.canvas.draw()

    if filename is not None:
        fig.savefig(filename)

    return fig


# FIT BEFORE INTERACT >
if initial_fit:
    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        state_channels=STATE_CHANNELS,
        action_channels=("direction_index",),
        lookback=lookback,
        control=nfq,
    )
    # fakes = create_fake_episodes(sart_folder, lookback, batch.num_samples)
    # batch.append(fakes)

    # Fit the normalizer
    nfq.fit_normalizer(batch.observations, method="meanstd")

    batch_size = max(512, batch.num_samples // 10)
    print(f"Batch size: {batch_size}/{batch.num_samples}")

    # Fit the controller
    try:
        nfq.fit(
            batch,
            costfunc=costfunc,
            iterations=50,
            epochs=20,
            minibatch_size=batch_size,
            gamma=gamma,
            callbacks=[callback],
            verbose=0,
        )
    except KeyboardInterrupt:
        pass
    nfq.save(net_name)


start_time = time.time()
cycles = 500
epsilon = nfq.epsilon

pp = LoopPrettyPrinter(costfunc)

num_cycles_rand_start = 0

metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }
min_avg_step_cost = 0.01  # if avg costs of an episode are less than 105% of this, we save the model

fig = None
do_eval = True

for cycle in range(0, cycles):

    print("Cycle:", cycle)
    if cycle < num_cycles_rand_start:
        nfq.epsilon = 0.8
    elif cycle < 0: # 100:
        epsilon = max(0.1, epsilon - 0.05)
        nfq.epsilon = epsilon
    else:
        nfq.epsilon = 0.0  

    print("NFQ Epsilon:", nfq.epsilon)
    # time.sleep(1)

    # Collect data
    for _ in range(1):
        loop.run_episode(cycle + 1, max_steps=max_episode_length, pretty_printer=pp)

    if cycle < num_cycles_rand_start:
        continue   # don't fit yet

    # If the last episode ended in the goal state, save it for later viewing
    #if plant.is_solved:
    #    save_name = f"POTENTIAL-{cycle}-swingup-hardware-nfq-model.zip"
    #nfq.save(net_name)

    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        state_channels=STATE_CHANNELS,
        action_channels=("direction_index",),
        lookback=lookback,
        control=nfq,
    )
    # fakes = create_fake_episodes(sart_folder, lookback, batch.num_samples)
    # batch.append(fakes)

    # Fit the normalizer
   # if cycle < 40 and cycle % 5 == 0:    # at some point stop normalizing because we dont throw away the network and dont want the data to "move under the network"

#    plot_swingup_state_history(plant=plant, filename=f"episode-{ len(batch._episodes) }.eps")

    iterations = 2


    if cycle == num_cycles_rand_start:
        nfq.fit_normalizer(batch.observations, method="meanstd")
        

    if cycle % 10 == 0 and cycle > 0 and cycle < 100:   
        nfq.fit_normalizer(batch.observations, method="meanstd")
        iterations = 20

    batch_size = max(
        256 * ((cycle // 20) + 1), batch.num_samples // 10
    )  # max(batch.num_samples // 1, 512)
    print(f"Batch size: {batch_size}/{batch.num_samples}")

    try:
    # Fit the controller
        nfq.fit(
            batch,
            costfunc=costfunc,
            iterations=4, # iterations,
            epochs= 8,
            minibatch_size=2048, #batch_size,
            gamma=gamma,
            callbacks=[callback],
            verbose=1,
        )
        try:
            os.rename("model-latest", "model-latest-backup")
        except KeyboardInterrupt:
            pass
        nfq.save(f"model-latest")  # this is always saved to allow to continue training after
    except KeyboardInterrupt:
        pass

    if do_eval:
        old_epsilon = nfq.epsilon
        nfq.epsilon = 0.0
        eval_loop.run(1, max_episode_steps=400)
        nfq.epsilon = old_epsilon

        episode_metrics = eval_loop.metrics[1] # only one episode was run

        metrics["total_cost"].append(episode_metrics["total_cost"])
        metrics["cycles_run"].append(episode_metrics["cycles_run"])
        metrics["wall_time_s"].append(episode_metrics["wall_time_s"])
        metrics["avg_cost"].append(episode_metrics["total_cost"] / episode_metrics["cycles_run"])

        print(">>> metrics['avg_cost']", metrics["avg_cost"])
        print(metrics)
        print(episode_metrics)

#        fig = plot_metrics(metrics, fig=fig, filename=f"metrics-latest.png")
#        if fig is not None:
#            fig.show()

        avg_step_cost = episode_metrics["total_cost"] / episode_metrics["cycles_run"]

        if avg_step_cost < min_avg_step_cost * 1.05:
            filename = f"model-candidate-{len(batch._episodes)}"
            print("Saving candidate model: ", filename)
            nfq.save(filename)

        if avg_step_cost < min_avg_step_cost:
            min_avg_step_cost = avg_step_cost
            try:
                os.rename("model-very_best", "model-second_best")
            except OSError:
                pass
            nfq.save("model-very_best")

print("Elapsed time:", time.time() - start_time)

