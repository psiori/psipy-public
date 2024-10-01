# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Example script that learns Cartpole Swingup with NFQ on the Swingup Hardware.

In order to connect, you need to follow the instructions in :class:`SwingupPlant`.
"""

import glob
import time
import sys
import os
import numpy as np

from getopt import getopt

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from matplotlib import pyplot as plt
from psipy.rl.controllers.nfq import NFQ, tanh2
from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTReader
from psipy.rl.loop import Loop, LoopPrettyPrinter
from psipy.rl.visualization.plotting_callback import PlottingCallback

from psipy.rl.plants.real.pact_cartpole.cartpole import (
    SwingupContinuousDiscreteAction,
    SwingupPlant,
    SwingupState,
#    plot_swingup_state_history
)


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
    center = (SwingupPlant.LEFT_SIDE + SwingupPlant.RIGHT_SIDE) / 2.0
    margin = abs(SwingupPlant.RIGHT_SIDE - SwingupPlant.LEFT_SIDE) / 2.0 * 0.3  # 25% of distance from center to hard endstop

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


        
ActionType = SwingupContinuousDiscreteAction # SL
StateType = SwingupState
lookback = 6
gamma = 0.98


callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "avg_qdelta"
)


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



start_time = time.time()

pp = LoopPrettyPrinter(costfunc)

num_cycles_rand_start = 0



fig = None
do_eval = True

class Schedule:
    def __init__(self):
        pass

    def value(self, episode):
        return 0.0

class LinearSchedule(Schedule):
    def __init__(self, start, end, num_episodes):
        super().__init__()
        self._start = start
        self._end = end
        self._step = (self._end - self._start) / num_episodes  # pls note: this can be negative, by intention, if end < start!

    def value(self, episode):
        v = self._start + self._step * episode

        if self._end > self._start: 
            return max(self._start, min(self._end, v))
        else:
            return min(self._start, max(self._end, v))
        
class ModuloWrapperSchedule(Schedule):
    """
    Wraps a schedule and returns a default value for every n-th episode, and
    the wrapped schedule's value for the other episodes. Can be used for
    having a greedy evaluation every nth episode, for instance. With
    negate=True, the behavior is inverted, thus returning the wrapped 
    schedule's value only for every n-th episode.
    """
    def __init__(self, schedule, modulo, default_value=0.0, negate=False):
        super().__init__()
        self._schedule = schedule
        self._modulo = modulo
        self._default_value = default_value
        self._negate = negate

    def value(self, episode):
        if self._negate:
            return self._default_value if episode % self._modulo == 0 else self._schedule.value(episode)
        else:
            return self._schedule.value(episode) if episode % self._modulo != 0 else self._default_value
        

def initial_fit(controller,
                sart_folder="psidata-cartpole-train", 
                td_iterations=400, 
                epochs_per_iteration=1,
                minibatch_size=2048,
                callback=None,
                verbose=False):
    

    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        state_channels=STATE_CHANNELS,
        action_channels=("direction_index",),
        lookback=lookback,
        controller=controller,
    )

    print("Initial fitting with {} episodes from {} for {} iterations with {} epochs each and minibatch size of {}.".format(len(batch._episodes),   sart_folder, td_iterations, epochs_per_iteration, minibatch_size))

    # fakes = create_fake_episodes(sart_folder, lookback, batch.num_samples)
    # batch.append(fakes)

    # Fit the normalizer
    print("Initial fitting with data from {} for {} iterations with {} epochs each and minibatch size of {}.".format(sart_folder, td_iterations, epochs_per_iteration, minibatch_size))

    print("Fitting normalizer...")
    controller.fit_normalizer(batch.observations, method="meanstd")

    callbacks = [callback] if callback is not None else None

    # Fit the controller
    print("Initial fitting of controller...")
    try:
        controller.fit(batch,
                       costfunc=costfunc,
                       iterations=td_iterations,
                       epochs=epochs_per_iteration,
                       minibatch_size=minibatch_size,
                       gamma=gamma,
                       callbacks=callbacks,
                       verbose=verbose)
    except KeyboardInterrupt:
        pass
    controller.save("model-initial-fit")    

def learn(plant, 
          controller, 
          sart_folder_base="psidata-cartpole",
          num_episodes=-1,
          max_episode_length=400,
          refit_normalizer=True,
          do_eval=True):

    metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }
    min_avg_step_cost = 0.01    # if avg costs of an episode are less than 100+x% of this, we save the model
    
    sart_folder = f"{sart_folder_base}-train"
    sart_folder_eval = f"{sart_folder_base}-eval"

    episode = 0

    epsilon_schedule = LinearSchedule(start=1.0,
                                      end=0.05,
                                      num_episodes=num_episodes / 10)
    try:
        batch = Batch.from_hdf5(
            sart_folder,
            state_channels=STATE_CHANNELS,
            action_channels=("direction_index",),
            lookback=lookback,
            control=controller,
        )
        print(f"Found {len(batch._episodes)} episodes in {sart_folder}. Will use these for fitting and continue with episode {len(batch._episodes)}")

        if refit_normalizer:
            print("Refit the normalizer again using meanstd.")
            controller.fit_normalizer(batch.observations, method="meanstd")

        episode = len(batch._episodes)
        

    except OSError:
        print("No saved episodes found, starting from scratch.")

    loop = Loop(plant, controller, "Hardware Swingup", sart_folder)
    eval_loop = Loop(plant, controller, "Hardware Swingup", sart_folder_eval)


    while episode < num_episodes or num_episodes < 0:
        print("Starting episode:", episode)

        controller.epsilon = epsilon_schedule.value(episode)
        print("NFQ Epsilon:", controller.epsilon)

        loop.run_episode(episode, max_steps=max_episode_length, pretty_printer=pp)
        episode += 1

        # Load the collected data
        batch = Batch.from_hdf5(
            sart_folder,
            state_channels=STATE_CHANNELS,
            action_channels=("direction_index",),
            lookback=lookback,
            control=controller,
        )
        # fakes = create_fake_episodes(sart_folder, lookback, batch.num_samples)
        # batch.append(fakes)

        # plot_swingup_state_history(plant=plant, filename=f"episode-{ len(batch._episodes) }.eps")

        if refit_normalizer and episode % 10 == 0 and episode < num_episodes / 2:   
            print("Refit the normalizer again using meanstd.")
            controller.fit_normalizer(batch.observations, method="meanstd")


        try:
            # Fit the controller
            controller.fit(
            batch,
            costfunc=costfunc,
            iterations=4, # iterations,
            epochs= 8,
            minibatch_size=2048, #batch_size,
            gamma=gamma,
            callbacks=[callback],
            verbose=1,
        )
        except KeyboardInterrupt:
            pass


        try:
            os.rename("model-latest.zip", "model-latest-backup.zip")
        except OSError:
            pass
        controller.save(f"model-latest")  # this is always saved to allow to continue training after
    
    
        if do_eval:
            old_epsilon = controller.epsilon
            controller.epsilon = 0.0
            eval_loop.run(1, max_episode_steps=400)
            controller.epsilon = old_epsilon

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

            if avg_step_cost < min_avg_step_cost * 1.075:
                filename = f"model-candidate-{len(batch._episodes)}"
                print("Saving candidate model: ", filename)
                controller.save(filename)

            if avg_step_cost < min_avg_step_cost:
                min_avg_step_cost = avg_step_cost
                try:
                    os.rename("model-very_best.zip", "model-second_best.zip")
                except OSError:
                    pass

                controller.save("model-very_best")

    print("Elapsed time:", time.time() - start_time)


def play(plant, controller,
         sart_folder="psidata-cartpole-play",
         num_episodes=-1):
    episode = 0
    loop = Loop(plant, controller, "Hardware Swingup", sart_folder)

    while episode < num_episodes or num_episodes < 0:
        loop.run_episode(episode, max_steps=-1);
        episode += 1

if __name__ == "__main__":
    load_network = False
    initial_fit = False
    play_only = False
    controller = None
    sart_folder_base = "psidata-cartpole"
    max_episode_length = 400
    play_after_initial_fit = False

    try:
        opts, args = getopt(sys.argv[1:], "hfps:l:",
                            ["help", "play-only", "initial-fit", "sart-folder-base=", "load-model="])
    except getopt.GetoptError as err:
        print("Usage: python nfq_hardware_swingup.py [--play <model.zip>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("Usage: python nfq_hardware_swingup.py [--play <model.zip>]")
            sys.exit()
        elif opt in ("-p", "--play-only"):
            play_only = True
        elif opt in ("f", "--initial-fit"):
            initial_fit = True
        elif opt in ("s", "--sart-folder-base"):
            sart_folder_base = arg
        elif opt in ("l", "--load-model"):
            print("Loading controller from file with name: ", arg)
            controller = NFQ.load(arg, custom_objects=[ActionType])

    plant = SwingupPlant(hilscher_port="5555",
                         sway_start=False,
                         cost_function=costfunc)#, 
    
    if controller is None:  # not loaded from file
        model = make_model(len(STATE_CHANNELS), len(ActionType.legal_values[0]), lookback,)

        print("Creating a new controller.")
        controller = NFQ(model=model,
                         state_channels=STATE_CHANNELS,
                         action_channels=("direction",),
                         action=ActionType,
                         action_values=ActionType.legal_values[0],
                         lookback=lookback,
                         scale=True)
            
    if initial_fit:
        initial_fit(controller, 
                    sart_folder=sart_folder_base + "-train")

    if play_only:
        play(plant, controller, 
             sart_folder=sart_folder_base + "-play")
        sys.exit()

    else:
            
        if play_after_initial_fit:
            play(plant, controller, 
                 sart_folder=sart_folder_base + "-play")
            sys.exit()

        else:
            learn(plant, 
                  controller, 
                  sart_folder_base=sart_folder_base, 
                  num_episodes=200,
                  max_episode_length=max_episode_length)







"""
==============================================================================

   END OF IMPLEMENTATION --------  START OF NOTES (SL)

==============================================================================

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
