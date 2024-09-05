# PSIORI Reinforcement Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Example script that learns cartpole with NFQ.

10 iterations @ 100 epochs:
    With 50 samples and no reset, the problem is solved
These results were found with hand tuning; more efficiency can probably be found!
"""
import time

import tensorflow as tf
from tensorflow.keras import layers as tfkl

from psipy.rl.control.nfq import NFQ, tanh2
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop, LoopPrettyPrinter
from psipy.rl.plant.swingup_plant import SwingupPlant, SwingupAction, SwingupState, SwingupDiscretizedAction, \
    SwingupContinuouDiscreteAction
from psipy.rl.visualization.plotting_callback import PlottingCallback
import numpy as np

# Define where we want to save our SART files
sart_folder = "evenmore-act-balance3-fulltrain"
model_name = sart_folder + "/latest_model.zip"
STATE_CHANNELS = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_velocity",
    "direction_ACT",
)
THETA_CHANNEL_IDX = STATE_CHANNELS.index("pole_angle")

# Create a model based on state, action shapes and lookback
def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(20, activation="tanh")(net) # ursprunglich 10
    net = tfkl.Dense(20, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
    return tf.keras.Model(inp, net)


def costfunc(states: np.ndarray) -> np.ndarray:
    theta = states[:, THETA_CHANNEL_IDX]
    costs = tanh2(theta, C=1/200, mu=.2)
    # costs = np.zeros(len(theta))
    # costs[np.abs(theta) >= .78] = 1

    return costs


# Create some placeholders so lines don't get too long
plant = SwingupPlant("5555", speed_value=800, continuous=True, balance_task=True)  # note: this is instantiated!
ActionType = SwingupContinuouDiscreteAction
StateType = SwingupState
lookback = 3

LOAD_MODEL = False
if not LOAD_MODEL:
    # Make the NFQ model
    model = make_model(len(StateType.channels()), len(ActionType.legal_values[0]), lookback)
    nfq = NFQ(
        model=model,
        state_channels=STATE_CHANNELS,
        action=ActionType,
        action_values=ActionType.legal_values[0],
        lookback=lookback,
        action_channels=("direction",),
        num_repeat=1
    )
else:
    nfq = NFQ.load(sart_folder +"/latest.zip", custom_objects=[SwingupContinuouDiscreteAction])

loop = Loop(plant, nfq, "MoreAct Balance", sart_folder)

max_eps_length = 200
nfq.epsilon = .8
callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="mse", is_ax2=lambda x: x == "loss"
)
training_cycles = 500
pp = LoopPrettyPrinter(costfunc)
for cycle in range(training_cycles):
    print(f"Cycle: {cycle+1}")
    nfq.epsilon = max(.01, nfq.epsilon - .02)
    print("Epsilon:", nfq.epsilon)
    print("LET GO!!!!!!!!")
    time.sleep(1)
    loop.run_episode(cycle+1, max_steps=max_eps_length, pretty_printer=pp)
    nfq.save(model_name)

    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        action_channels=(f"direction_index",),
        lookback=lookback,
        control=nfq,
    )

    # Fit the normalizer
    nfq.fit_normalizer(batch.observations, method="max")

    batch_size = min(512 * ((cycle // 20) + 1), batch.num_samples)
    nfq.fit(
        batch,
        costfunc=costfunc,
        iterations=1000,#10
        epochs=5,
        minibatch_size=batch_size,
        gamma=0.999,
        callbacks=[callback],
        verbose=0
    )
    break

nfq.save("finished-balance-model.zip")