# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.
"""
Solves CartPole-v0 by using the growing batch approach.

Note: If you want to visualize the learning process, you can set render in
Loop instantiation to True.
"""

import tensorflow as tf
from tensorflow.keras import layers as tfkl

from psipy.rl.control import DiscreteRandomActionController
from psipy.rl.control.nfq import NFQ
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleGymAction,
    CartPoleState,
    CartPolePlant,
)

# Define where we want to save our SART files
sart_folder = "./sart-cartpole-growing"

# Create a model based on state, action shapes and lookback
def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="state")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(40, activation="tanh")(net)
    net = tfkl.Dense(40, activation="tanh")(net)
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
    return tf.keras.Model(inp, net)


# Create some placeholders so lines don't get too long
plant = CartPolePlant()  # note: this is instantiated!
ActionType = CartPoleGymAction
StateType = CartPoleState
lookback = 1

# Collect initial data with a discrete random action controller
explorer = DiscreteRandomActionController(StateType.channels(), ActionType)
loop = Loop(plant, explorer, "CartPolev0", sart_folder)
loop.run(5)

# Make the NFQ model
model = make_model(len(StateType.channels()), len(ActionType.legal_values[0]), lookback)
nfq = NFQ(
    model=model,
    state_channels=StateType.channels(),
    action=ActionType,
    action_values=(0, 1),
    lookback=lookback,
)

# Load the collected data
batch = Batch.from_hdf5(sart_folder, lookback=lookback, control=nfq)

# This loop will use the fitted controller to generate more data and
# subsequently use it to fit again.  Set reps to 1 to not use this.
cycles = 10
for cycle in range(cycles):
    # Fit the normalizer
    nfq.normalizer.fit(batch.observations, method="meanstd")

    # Fit the controller
    nfq.fit(batch, iterations=5, epochs=100, minibatch_size=5, gamma=1.0, verbose=0)
    nfq.save("cartpole-growing-model.zip")

    eval_loop = Loop(plant, nfq, "EvalGrowingBatch", f"eval-{sart_folder}")
    # We clear since don't want to evaluate the success condition on old runs
    # Note: this is specific to the cartpole implementation!
    plant.success_deque.clear()
    eval_loop.run(200)
    if plant.is_solved:
        print("SOLVED!")
        break

    loop = Loop(plant, nfq, "GrowingBatch", sart_folder, render=True)
    loop.run(5)
    batch.append_from_hdf5(sart_folder)

# This is how loading works
loaded = NFQ.load("cartpole-growing-model.zip", custom_objects=[CartPoleGymAction])

# Eval the controller with rendering on.  Enjoy!
loop = Loop(plant, loaded, "CartPolev0GrowingEval", f"live-{sart_folder}", render=True)
loop.run(100)

print(f"Solved at: {batch.num_episodes if plant.is_solved else 'Not solved!'}")
print(
    f"Cycle time: {CartPolePlant().calculate_cycle_time(len(batch.observations)):.2f} seconds"
)
