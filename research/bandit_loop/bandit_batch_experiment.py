"""Fit a new controller for CartPoleSwingup on all the batch data collected via the bandit arms and run evaluation loops."""


import glob
from distutils.dir_util import copy_tree
from os.path import join

import tensorflow as tf

from psipy.rl import Loop
from psipy.rl.control import NFQ
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleBangAction,
    CartPoleTrigState,
    CartPoleUnboundedPlant,
)
from research.bandit_loop.bandit_loop_experiment import SwingupNeuralTrainer
from research.bandit_loop.cost_functions import cost_func_cos99

# Set CPU as available physical device

tf.config.set_visible_devices([], "GPU")


def batch_train(experiment_dir, costfunc):
    print("Starting batch train...")
    results_dir = join(experiment_dir, "batch")
    copy_tree(join(experiment_dir, "bandit_data"), join(results_dir, "data"))
    trainer = SwingupNeuralTrainer(
        mode="batch",
        plant=CartPoleUnboundedPlant(swingup=True, cost_func=costfunc),
        max_episode_steps=200,
        sart_dir=join(results_dir, "data"),
        training_curve_save_directory=results_dir,
        save_dir=results_dir,
        render=True,
    )
    trainer.initialize_control(
        control_type=NFQ,
        neural_structure=(2, 20),
        state_channels=CartPoleTrigState.channels(),
        action=CartPoleBangAction,
        action_channel="move",
        lookback=1,
        iterations=100,
        epochs=40,
        minibatch_size=1024,
        gamma=0.99,
        costfunc=costfunc,
        scale=False,
    )
    trainer.train()
    trainer.metrics_to_csv(results_dir)


def evaluate_model(experiment_dir: str, iters: int = 10):
    print("Creating eval data for batch model...")
    results_dir = join(experiment_dir, "batch")
    model = glob.glob(join(results_dir, "*.zip"))
    assert len(model) == 1
    model = model[0]
    print("Evaluating model")
    nfq = NFQ.load(model)
    nfq.epsilon = 0
    loop = Loop(
        CartPoleUnboundedPlant(swingup=True), nfq, logdir=join(results_dir, "eval"),
    )
    loop.run(episodes=iters, max_episode_steps=200)


if __name__ == "__main__":
    experiment_name = "SwingUpMP-ignore-bad-9arms-1"
    batch_train(experiment_name, cost_func_cos99)
    evaluate_model(experiment_name)
