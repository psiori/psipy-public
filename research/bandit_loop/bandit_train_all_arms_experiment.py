"""Fit all arms of the bandit individually and run evaluation loops."""

import pickle
import time
from multiprocessing import Pool
from os.path import join

import tensorflow as tf

from psipy.core.rate_schedulers import LinearRateScheduler
from psipy.rl.control import NFQ
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleBangAction,
    CartPoleTrigState,
    CartPoleUnboundedPlant,
)
from research.bandit_loop.bandit_loop_experiment import SwingupNeuralTrainer

# Set CPU as available physical device
tf.config.set_visible_devices([], "GPU")


def train_arm_raw(entry_params):
    experiment_dir, arm = entry_params
    with open(join(experiment_dir, "arms.p"), "rb") as f:
        config = pickle.load(f)[arm]
    print(f"Training arm {arm}...")
    results_dir = join(experiment_dir, "individual_train", str(arm))
    trainer = SwingupNeuralTrainer(
        mode="growing",
        plant=CartPoleUnboundedPlant(swingup=True, cost_func=config.cost_function),
        max_episode_steps=200,
        sart_dir=join(results_dir, "sart"),
        training_curve_save_directory=results_dir,
        num_episodes=800,
        save_dir=join(results_dir, "models"),
        render=False,
    )
    # Same as the bandit loop experiment's epsilon decay
    # 10 episodes to minimum eps
    eps_scheduler = LinearRateScheduler(0.9, -0.9 / 10, min=0.01)
    trainer.initialize_control(
        control_type=NFQ,
        neural_structure=config.neural_structure,
        state_channels=CartPoleTrigState.channels(),
        action=CartPoleBangAction,
        action_channel="move",
        lookback=config.lookback,
        iterations=config.iterations,
        epochs=config.epochs,
        minibatch_size=config.batchsize,
        gamma=config.gamma,
        costfunc=config.cost_function,
        epsilon_decay_scheduler=eps_scheduler,
        reset_params=config.reset_params,
        prioritization=config.prioritization,
        scale=config.scale,
        double=config.double,
        dueling=config.dueling,
        batch_only_newest=config.only_newest,
    )
    trainer.train()
    trainer.metrics_to_csv(results_dir)

def get_params(name, arms):
    print(f"Training {len(arms)} arms ({arms}).")
    entry_params = []
    for arm in arms:
        entry_params.append((name, arm))
    return entry_params

if __name__ == "__main__":
    import argparse, sys

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment", help="The path to the experiment folder")
        parser.add_argument(
            "whicharms", nargs="+", help="Which arms to individually train"
        )
        args = parser.parse_args()
        name = args.experiment
        arms = [int(arm) for arm in args.whicharms]
    else:
        name = "Sway-Test-2"
        arms = (0, 1)  # example

    entry_params = get_params(name, arms)

    s = time.time()
    with Pool(len(arms)) as p:
        p.map(train_arm_raw, entry_params)

    print(f"Total time for {len(arms)} arms: {round((time.time() - s)/60, 2)}m")
