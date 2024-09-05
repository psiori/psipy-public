"""Fit a new controller for CartPoleSwingup with the config of the best bandit arm and run evaluation loops."""


import glob
import os
import pickle
import shutil
import time
from os.path import join

import tensorflow as tf

from psipy.core.rate_schedulers import LinearRateScheduler
from psipy.rl import Loop
from psipy.rl.bandit_loop_mp import LoopArm
from psipy.rl.control import NFQ
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleBangAction,
    CartPoleTrigState,
    CartPoleUnboundedPlant,
)
from research.bandit_loop.bandit_loop_experiment import SwingupNeuralTrainer

# Set CPU as available physical device
tf.config.set_visible_devices([], "GPU")

for i in range(100):
    print("This script has been superceded by the individual training of each arm.")
    print(
        "Take the best arm's individual training and use that data; do not retrain a new one!"
    )
    print(
        "Accessing the best arm data is the same as accessing the best arm within 'individual_train'."
    )
    print()


def train_singular_best_arm(
    experiment_dir, num_episodes, config: LoopArm, arm_index: int
) -> bool:
    """Returns True if the best arm needs to be evaluated."""
    if "individual_train" in os.listdir(experiment_dir):
        print("Already ran in individual arms, copying over...")
        shutil.copytree(
            join(experiment_dir, "individual_train", str(arm_index)),
            join(experiment_dir, f"best_arm_singular-{arm_index}"),
        )
        return False
    print("Training the singular best arm...")
    results_dir = join(experiment_dir, f"best_arm_singular-{arm_index}")
    trainer = SwingupNeuralTrainer(
        mode="growing",
        plant=CartPoleUnboundedPlant(swingup=True, cost_func=config.cost_function),
        max_episode_steps=200,
        sart_dir=join(results_dir, "sart"),
        training_curve_save_directory=results_dir,
        num_episodes=num_episodes,
        save_dir=join(results_dir, "models"),
        render=False,
    )
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
        epsilon_decay_scheduler=LinearRateScheduler(
            config.epsilon_init, config.epsilon_decay, min=config.epsilon_min,
        ),
        reset_params=config.reset_params,
        prioritization=config.prioritization,
        scale=config.scale,
        double=config.double,
        dueling=config.dueling,
        batch_only_newest=config.only_newest,
    )
    trainer.train()
    trainer.metrics_to_csv(results_dir)
    return True


def evaluate_models(experiment_dir: str, arm_index: int, iters: int = 10):
    print("Creating eval data for best arm models during training...")
    results_dir = join(experiment_dir, f"best_arm_singular-{arm_index}")
    for i, path in enumerate(
        sorted(glob.glob(join(results_dir, "models", "*.zip")), key=os.path.getmtime)
    ):
        print(
            f"Evaluating model {os.path.split(path)[-1].split('_')[-1].split('.zip')[0]}"
        )
        nfq = NFQ.load(path)
        nfq.epsilon = 0
        loop = Loop(
            CartPoleUnboundedPlant(swingup=True),
            nfq,
            logdir=join(results_dir, "eval", str(i)),
        )
        loop.run(episodes=iters, max_episode_steps=200)
        time.sleep(
            0.5
        )  # hope that this allows the CM timers to stop properly before the next model is loaded


if __name__ == "__main__":
    experiment_name = "INVALIDSwingUp-Base-1"
    with open(join(experiment_name, "results.p"), "rb") as f:
        best_arm_index = pickle.load(f)["best_final_arm"]
    with open(join(experiment_name, "arms.p"), "rb") as f:
        arm_configs = pickle.load(f)
    print("Best arm was:", best_arm_index)
    evaluate = train_singular_best_arm(
        experiment_name,
        len(os.listdir(join(experiment_name, "bandit_data"))),
        arm_configs[best_arm_index],
        best_arm_index,
    )
    if evaluate:
        evaluate_models(experiment_name, best_arm_index)
