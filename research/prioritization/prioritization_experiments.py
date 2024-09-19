# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Ablation study of various prioritization methods for NFQ.

This code will run two different prioritization methods against no
prioritization for the cartpole plant, in parallel.
"""

import os
import time
from os.path import join
from typing import Optional, Tuple

import tensorflow as tf

from psipy.core.rate_schedulers import LinearRateScheduler
from psipy.rl.control.nfq import NFQ, expected_discounted_cost
from psipy.rl.neural_trainer import NeuralTrainer
from psipy.rl.plant import Action, Plant, State
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleBangAction,
    CartPoleTrigState,
    CartPoleUnboundedPlant,
)

C99 = expected_discounted_cost(200, 0.99)


def cost_function(states):
    cos = states[..., 3]
    cost = ((-cos + 1) / 2) * C99
    return cost


class CartpoleNeuralTrainer(NeuralTrainer):
    def calculate_custom_metrics(self, episode: Optional[int] = None):
        """Record when the plant ended in the goal state and when it was solved."""
        # Set the default failure case
        self.metrics[f"episode_{episode}"]["Solved"] = False
        self.metrics[f"episode_{episode}"]["EndOnGoal"] = False

        if self.plant.is_solved:
            print("At goal!")
            self.metrics[f"episode_{episode}"]["EndOnGoal"] = True


class PrioritizationGrowingBatchAblationBattery:
    def __init__(
        self,
        plant: Plant,
        state: State,
        action: Action,
        experiment_folder: Optional[str] = None,
    ):
        self._plant = plant
        self._state = state
        self._action = action
        self.experiment_folder = experiment_folder

    @staticmethod
    def run_ablation(cfg: Tuple[str, int, str]):
        prioritization, test_num, name = cfg
        controller_type = NFQ
        cost_func = cost_function
        iterations: int = 1
        epochs: int = 20
        episodes: int = 400
        max_episode_steps: int = 200
        lookback: int = 1
        render: bool = False

        _plant = CartPoleUnboundedPlant(swingup=True)
        _state = CartPoleTrigState
        _action = CartPoleBangAction
        experiment_folder = "PrioritizationSwingupMP"

        base_path = join(f"{experiment_folder}/{name}-test{test_num}")
        data_folder = join(base_path, "collected_episodes")
        os.makedirs(base_path)

        start = time.time()
        print(f"Starting {name} test...")

        trainer = CartpoleNeuralTrainer(
            mode="growing",
            plant=_plant,
            num_episodes=episodes,
            max_episode_steps=max_episode_steps,
            sart_dir=data_folder,
            training_curve_save_directory=base_path,
            save_dir=None,
            callbacks=None,
            name=f"{name}-{test_num}",
            render=render,
            pretty_printer=None,
        )
        trainer.initialize_control(
            control_type=controller_type,
            neural_structure=(2, 20),
            state_channels=_state.channels(),
            action=_action,
            action_channel="move",
            lookback=lookback,
            iterations=iterations,
            epochs=epochs,
            minibatch_size=1024,
            gamma=0.99,
            costfunc=cost_func,
            epsilon_decay_scheduler=LinearRateScheduler(0.9, -0.9 / 10, min=0),
            norm_method="max",
            prioritization=prioritization,
        )
        trainer.train()

        trainer.metrics_to_csv(base_path)

        # self._plant.success_deque.clear()

        tt = round((time.time() - start) / 60, 4)
        print(f"Total time to complete {name} test: {tt}m")


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    def growing_experiments():
        from multiprocessing import Pool

        # Growing Batch test on prioritization
        num_repeats = 10
        prios = {"Base": None, "PrioProp": "proportional", "PrioRank": "rank"}
        configs = []
        for exp in ["Base", "PrioProp", "PrioRank"]:
            for run in range(1, num_repeats + 1):
                configs.append((prios[exp], run, exp))

        configs = list(zip(*(iter(configs),) * 10))
        for cfgs in configs:
            with Pool(len(cfgs)) as p:
                p.map(PrioritizationGrowingBatchAblationBattery.run_ablation, cfgs)

    growing_experiments()
