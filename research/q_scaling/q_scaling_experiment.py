# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Ablation study of various Q target scaling for NFQ."""

import os
from os.path import join

import tensorflow as tf

from psipy.rl.control import DiscreteRandomActionController
from psipy.rl.control.nfq import NFQ, tanh2
from psipy.rl.loop import Loop
from psipy.rl.neural_trainer import NeuralTrainer
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleGymAction,
    CartPolePlant,
    CartPoleState,
)

if "q_scaling_sart" not in os.listdir():
    # Collect random data and train everything on that data
    print("Collecting initial data...")
    drac = DiscreteRandomActionController(CartPoleState.channels(), CartPoleGymAction)
    loop = Loop(CartPolePlant(), drac, "CartPolev0", "q_scaling_sart")
    loop.run(200, max_episode_steps=200)


class ScalingExperiment:
    def __init__(self, save_dir, name, costfunc):
        self.save_dir = save_dir
        self.plant = CartPolePlant(use_gym_solve=False)
        self.name = name
        self.costfunc = costfunc

    def create_trainer(self):
        trainer = NeuralTrainer(
            mode="batch",
            plant=self.plant,
            max_episode_steps=200,
            sart_dir="q_scaling_sart",
            training_curve_save_directory=self.save_dir,
            num_episodes=None,
            name=self.name,
            costfunc=self.costfunc,
        )
        trainer.initialize_control(
            NFQ,
            (2, 20),
            CartPoleState.channels(),
            CartPoleGymAction,
            "move",
            lookback=1,
            iterations=40,
            epochs=20,
            minibatch_size=1024,
            gamma=0.99,
            epsilon_decay_scheduler=None,
            # norm_method="meanstd"
        )
        return trainer

    def run_experiment(self):
        trainer = self.create_trainer()
        trainer.train()
        trainer.metrics_to_csv(self.save_dir)
        save_dir, _ = os.path.split(self.save_dir)
        trainer.control.WRITE_CSV(save_dir, self.name)

        eval_plant = CartPolePlant(use_gym_solve=True)
        loop = Loop(eval_plant, trainer.control, logdir=join(self.save_dir, "eval"))
        solves = 0
        for i in range(10):
            loop.run_episode(i + 1, 200)
            if eval_plant.is_solved:
                solves += 1

        with open(join(self.save_dir, "numsolves"), "w") as f:
            f.write(str(solves))

        print(f"This run solved the plant {solves}# times.")

    # Note that all targets > 0 restriction had to be relaxed


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    def cost_function100(states):
        return tanh2(states[..., 2], C=100, mu=0.05)

    def cost_function1(states):
        return tanh2(states[..., 2], C=1, mu=0.05)

    def cost_function01(states):
        return tanh2(states[..., 2], C=0.1, mu=0.05)

    def cost_function001(states):
        return tanh2(states[..., 2], C=0.01, mu=0.05)

    def cost_function0001(states):
        return tanh2(states[..., 2], C=0.001, mu=0.05)

    def cost_function00001(states):
        return tanh2(states[..., 2], C=0.0001, mu=0.05)

    cost_functions = [
        cost_function100,
        cost_function1,
        cost_function01,
        cost_function001,
        cost_function0001,
        cost_function00001,
    ]

    cs = [100, 1, 0.1, 0.01, 0.001, 0.0001]

    num_runs = 10

    for func, c in zip(cost_functions, cs):
        abl = "Riedmiller"
        for i in range(num_runs):
            experiment = ScalingExperiment(
                join("H_q_scaling_experiments", str(c), abl, str(i)),
                f"{c}-{i}-{abl}",
                func,
            )
            experiment.run_experiment()
