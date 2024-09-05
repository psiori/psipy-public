# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Find a performant controller for Swingup with the use of a Multi Armed Bandit.

This was the main experiment code used in the thesis

    ``Self Optimizing Growing Batch Reinforcement Agents``

"""


""""""
import multiprocessing as mp
import pickle
import time
from os.path import join
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf

from psipy.core.rate_schedulers import LinearRateScheduler
from psipy.rl.bandit_loop_mp import (
    BanditLoop,
    LoopArm,
    LoopMultiArmedBandit,
    get_processes,
)
from psipy.rl.control.bandits.bandit_optimizers import (
    EpsilonGreedySlidingWindowUCBBanditOptimizer,
)
from psipy.rl.neural_trainer import NeuralTrainer, NeuralTrainerProcess
from psipy.rl.plant.gym.cartpole_plants import (
    CartPoleBangAction,
    CartPoleTrigState,
    CartPoleUnboundedPlant,
)
from psipy.rl.visualization.bandit_plottiing_callbacks import (
    ArmSelectionCallback,
    UCBProbabilityCallback,
)
from research.bandit_loop.cost_functions import *

# Set CPU as available physical device
tf.config.set_visible_devices([], "GPU")


class SwingupNeuralTrainer(NeuralTrainer):
    def calculate_custom_metrics(self, episode: Optional[int] = None):
        """Record when the plant ended in the goal state and when it was solved."""
        # Set the default failure case
        if f"episode_{episode}" not in self.metrics.keys():
            self.metrics[f"episode_{episode}"] = dict()
        self.metrics[f"episode_{episode}"]["EndOnGoal"] = False
        if self.plant.is_solved:
            print("At goal!")
            self.metrics[f"episode_{episode}"]["EndOnGoal"] = True


class SwingupNeuralTrainerProcess(NeuralTrainerProcess):
    def calculate_custom_metrics(self, episode: Optional[int] = None):
        """Record when the plant ended in the goal state and when it was solved."""
        # Set the default failure case
        if f"episode_{episode}" not in self.metrics.keys():
            self.metrics[f"episode_{episode}"] = dict()
        self.metrics[f"episode_{episode}"]["EndOnGoal"] = False
        if self.plant.is_solved:
            print("At goal!")
            self.metrics[f"episode_{episode}"]["EndOnGoal"] = True


# Define each arm here and load them into processes in the training function
arm1 = LoopArm(
    gamma=0.99,
    cost_function=cost_func_cos99,
    iterations=1,
    epochs=20,
    batchsize=1024,
    lookback=1,
    reset_params=None,
    prioritization=None,
    neural_structure=(2, 20),
    double=False,
    dueling=False,
    only_newest=None,
    scale=False,
)
arm2 = LoopArm(
    gamma=0.999,
    cost_function=cost_func_cos999,
    iterations=1,
    epochs=20,
    batchsize=1024,
    lookback=1,
    reset_params=None,
    prioritization=None,
    neural_structure=(2, 20),
    double=False,
    dueling=False,
    only_newest=None,
    scale=False,
)
arm3 = LoopArm(
    gamma=0.99,
    cost_function=cost_func_stepped99,
    iterations=1,
    epochs=20,
    batchsize=1024,
    lookback=1,
    reset_params=None,
    prioritization=None,
    neural_structure=(2, 20),
    double=False,
    dueling=False,
    only_newest=None,
    scale=False,
)
arm4 = LoopArm(
    gamma=0.99,
    cost_function=cost_func_tanh299,
    iterations=1,
    epochs=20,
    batchsize=1024,
    lookback=1,
    reset_params=None,
    prioritization=None,
    neural_structure=(2, 20),
    double=False,
    dueling=False,
    only_newest=None,
    scale=False,
)
arm5 = LoopArm(
    gamma=0.999,
    cost_function=cost_func_tanh2999,
    iterations=1,
    epochs=20,
    batchsize=1024,
    lookback=1,
    reset_params=None,
    prioritization=None,
    neural_structure=(2, 20),
    double=False,
    dueling=False,
    only_newest=None,
    scale=False,
)
badarm = LoopArm(
    gamma=0.5,
    cost_function=cost_func_cos9,
    iterations=1,
    epochs=1,
    batchsize=1024,
    lookback=1,
    reset_params=None,
    prioritization=None,
    neural_structure=(1, 10),
    double=False,
    dueling=False,
    only_newest=None,
    scale=False,
)
adversarialarm = LoopArm(
    gamma=0.99,
    cost_function=cost_func_adversarial99,
    iterations=1,
    epochs=20,
    batchsize=1024,
    lookback=1,
    reset_params=None,
    prioritization=None,
    neural_structure=(2, 20),
    double=False,
    dueling=False,
    only_newest=None,
    scale=False,
)


def train_bandit_loop(
    experiment_name: str,
    use_bad: bool,
    use_stability: bool,
    random: bool,
    adversarial: bool,
    plot: bool,
):
    def get_arms(
        arm_configs: List[LoopArm],
        episodes: int,
        max_steps: int,
        experiment_dir: str,
        callback: bool = False,
        cmd_port: int = 7900,
        report_port: int = 7901,
    ) -> List[Tuple]:
        process_configs: List[Tuple] = []
        for i, config in enumerate(arm_configs):
            process_config = (
                i,  # index
                config,  # LoopArm configuration
                SwingupNeuralTrainerProcess,
                CartPoleUnboundedPlant(swingup=True, cost_func=config.cost_function),
                CartPoleTrigState,
                CartPoleBangAction,
                "move",  # action channel
                episodes,  # num episodes
                max_steps,  # max episode length
                experiment_dir,  # top level experiment dir
                callback,  # plot q curves
                5,  # random action repeat
                cmd_port,  # command port
                report_port,  # report port
            )
            process_configs.append(process_config)
        return process_configs

    arm_configs = [
        arm1,
        arm2,
        arm3,
        arm4,
        arm5,
        # badarm
    ]
    assert len(arm_configs) == 5
    if use_bad:
        print("Using an underspecified arm.")
        arm_configs.append(badarm)
        assert len(arm_configs) == 6
    if adversarial:
        print("Using an adversarial arm.")
        arm_configs.append(adversarialarm)
        assert len(arm_configs) == 6

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    epsilon = 0.1  # 0.4  # 0.25
    if random:
        epsilon = 1.0
        print("Bandit will randomly select arms!")
    window_size = 40  # 50  #TODO: Parametrize on number of arms?
    num_episodes = 800
    max_episode_length = 200
    experiment_dir = experiment_name
    bandit_mode = "cost"
    fitting_strategy = "all"
    # If running multiple scripts simultaneously, the zmq ports must be different
    # This makes the ports different based on the experiment/seed #
    experiment_index = int(experiment_dir.split("-")[-1])
    cmd_port = 7900 + ((experiment_index - 1) * 2)  # command port
    report_port = 7900 + ((experiment_index - 1) * 2 + 1)  # report port
    arms = get_processes(
        get_arms(
            arm_configs,
            num_episodes,
            max_episode_length,
            experiment_dir,
            callback=False,
            cmd_port=cmd_port,
            report_port=report_port,
        ),
        add_stability=use_stability,
    )

    opt = EpsilonGreedySlidingWindowUCBBanditOptimizer(
        len(arms), window_size=window_size, epsilon=epsilon
    )
    bandit_callbacks = None
    if plot:
        figure, axes = plt.subplots(ncols=2, figsize=(12, 7))
        upc = UCBProbabilityCallback(axis=axes[-2], side_by_side=True)
        asc = ArmSelectionCallback(axis=axes[-1])
        bandit_callbacks = [upc, asc]
    # 10 episodes to minimum eps
    eps_scheduler = LinearRateScheduler(0.9, -0.9 / 10, min=0.01)
    bandit = LoopMultiArmedBandit(
        arms,
        opt,
        max_episode_length=max_episode_length,
        mode=bandit_mode,
        fitting_strategy=fitting_strategy,
        epsilon_rate_scheduler=eps_scheduler,
        true_cost_function=bandit_cost,
        callbacks=bandit_callbacks,
        plant=CartPoleUnboundedPlant(swingup=True),
        experiment_path=experiment_dir,
        command_port=cmd_port,
        report_port=report_port,
    )

    # Run the experiment
    loop = BanditLoop(bandit)
    best_arm = loop.fit(num_episodes)

    was_stability_arm = False
    if use_stability and best_arm == loop.bandit.stability_arm_index:
        was_stability_arm = True

    results = {
        "rewards_per_arm": loop.bandit.reward_per_arm,
        "chosen_arms": loop.chosen_arms,
        "bandit_reward": loop.collected_rewards,
        "best_arms": loop.best_arms,
        "upper_bounds_over_time": loop.ucb,
        "best_final_arm": best_arm,
        "final_arm_probs": loop.bandit.optimizer.arm_probabilities,
        "stability_arm_was_best": was_stability_arm,
    }
    if was_stability_arm:  # use the config of the parent arm
        config_arm = loop.bandit.stability_arm_history[-1]
        results["best_arm_cost_func"] = arm_configs[config_arm].cost_function
    else:
        results["best_arm_cost_func"] = arm_configs[best_arm].cost_function
    if use_stability:
        results["chosen_stability_arms"] = loop.bandit.stability_arm_history

    config = {
        "n_arms": len(arms),
        "bandit_epsilon": epsilon,
        "episode_length": max_episode_length,
        "num_episodes": num_episodes,
        "window_size": window_size,
        "bandit_mode": bandit_mode,
        "fitting_strategy": fitting_strategy,
        "used_stability": use_stability,
    }

    with open(join(experiment_dir, "bandit_config.p"), "wb") as p:
        pickle.dump(config, p)
    with open(join(experiment_dir, "results.p"), "wb") as p:
        pickle.dump(results, p)
    with open(join(experiment_dir, "arms.p"), "wb") as p:
        pickle.dump(arm_configs, p)


if __name__ == "__main__":
    import argparse, sys

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "experiment", help="The main directory name to store the results"
        )
        parser.add_argument(
            "-s", "--stability", action="store_true", help="Use the stability arm"
        )
        parser.add_argument("-b", "--badarm", action="store_true", help="Use a bad arm")
        parser.add_argument(
            "-r", "--random", action="store_true", help="Pick arms randomly"
        )
        parser.add_argument(
            "-a", "--adversarial", action="store_true", help="Use an adversarial arm"
        )
        parser.add_argument(
            "--liveplot",
            action="store_true",
            help="Plot the bandit live during training.",
        )
        args = parser.parse_args()
        name = args.experiment
        use_bad = args.badarm
        use_stability = args.stability
        random = args.random
        adversarial = args.adversarial
        plot = args.liveplot
    else:
        name = "SwingUpMP-Testeps-1"
        use_bad = False
        use_stability = False
        plot = False
        random = False
        adversarial = False

    s = time.time()
    train_bandit_loop(name, use_bad, use_stability, random, adversarial, plot)
    print(f"Total time: {round((time.time()-s)/60,2)}m")
