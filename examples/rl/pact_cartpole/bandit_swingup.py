"""Find a best controller for HH Balance with the use of a Multi Armed Bandit."""
import multiprocessing as mp
import pickle
import time
from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from cartpole_control.plant.cartpole_plant import (
    SwingupContinuousDiscreteAction,
    SwingupPlant,
    SwingupState,
)
from cartpole_control.training.swingup.cost_functions import *
from psipy.rl.bandit_loop_mp import (
    BanditLoop,
    LoopArm,
    LoopMultiArmedBandit,
    get_processes,
)
from psipy.rl.control.bandits.bandit_optimizers import (
    EpsilonGreedySlidingWindowUCBBanditOptimizer,
)
from psipy.rl.neural_trainer import NeuralTrainerProcess
from psipy.rl.visualization.bandit_plottiing_callbacks import (
    ArmSelectionCallback,
    UCBProbabilityCallback,
)

# Set CPU as available physical device
tf.config.set_visible_devices([], "GPU")


# Define each arm here and load them into processes in the training function
arm1 = LoopArm(
    gamma=0.99,
    epsilon_init=0.9,
    epsilon_decay=-0.3,
    epsilon_min=0.01,
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
    epsilon_init=0.9,
    epsilon_decay=-0.3,
    epsilon_min=0.01,
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
    epsilon_init=0.9,
    epsilon_decay=-0.3,
    epsilon_min=0.01,
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
    epsilon_init=0.9,
    epsilon_decay=-0.3,
    epsilon_min=0.01,
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
    epsilon_init=0.9,
    epsilon_decay=-0.3,
    epsilon_min=0.01,
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


arm_configs = [
    arm1,
    arm2,
    arm3,
    arm4,
    arm5,
]
assert len(arm_configs) == 5


VELOCITIES = [
    100,
    400,
    800,
    1000,
    200,
    20,
    600,
]


def train_bandit_loop(experiment_name: str, use_stability: bool, plot: bool):
    def get_arms(
        arm_configs: List[LoopArm],
        episodes: int,
        max_steps: int,
        experiment_dir: str,
        callback: bool = False,
    ) -> List[Tuple]:
        process_configs: List[Tuple] = []
        for i, config in enumerate(arm_configs):
            process_config = (
                i,  # index
                config,  # LoopArm configuration
                NeuralTrainerProcess,
                SwingupPlant,
                SwingupState,
                SwingupContinuousDiscreteAction,
                "direction",  # action channel
                episodes,  # num episodes
                max_steps,  # max episode length
                experiment_dir,  # top level experiment dir
                callback,  # plot q curves
                5,  # random action repeat
            )
            process_configs.append(process_config)
        return process_configs

    mp.set_start_method("spawn")
    epsilon = 0.1  # 0.4  # 0.25
    window_size = 40  # 50  #TODO: Parametrize on number of arms?
    num_episodes = 400
    max_episode_length = 200
    experiment_dir = experiment_name
    bandit_mode = "cost"
    fitting_strategy = "all"
    figure, axes = plt.subplots(ncols=2, figsize=(12, 7))
    arms = get_processes(
        get_arms(
            arm_configs,
            num_episodes,
            max_episode_length,
            experiment_dir,
            callback=False,
        ),
        add_stability=use_stability,
    )

    opt = EpsilonGreedySlidingWindowUCBBanditOptimizer(
        len(arms), window_size=window_size, epsilon=epsilon
    )
    bandit_callbacks = None
    if plot:
        upc = UCBProbabilityCallback(axis=axes[-2], side_by_side=True)
        asc = ArmSelectionCallback(axis=axes[-1])
        bandit_callbacks = [upc, asc]
    bandit = LoopMultiArmedBandit(
        arms,
        opt,
        max_episode_length=max_episode_length,
        mode=bandit_mode,
        fitting_strategy=fitting_strategy,
        true_cost_function=bandit_cost,
        callbacks=bandit_callbacks,
        plant=SwingupPlant(),
        experiment_path=experiment_dir,
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
        parser.add_argument(
            "--liveplot",
            action="store_true",
            help="Plot the bandit live during training.",
        )
        args = parser.parse_args()
        name = args.experiment
        use_stability = args.stability
        plot = args.liveplot
    else:
        name = "SwingUpMP-6arms-TEST"
        use_stability = False
        plot = True

    s = time.time()
    train_bandit_loop(name, use_stability, plot)
    print(f"Total time: {round((time.time()-s)/60,2)}m")
