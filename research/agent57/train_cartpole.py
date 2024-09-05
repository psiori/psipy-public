# PSIORI PACT
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""WIP Script that trains NGU on Cartpole."""

if __name__ == "__main__":

    import logging
    import os
    import sys
    from datetime import datetime

    import numpy as np

    from psipy.core.utils import git_commit_hash
    from psipy.rl import Loop
    from psipy.rl.control.agent57.ngu import NGU, NGUPlantMixin, NGUState
    from psipy.rl.control.agent57.utils.callbacks import MultiLossCallback
    from psipy.rl.io.batch import Batch
    from psipy.rl.loop import LoopPrettyPrinter
    from psipy.rl.plant import TState
    from psipy.rl.plant.gym.cartpole_plants import (
        CartPoleAction,
        CartPoleSwingUpPlant,
    )  # CartPoleGymDiscretizedAction,; CartPolePlant,; CartPoleBalancePlant,
    from psipy.rl.visualization import PlottingCallback

    def reward_func(states: np.ndarray) -> np.ndarray:
        theta = states[:, THETA_CHANNEL_IDX]
        costs = np.zeros(len(states)) / 200
        costs[np.abs(theta) <= 2] = C / 8  # REWARD!
        costs[np.abs(theta) <= 1] = C / 4  # REWARD!
        costs[np.abs(theta) <= 0.5] = C / 2  # REWARD!
        costs[np.abs(theta) <= 0.1] = C  # REWARD!
        return costs

    class NGUCartPoleState(NGUState):
        _channels = (
            "cart_position",
            "cart_velocity",
            "pole_angle",
            "pole_velocity",
        )

    class NGUCartPoleSwingUpPlant(NGUPlantMixin, CartPoleSwingUpPlant):
        """Cartpole, but the pole starts down and must swing up first before balancing."""

        def __init__(self, ngukwargs, cartpolekwargs):
            NGUPlantMixin.__init__(self, **ngukwargs)
            CartPoleSwingUpPlant.__init__(self, **cartpolekwargs)

        def _compute_cost(self, state: TState, cost: float) -> float:
            return reward_func(state.as_array()[None, ...])[0]

        def solve_condition(self, state: TState) -> bool:
            return super().solve_condition(state)

        renderable = True
        state_type = NGUCartPoleState
        action_type = CartPoleAction

    ###### Configuration ######
    # [AGENT]

    LOOKBACK = 1
    NEURONS_PER_LAYER = 20
    LAYERS = 2
    NORMALIZER = "max"
    # [COST]
    C = 0.005
    # [TRAINING]
    BATCHSIZE_MULTIPLE = 512
    EPISODE_LENGTH = 200
    EPOCHS = 30
    INITIAL_EPSILON = 0
    ITERATIONS = 5
    NUM_EPISODES = 500
    ###### Configuration ######

    LOG = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    #: Set the reinforcement learning agent that should be trained
    agent_name = "NGU"
    #: The folder to save data generated from this script. Includes the git commit
    #: hash for versioning. There are no checks for data compatibility! This must
    #: be assured user-side.
    date = datetime.now().strftime("%y%m%d-%H%M%S")
    #: The git commit hash for the model version being used.
    MODEL_HASH = git_commit_hash(True, fallback=True)
    # _folder_header = f"gym-adam-reset-no-1-balance-{agent_name}-"
    _folder_header = f"NGU-0-{agent_name}-"
    SART_FOLDER = f"{_folder_header}-{MODEL_HASH}"

    #: The list of folders to load data from.
    SART_FOLDERS = [SART_FOLDER]

    #: The path to the model to be trained, or to be created at this path.
    MODEL_PATH = f"{date}-balance-{agent_name}-{MODEL_HASH}.zip"
    MODEL_PATH = os.path.join(SART_FOLDER, MODEL_PATH)

    STATE_CHANNELS = (
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_velocity",
    )
    LEN_STATE = len(STATE_CHANNELS)
    THETA_CHANNEL_IDX = STATE_CHANNELS.index("pole_angle")
    CART_SPEED_IDX = STATE_CHANNELS.index("cart_velocity")
    POLE_SPEED_IDX = STATE_CHANNELS.index("pole_velocity")

    ActionType = CartPoleAction
    StateType = NGUCartPoleState

    def train(
        num_iterations: int, num_epochs: int, num_episodes: int,
    ):
        """Trains the model and returns the finished model."""
        import matplotlib.pyplot as plt

        num_mixtures = 2
        figure, ax = plt.subplots(ncols=2, figsize=(10, 5))
        multicallback2 = MultiLossCallback(ax[0], figure.number, "Intra Loss")
        multicallback3 = MultiLossCallback(ax[1], figure.number, "RND Loss")
        ngu = NGU(
            num_mixtures=num_mixtures,
            state_channels=StateType.channels(),
            action=ActionType,
            network_structure=(2, 40),
            action_values=ActionType.legal_values[0],
            lookback=LOOKBACK,
            prioritized=True,
            num_repeat=5,
        )
        ngu_plant_params = dict(
            beta=0,
            rnd_epochs=2,
            embedding_size=10,
            k_neighbors=2,
            embedding_epochs=20,
            embedding_batch_size=1024,
            callbacks=[multicallback2, multicallback3],
            max_reward_scale=5,
        )
        plant_params = dict(cost_func=reward_func, add_action_to_state=False)
        plant = NGUCartPoleSwingUpPlant(ngu_plant_params, plant_params)

        callback = PlottingCallback(
            ax1="q",
            is_ax1=lambda x: x.endswith("q"),
            ax2="mse",
            is_ax2=lambda x: x == "loss",
        )
        loop = Loop(plant, ngu, "Balance", SART_FOLDER, render=True)
        pp = LoopPrettyPrinter(reward_func)
        for cycle in range(num_episodes):
            gamma, beta = ngu.get_active_mixture()
            plant.beta = beta
            LOG.info(f"Cycle {cycle+1}")
            print(
                "Using gamma and beta", gamma, beta, "from mixture", ngu._mixture_index
            )
            loop.run_episode(cycle + 1, max_steps=EPISODE_LENGTH, pretty_printer=pp)
            if plant.is_solved:
                print("Plant solved @", cycle + 1)
                # sys.exit()

            batch = Batch.from_hdf5(
                SART_FOLDER,
                lookback=LOOKBACK,
                control=ngu,
                prioritization="proportional",
                action_channels=("move_index",),
            )
            LOG.info(f"Loaded {batch.num_samples} transitions.")

            norm = NORMALIZER
            ngu.fit_normalizer(batch.observations, method=norm)
            batch_size = min(
                BATCHSIZE_MULTIPLE * ((cycle // 20) + 1), batch.num_samples
            )
            LOG.info(f"Batch size: {batch_size}")

            pp.total_cost = 0
            pp.total_transitions = 0
            ngu.fit(
                batch,
                # costfunc=reward_func, #TODO: CANT USE BECAUSE OVERWRITES EXTRIN+INTRIN
                iterations=num_iterations,
                epochs=num_epochs,
                minibatch_size=batch_size,
                gamma=gamma,
                callbacks=[callback],
                verbose=0,
            )
            ngu.active_controller.epsilon = max(ngu.active_controller.epsilon - 0.05, 0)
            batch.set_minibatch_size(-1)
            plant.lifelong_novelty.fit(batch.states[0][:, :4])
            plant.episodic_novelty.fit_inverse_dynamics_model(
                batch.states[0][:, :4],
                batch.nextstates[0][:, :4],
                batch.states_actions[0][-1],
            )

        return model

    model = train(
        num_iterations=ITERATIONS, num_epochs=EPOCHS, num_episodes=NUM_EPISODES,
    )
    print("Training finished.")
