import logging
import pickle
import time
from collections import defaultdict, deque
from multiprocessing import Process
from os.path import join
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type, Union

import numpy as np
import zmq

from psipy.core.rate_schedulers import RateScheduler
from psipy.rl import Loop
from psipy.rl.control import NFQ
from psipy.rl.control.bandits.bandit_optimizers import (
    BanditOptimizer,
    EpsilonGreedySlidingWindowUCBBanditOptimizer,
)
from psipy.rl.control.bandits.multiarmed_bandits import MultiArmedBandit
from psipy.rl.neural_trainer import NeuralTrainerProcess, ResetParams
from psipy.rl.plant import Action, Plant, State
from psipy.rl.visualization import PlottingCallback

LOG = logging.getLogger(__name__)


class LoopArm(NamedTuple):
    """Configuration of NFQ for multiprocessed bandit arms.

    Args:
        gamma: Discount factor
        cost_function: Cost function to be used to train the controller.
                       Be sure that all cost functions have the same max
                       expected value!
        iterations: NFQ iterations
        epochs: NFQ epochs
        lookback: Lookback
        reset_params: A :class:`ResetParams` tuple or None, if NFQ should
                      not reset.
        prioritization: Prioritization type, if desired
        neural_structure: Tuple of number of layers, number of neurons per layer
        double: Use double Q learning in NFQ
        dueling: Use a dueling architecture in NFQ
        only_newest: Use only this many latest episodes in the batch
        scale: Whether or not to scale the q target values
    """

    gamma: float
    cost_function: Callable
    iterations: int
    epochs: int
    batchsize: int
    lookback: int
    reset_params: Optional[ResetParams]
    prioritization: Optional[str]
    neural_structure: Tuple[int, int]
    double: bool
    dueling: bool
    only_newest: Optional[int]
    scale: bool


class LoopMultiArmedBandit(MultiArmedBandit):
    """A multi-armed bandit that controls multiple reinforcement learning loops.

    The bandit chooses which loop to control an episode, where each loop can use
    a different controller. Over time, the controllers that show more promise are
    used for evaluation more often. Controllers use the data collected from every
    other controller. This aids in exploration.

    Training can occur after every episode or JIT (just in time). The JIT version
    will only train the controller once after it has been pulled. Training is
    multiprocessed, while interaction is not. Interaction must stay sequential since
    real systems can not be parallelized. The arms are Processes that run the
    function :meth:`open_arm_process` and are configured by :class:`LoopArm` named
    tuples.

    An extra arm can be added to the provided arms, which is the "stability arm". This
    is always the last arm, and will be the best model without exploration and without
    training. The idea of this arm is to stabilize training by providing a greater
    number of useful episodes. Experiments showed that this does not add much benefit,
    but if your policies are easily degrading it could be helpful.

    A sleep is necessary to allow all processes to start up, before any episodes can
    be run. If the main process hangs while waiting for an episode to start, the sleep
    was too short.

    Note: :class:`CycleManager` timers must be disabled in order to run multiprocessed
    loop instances!

    Args:
        arms: Tuple of Processes, one for each arm, plus an extra string at the end
              called "Stability" if the stability arm is desired (this is taken care of
              by the :meth:`get_processes` function.
        optimizer: The bandit optimizer. A nonstationary optimizer is highly recommended.
        plant: The instantiated plant.
        experiment_path: Path to the base experiment folder, where arm data is saved.
        max_episode_length: Max episode length for the plant.
        mode: Learning progress measurement. Currently only ``cost`` is supported.
        fitting_strategy: ``all`` or JIT (``single``) training.
        command_port: ZMQ port for sending commands to the arms.
        report_port: ZMQ port for receiving status and info from the arms.
        true_cost_function: Cost function that the bandit uses to score episodes.
        callbacks: Any number of multiarmed bandit callbacks.
    """

    def __init__(
        self,
        arms: Tuple[Union[Process, str], ...],
        optimizer: BanditOptimizer,
        plant: Plant,
        experiment_path: str,
        max_episode_length: int,
        mode: str,
        epsilon_rate_scheduler: Optional[RateScheduler],
        command_port: int = 7900,
        fitting_strategy: str = "all",
        report_port: int = 7901,
        true_cost_function: Optional[Callable[[List[np.ndarray]], np.ndarray]] = None,
        callbacks: Optional[List] = None,
    ):
        assert mode in ["cost"]  # , "LP"]
        assert fitting_strategy in ["single", "all"]
        self.has_stability_arm = arms[-1] == "Stability"
        self.num_processes = len(arms) - self.has_stability_arm
        assert all([isinstance(arm, Process) for arm in arms[: self.num_processes]])
        super().__init__(arms, optimizer, callbacks=callbacks)

        # Set up zmq comms between processes
        context = zmq.Context()
        self.cmd_socket = context.socket(zmq.PUB)
        self.report_socket = context.socket(zmq.SUB)
        self.cmd_socket.bind("tcp://*:%s" % command_port)
        self.report_socket.bind("tcp://*:%s" % report_port)
        self.report_socket.setsockopt_string(zmq.SUBSCRIBE, f"ready")
        # Subscribe to report topics for the arms that send reports (not stability)
        for i in range(self.num_processes):
            self.report_socket.setsockopt_string(zmq.SUBSCRIBE, f"Arm{i}-report")

        # Start each NeuralTrainer process
        for arm in arms[: self.num_processes]:
            arm.start()

        self.plant = plant
        self.epsilon_rate_scheduler = epsilon_rate_scheduler
        self.experiment_path = experiment_path
        self.max_steps = max_episode_length
        self.reward_per_arm: Dict[int, List[float]] = defaultdict(list)
        # A count of how many episodes each arm was run
        self.arm_episodes = {i: 0 for i in range(len(arms))}
        self.mode = mode
        self.fitting_strategy = fitting_strategy
        # Cost function that the bandit will use, instead of
        # the costs coming from the arms
        self.true_cost_function = true_cost_function

        if self.has_stability_arm:
            # Stability arm tracking
            print(self.sign_log("Using a stability arm!"))
            self.stability_arm = None
            self.stability_arm_index = len(arms) - 1  # last arm, array indices
            self.stability_arm_history: List[Optional[int]] = []
            self.stability_arm_costs: List[float] = []

        # Wait until all processes are open
        time.sleep(1.75 * len(arms))

    def finish(self) -> None:
        """Tell all arm processes to shut down."""
        for i, arm in enumerate(self.arms[: self.num_processes]):
            print(self.sign_log(f"Telling Arm{i} to finish."))
            self.cmd_socket.send_string(f"Arm{i} finish")
            arm.join()

    def set_then_update_global_epsilon(self) -> None:
        """Set epsilon in all arms to same value, then update for the next episode."""
        if self.epsilon_rate_scheduler is not None:
            # If already at the minimum value, skip this step
            epsilon = self.epsilon_rate_scheduler.current_value
            # Strip the decimal from the float; it is added back in the arms
            assert str(epsilon)[:2] == "0."  # sanity check
            epsilon = str(epsilon)[2:]
            for i, arm in enumerate(self.arms[: self.num_processes]):
                self.cmd_socket.send_string(f"Arm{i} eps{epsilon}")
                self.wait_until_ready(i)
            self.epsilon_rate_scheduler.update()

    def fit_single_arm(self, arm) -> None:
        """Tell a single process to train and wait until it is finished."""
        print(self.sign_log(f"Telling only Arm{arm} to train."))
        self.cmd_socket.send_string(f"Arm{arm} train")
        self.wait_until_ready(arm)

    def fit_all_arms(self) -> None:
        """Tell all processes to train and wait until they are all finished."""
        for i in range(self.num_processes):
            print(self.sign_log(f"Telling Arm{i} to train."))
            self.cmd_socket.send_string(f"Arm{i} train")
        self.wait_until_ready()

    def sign_log(self, string: str) -> str:
        """Adds the signature of the bandit to the front of a log message."""
        return f"[BANDIT]: {string}"

    def wait_until_ready(self, arm: Optional[int] = None) -> None:
        needs_to_be_ready = set(range(self.num_processes))
        if arm is not None:
            needs_to_be_ready = {arm}
        ready_arms: Set[int] = set()
        while ready_arms != needs_to_be_ready:
            ready_msg = self.report_socket.recv_string()
            ready, arm = ready_msg.split(" ")
            assert ready == "ready", f"Message out of order ({ready} != 'ready')"
            print(self.sign_log(f"Got ready notice from arm {arm}"))
            ready_arms.add(int(arm))

    def set_stability_arm(self, cost, arm) -> None:
        """Set the stability arm if certain conditions are met."""
        # If not all real arms have been pulled at least once, do not continue
        if (
            not set(range(self.optimizer._num_arms - 1))
            <= self.optimizer._optimized_arms
        ):
            print(
                self.sign_log(
                    "Not setting leading arms yet because some arms were not pulled"
                )
            )
            return

        cost_to_beat = np.inf
        if self.stability_arm is not None:
            cost_to_beat = sum(self.stability_arm_costs) / len(self.stability_arm_costs)
            print(f"\t\tSetting stability arm when {cost} < {cost_to_beat}")
        if self.stability_arm is None or cost < cost_to_beat:
            print(
                f"\t\tSetting stability arm as arm {arm} (#{len(self.stability_arm_history)})"
            )
            if cost < cost_to_beat:
                self.stability_arm = arm
            else:
                # Take the current best arm. All arms at this point will have only
                # 1 pull, assuming UCB bandit optimization.
                sorted_best_arm_args = np.argsort(self.optimizer.arm_probabilities[:-1])
                self.stability_arm = sorted_best_arm_args[-1]
            self.stability_arm_costs = [cost]  # reset the stability arm's costs
            print(
                self.sign_log(
                    f"Telling Arm{self.stability_arm} to save as stability model."
                )
            )
            self.cmd_socket.send_string(f"Arm{self.stability_arm} save")

            # Wait until the model is saved
            self.wait_until_ready(self.stability_arm)
            print(self.sign_log(f"Model finished saving."))

            # Set the confidence bound for the stability arm to be the same as the
            # confidence bound of the arm it is a copy of
            assert isinstance(
                self.optimizer, EpsilonGreedySlidingWindowUCBBanditOptimizer
            )
            maxlen = self.optimizer.rewards[0].maxlen
            self.optimizer.rewards[-1] = deque([-cost], maxlen=maxlen)
            self.optimizer.arm_pulls[-1] = deque([1], maxlen=maxlen)
            print(self.optimizer.rewards[-1])
            print(self.optimizer.arm_pulls[-1])

    def start_stability_loop(
        self, final_eval: bool = False
    ) -> Tuple[float, int, List[State]]:
        """Run an episode with the main process stability arm."""
        # Note: this arm isn't trained so it doesn't need a separate process
        control = NFQ.load(
            join(
                self.experiment_path,
                "arm-stability",
                f"stability-model-Arm{self.stability_arm}.zip",
            )
        )
        control.epsilon = 0
        logdir = join(self.experiment_path, "bandit_data")
        name = str(self.stability_arm) + "-stability"
        if final_eval:
            logdir = join(logdir, "evaluation")
            name += "-evaluation"
        loop = Loop(self.plant, control, logdir=logdir, name=name)
        loop.run_episode(1, max_steps=self.max_steps)
        metrics = loop.metrics[1]
        traj = loop.trajectory
        cost = metrics["total_cost"]
        cycles = metrics["cycles_run"]
        return cost, cycles, traj

    def start_loop(
        self, arm: int, final_eval: bool = False
    ) -> Tuple[float, int, List[State]]:
        """Run an episode with a specific process and wait for performance report."""
        print(self.sign_log(f"Telling {arm} to run a loop."))
        if self.has_stability_arm and arm == self.stability_arm_index:
            cost, cycles, traj = self.start_stability_loop()
        else:
            run_command = "run" if not final_eval else "run-eval"
            self.cmd_socket.send_string(f"Arm{arm} {run_command}")
            print(self.sign_log(f"Waiting for report on topic Arm{arm}-report..."))
            # Will always get the reward for the current run because no other runs
            # can take place before it gets this reward
            topic, cost, cycles, traj = self.report_socket.recv_multipart()
            topic = topic.decode("utf-8")
            cost = float(cost.decode("utf-8"))  # type: ignore
            cycles = int(cycles.decode("utf-8"))  # type: ignore
            assert topic == f"Arm{arm}-report"
            traj = pickle.loads(traj)
            print(self.sign_log("Got report."))
        return cost, cycles, traj

    def choose_arm(self) -> float:
        """Select an arm, evaluate it, and then update the optimizer's parameters.

        The reward for the arm is returned.
        """
        self.set_then_update_global_epsilon()
        arm = self.optimizer.select_arm()
        reward = self._evaluate_arm(arm)
        self.optimizer.optimize(arm, reward)

        # If the stability arm has not yet been set, set it;
        # If it has, only check for updating it when a normal arm is pulled
        # and it did not get cost greater than the sigmoid (in all cases this
        # is a terminal case; however not all terminals result in > 1 cost).
        if self.has_stability_arm and (
            (arm != self.stability_arm_index and -reward < 1)
            or self.stability_arm is None
        ):
            # Negative for cost
            self.set_stability_arm(-self.optimizer.arm_probabilities[arm], arm)

        self._arm_history.append(arm)
        if self.has_stability_arm:
            self.stability_arm_history.append(self.stability_arm)
        LOG.info(f"Arm probabilities: {self.optimizer.arm_probabilities}")
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.plot(self)
        return reward

    def _calc_bandit_reward(
        self, total_cost: float, total_cycles: int, trajectory: List[State]
    ) -> float:
        """Calculate the reward for the bandit from the loop results."""
        traj = [t.as_array() for t in trajectory]
        # Reward is the average cost over the entire episode;
        # this thereby deals with variable length episodes
        cost = total_cost / total_cycles
        if self.true_cost_function is not None:
            true_cost = self.true_cost_function(traj)
            cost = np.sum(true_cost) / len(true_cost)
        if trajectory[-1].terminal:
            cost *= 2  # add a lot of cost for terminal #TODO Relogic?
        return -cost

    def _evaluate_arm(self, arm: int) -> float:
        """Run the selected arm and train arms afterwards."""
        print(self.sign_log(f"Pulled arm: {arm}"))
        self.arm_episodes[arm] += 1
        total_cost, total_cycles, trajectory = self.start_loop(arm)

        if self.mode == "cost":
            reward = self._calc_bandit_reward(total_cost, total_cycles, trajectory)
            cost = -reward
            print(f"Got reward: {reward} with arm {arm}.")

        if self.fitting_strategy == "single" and (
            self.has_stability_arm and arm != self.stability_arm_index
        ):
            self.fit_single_arm(arm)
        else:
            self.fit_all_arms()

        # print(f"cnvrg_linechart_BanditCost value: {cost}")
        self.reward_per_arm[arm].append(reward)

        if self.has_stability_arm and arm == self.stability_arm_index:
            self.stability_arm_costs.append(cost)

        return reward

    def finalize_best_arm(self, num_runs: int) -> int:
        """Evaluates all arms on the plant, returning the best performing arm index.

        Args:
            num_runs: How many times to evaluate an arm to average out behavior with
                      different starting conditions.
        """
        costs = []
        for arm in range(self.num_processes):
            arm_costs = []
            for _ in range(num_runs):
                total_cost, total_cycles, trajectory = self.start_loop(
                    arm, final_eval=True
                )
                cost = -self._calc_bandit_reward(total_cost, total_cycles, trajectory)
                if trajectory[-1].terminal:
                    cost = np.inf
                arm_costs.append(cost)
            costs.append(np.mean(arm_costs))

        if self.has_stability_arm:
            total_cost, total_cycles, trajectory = self.start_stability_loop(
                final_eval=True
            )
            cost = -self._calc_bandit_reward(total_cost, total_cycles, trajectory)
            if trajectory[-1].terminal:
                cost = np.inf
            costs.append(cost)

        return int(np.argmin(costs))

    @property
    def total_episodes(self) -> int:
        return sum(v for v in self.arm_episodes.values())


class BanditLoop:
    """Wrapper to run multiprocessed :class:`LoopMultiArmedBandit`."""

    def __init__(self, bandit: LoopMultiArmedBandit, final_eval_runs: int = 1):
        assert hasattr(
            bandit.optimizer, "upper_bounds"
        ), "Must use a UCB based optimizer."
        self.bandit = bandit
        self._rewards: List[float] = []
        self.best_arms: List[int] = []
        self.ucb: Dict[int, List[float]] = defaultdict(list)
        self.mean_rewards: Dict[int, List[float]] = defaultdict(list)
        self.final_best_arm: Optional[int] = None
        self.final_eval_runs = final_eval_runs

    def fit(self, steps: int) -> int:
        """Run the bandit for n steps and return the best arm."""
        for step in range(steps):
            self._rewards.append(self.bandit.choose_arm())
            self.best_arms.append(self.bandit.get_best_arm())
            ucb = self.bandit.optimizer.upper_bounds
            mean_reward = self.bandit.optimizer.arm_probabilities
            for a in range(len(self.bandit.arms)):
                self.ucb[a].append(ucb[a])
                self.mean_rewards[a].append(mean_reward[a])
        self.final_best_arm = self.bandit.finalize_best_arm(self.final_eval_runs)
        self.bandit.finish()

        return self.final_best_arm

    @property
    def collected_rewards(self) -> List[float]:
        return self._rewards

    @property
    def chosen_arms(self) -> List[int]:
        return self.bandit.arm_history


def open_arm_process(
    index: int,
    config: LoopArm,
    trainer_type: Type[NeuralTrainerProcess],
    plant: Plant,
    state: State,
    action: Action,
    action_channel: str,
    num_episodes: int,
    max_episode_steps: int,
    experiment_dir: str,
    use_callback: bool = False,
    random_act_repeat: int = 0,
    cmd_port: int = 7900,
    report_port: int = 7901,
) -> None:
    """Opens a process running a NeuralTrainer with the given config.

    Args:
        index: The index of the arm, i.e. 0, 1, 2, ...
        config: The arm's :class:`LoopArm` configuration
        trainer_type: The configured :class:`NeuralTrainerProcess`
        plant: The plant to run in
        state: The plant's state
        action: The controller's action
        action_channel: The action channel to be controlled
        num_episodes: Number of episodes to run
        max_episode_steps: Max episode steps
        experiment_dir: Top level directory to save results
        use_callback: True to plot Q values during training
        random_act_repeat: Potentially repeat actions when they are random
        cmd_port: Port for receiving commands from the main thread
        report_port: Port for sending reports from the process to main

    """
    save_dir = join(experiment_dir, f"arm-{index}")
    sart_dir = join(experiment_dir, "bandit_data")
    callback = []
    if use_callback:
        callback = [
            PlottingCallback(
                ax1="q",
                is_ax1=lambda x: x.endswith("q"),
                ax2="mse",
                is_ax2=lambda x: x == "loss",
                plot_freq="end",
                dims=(5, 3),
                title=f"Arm-{index}",
            )
        ]
    trainer = trainer_type(
        index=index,
        stability_model_dir=join(experiment_dir, "arm-stability"),
        command_port=cmd_port,
        report_port=report_port,
        mode="growing",
        plant=plant,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        sart_dir=sart_dir,
        training_curve_save_directory=save_dir,
        save_dir=join(save_dir, "models"),
        render=False,
        name=str(index),
        callbacks=callback,
    )

    trainer.initialize_control(
        control_type=NFQ,
        neural_structure=config.neural_structure,
        state_channels=state.channels(),
        action=action,
        action_channel=action_channel,
        lookback=config.lookback,
        iterations=config.iterations,
        epochs=config.epochs,
        minibatch_size=config.batchsize,
        gamma=config.gamma,
        costfunc=config.cost_function,
        epsilon_decay_scheduler=None,  # LinearRateScheduler(0.9, -0.45, min=0.01),
        prioritization=config.prioritization,
        reset_params=config.reset_params,
        scale=config.scale,
        double=config.double,
        dueling=config.dueling,
        batch_only_newest=config.only_newest,
        random_action_repeat=random_act_repeat,
    )
    trainer.run()


def get_processes(
    process_configs: List[Tuple], add_stability: bool = True
) -> Tuple[Union[Process, str], ...]:
    """Create n many processes for n arms, plus an extra 'stability' arm.

    Feed the result of this function into the :class:`LoopMultiArmedBandit`.

    Args:
        process_configs: list of :class:'LoopArm' config tuples
        add_stability: True if a stability arm should be added
    """
    processes: List[Union[Process, str]] = []
    for config in process_configs:
        processes.append(Process(target=open_arm_process, args=config))
    if add_stability:
        processes.append("Stability")
    return tuple(processes)
