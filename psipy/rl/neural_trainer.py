import gc
import logging
import os
import pickle
import time
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partial
from os.path import join
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union, cast

import pandas as pd
import tensorflow as tf
import zmq
from tensorflow.keras import layers as tfkl

from psipy.core.rate_schedulers import LinearRateScheduler, RateScheduler
from psipy.rl import Loop
from psipy.rl.control import NFQ, NFQs
from psipy.rl.io.batch import Batch
from psipy.rl.loop import LoopPrettyPrinter
from psipy.rl.plant import Action, Plant, State

LOG = logging.getLogger(__name__)

__all__ = ["NeuralTrainer", "NeuralTrainerProcess", "ResetParams"]


class ResetParams(NamedTuple):
    """Parameters used to configure network resetting in :class:`NeuralTrainer`s.

    Args:
        every_n_episodes: How many episodes should pass before the network is reset.
        prefit_after_reset: Whether or not to repeat all iterations performed up until
                            the reset. For example, if the network performed
                            10 iterations and then was reset, it would perform
                            10 iterations directly after the reset in order to
                            "catch up", and then proceed as normal. When the next reset
                            comes (say, after another 10 iterations), the network will
                            prefit with 20 iterations, and so on.
    """

    every_n_episodes: int
    prefit_after_reset: bool


class NeuralTrainer:
    """Training Wrapper for NFQ and NFQs.

    The trainer abstracts away most of the boiler plate code that comes along
    with training neural models, and also ensures the proper inputs and neural
    models are used for each algorithm. Statistics of the training are saved
    for later analysis, as well as the finished models.
    
    Before use, the controller needs to be initialized via the
    :meth:`initialize_control`. This method will create the appropriate controller with
    appropriate action channel. It is separate from the main init purely for
    cleanliness reasons; controller related parameters are separate and more easy to
    reason about when in separate functions versus being all conglomerated into one
    huge init.

    This class is intended to not be used directly, but to be subclassed in most
    cases. Base metrics are recorded as the controller is trained, such as number of
    episodes completed, walltimes, and loop statistics. The :meth:`calculate_metrics`
    method can be implemented to record any specific metric needed for the training
    that is unique to the specific problem at hand. The metrics dictionary defaults
    to lists, so that new metrics that record performance over time do not need to
    instantiate their initial value. The custom metrics are mixed in with the base
    metrics during training.

    Args:
        mode: The training mode, either growing (growing batch) or batch
        plant: The initialized plant
        max_episode_steps: The maximum number of episode steps per episode
        sart_dir: The directory to store the SART data
        training_curve_save_directory: The directory to save the training q curve history to
        num_episodes: The number of episodes to run, if in batch mode this is not required
        episodes_between_fit: How many episodes to run before fit is called
        save_dir: The directory to save the model in; if None, models will not be saved
        callbacks: List of any Keras callbacks. The trainer will always use at least the
                   CSVLogger callback.
        name: The name of the experiment/training
        render: Whether or not to render the plant while it is being interacted with
        pretty_printer: Appropriate :class:`LoopPrettyPrinter`, if desired
        use_gpu: True to train the network on a GPU, if one exists
    """

    def __init__(
        self,
        mode: str,
        plant: Plant,
        max_episode_steps: int,
        sart_dir: str,
        training_curve_save_directory: str,
        num_episodes: Optional[int] = None,
        episodes_between_fit: int = 1,
        save_dir: Optional[str] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        name: Optional[str] = None,
        render: bool = False,
        pretty_printer: Optional[LoopPrettyPrinter] = None,
        use_gpu: bool = False,
    ):
        assert mode in ["batch", "growing"], "Invalid training mode."
        if not use_gpu:
            # Hide the GPU from Tensorflow to force CPU usage.
            tf.config.set_visible_devices([], "GPU")

        self.mode = mode
        self.plant = plant
        self.max_steps = max_episode_steps
        self.sart_dir = sart_dir
        self.pp = pretty_printer
        self.render = render
        self.save_dir = save_dir
        if num_episodes is not None:
            assert num_episodes % episodes_between_fit == 0, "Must be divisible."
        self.num_episodes = num_episodes
        self.eps_between_fit = episodes_between_fit

        # Create the path for the history callback
        if not os.path.exists(training_curve_save_directory):
            print(f"Creating directories '{training_curve_save_directory}'")
            os.makedirs(training_curve_save_directory)
        self.callbacks = []
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        self.callbacks.append(
            tf.keras.callbacks.CSVLogger(
                join(training_curve_save_directory, "q_history.csv"),
                separator=",",
                append=True,
            ),
        )

        self.name = name
        if name is None:
            self.name = cast(str, f"{self.plant.__class__.__name__}Trainer")

        self._control_initialized = False
        self.total_iterations_performed = 0
        self._episodes_before_reset = 0
        self.ep_num = 0

        # There are two internal loops, the train loop and the eval loop.
        # The train loop is used to collect more data to be used to train
        # the controller, while the eval loop is used to check controller
        # behavior and its data is NOT used for training.
        self.train_loop: Optional[Loop] = None
        self.eval_loop: Optional[Loop] = None
        self.metrics = defaultdict(list)
        # Instantiate the completed episodes metric as an int for later +=
        self.metrics["completed_episodes"] = 0

        #### TEMPO ####
        # self.memfigure, self.memax = plt.subplots(figsize=(5,3))
        # self.mem = []

    def initialize_control(
        self,
        control_type: Union[Type[NFQ], Type[NFQs]],
        neural_structure: Tuple[int, int],
        state_channels: Tuple[str, ...],
        action: Action,
        action_channel: str,
        lookback: int,
        iterations: int,
        epochs: int,
        minibatch_size: int,
        gamma: float,
        costfunc=None,
        epsilon_decay_scheduler: Optional[RateScheduler] = None,
        norm_method: str = "max",
        prioritization: Optional[str] = None,
        double: bool = False,
        dueling: bool = False,
        reset_params: Optional[ResetParams] = None,
        scale: bool = False,
        random_action_repeat: int = 0,
        batch_only_newest: Optional[int] = None,
    ) -> None:
        """Create and initialize the controller to be trained.
        
        This method creates and instantiates the proper controller structure and action
        channel. 
        
        Args:
            control_type: NFQ or NFQs classes
            neural_structure: Tuple of (num_layers, neurons_per_layer)
            state_channels: The state channels going into the controller
            action: The action type
            action_channel: The name of the action channel being controlled. Do not add _index!
                            This is done for you when necessary.
            lookback: The lookback
            iterations: How many DP steps to perform per fit
            epochs: How many epochs to train per DP step
            minibatch_size: How big the minibatches are
            gamma: The discount factor
            costfunc: The optional cost function
            epsilon_decay_scheduler: A rate scheduler to decay epsilon, if desired
            norm_method: The normalizer method
            prioritization: Prioritization method, if desired
            double: Whether or not to use double q learning
            dueling: Whether or not to use the dueling neural architecture (Currently experimental wrt sigmoid!)
            reset_params: Parameters denoting how the network should, if at all, be reset. See the docstring
                          of :class:`ResetParams` for more detail.
            scale: Whether or not to scale the q targets of the network. Only available in 'batch' mode
            random_action_repeat: How many times a random action can be repeated. Helpful to prevent cycling
                                  actions too fast in some problems.
            batch_only_newest: If provided, only loads the latest n episodes for training
        """
        # if scale:
        #     assert self.mode == "batch"
        self.control = self._create_controller(
            control_type,
            neural_structure,
            lookback,
            prioritization,
            state_channels,
            action,
            action_channel,
            double,
            dueling,
            scale,
            random_action_repeat,
        )
        self.state_channels = state_channels
        self.lookback = lookback
        self.iterations = iterations
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.cost_func = costfunc
        self.reset_params = reset_params
        self.batch_only_newest = batch_only_newest
        self.epsilon_scheduler = epsilon_decay_scheduler
        self.norm_method = norm_method
        self.prioritization = prioritization
        # Function to recreate the model from scratch, to ensure the network is reset.
        # This is because I do not trust the TF2.0 weight reset function just yet : )
        # Epsilon is set after this function, so that it does not show up in locals
        init_params = locals()
        init_params.pop("self")
        self._reinit_control = partial(self.initialize_control, **init_params)
        epsilon = 0
        self.control.epsilon = epsilon
        if epsilon_decay_scheduler is not None:
            epsilon = epsilon_decay_scheduler.current_value
            self.control.epsilon = epsilon
        # Scheduler to increase prioritization beta over the course of training
        self.beta_scheduler = LinearRateScheduler(0.5, 0)  # constant
        if self.num_episodes is not None and self.prioritization is not None:
            print("Using a beta scheduler.")
            self.beta_scheduler = LinearRateScheduler(
                0.5, 0.5 / self.num_episodes, max=1
            )

        self.train_loop = Loop(
            self.plant, self.control, self.name, self.sart_dir, render=self.render
        )
        self.eval_loop = Loop(
            self.plant,
            self.control,
            self.name + "-evaluation",
            join(self.sart_dir, "evaluation"),
            render=self.render,
        )

        # Record the hyperparameters of the controller
        self.metrics["iterations"] = iterations
        self.metrics["epochs"] = epochs
        self.metrics["gamma"] = gamma
        self.metrics["minibatch_size"] = minibatch_size
        self.metrics["lookback"] = lookback
        self.metrics["prioritization"] = prioritization
        self.metrics["initial_epsilon"] = epsilon
        self.metrics["norm"] = norm_method
        self.metrics["n_newest_batch"] = self.batch_only_newest
        self.metrics["reset_after_n"] = False
        self.metrics["prefit_after_reset"] = False
        if self.reset_params is not None:
            self.metrics["reset_after_n"] = self.reset_params.every_n_episodes
            self.metrics["prefit_after_reset"] = self.reset_params.prefit_after_reset

        self._control_initialized = True

    def _check_initialized(self) -> None:
        """Raises error if controller is not initialized."""
        if not self._control_initialized:
            raise RuntimeError(
                "The controller was not initialized. Run '.initialize_control'"
                " first before attempting to train."
            )

    @staticmethod
    def make_model(
        control_type: Union[Type[NFQ], Type[NFQs]],
        neural_structure: Tuple[int, int],
        n_inputs,
        n_actions,
        lookback,
        dueling,
    ):
        """Create a neural network to be used in NFQ or NFQs.

        Args:
            control_type: NFQ or NFQs classes
            neural_structure: Tuple of (num_layers, neurons_per_layer)
            n_inputs: Number of inputs; length of state channels
            n_actions: The number of possible actions (used in NFQ)
            dueling: Whether or not to use the dueling neural architecture (Currently experimental wrt sigmoid!)
        """
        state = tfkl.Input((n_inputs, lookback), name="state")
        inputs = state
        net = tfkl.Flatten()(state)
        if control_type == NFQs:
            # If NFQs, add the action as an input
            action = tfkl.Input((1,), name="actions")
            net = tfkl.Concatenate()([net, action])
            inputs = [state, action]
        for layer in range(neural_structure[0]):
            net = tfkl.Dense(neural_structure[1], activation="tanh")(net)

        # NFQs has 1 output neuron, NFQ has n_actions output neurons
        n_outputs = 1 if control_type == NFQs else n_actions

        if dueling:
            print("Creating a dueling architecture...")
            if control_type == NFQs:
                raise NotImplementedError

            value_stream = tfkl.Dense(neural_structure[1], activation="relu")(net)
            value_stream = tfkl.Dense(1, activation="linear")(value_stream)

            advantage_stream = tfkl.Dense(neural_structure[1], activation="relu")(net)
            advantage_stream = tfkl.Dense(n_outputs, activation="linear")(
                advantage_stream
            )

            net = value_stream + (advantage_stream - tf.reduce_mean(advantage_stream))
        else:
            net = tfkl.Dense(n_outputs, activation="sigmoid")(net)

        return tf.keras.Model(inputs, net)

    def _create_controller(
        self,
        control_type,
        neural_structure: Tuple[int, int],
        lookback: int,
        prioritized: Optional[str],
        state_channels: Tuple[str, ...],
        action: Action,
        action_channel: str,
        double: bool,
        dueling: bool,
        scale: bool,
        random_action_repeat: int = 0,
    ) -> Union[NFQ, NFQs]:
        """Creates the controller and sets proper action channel.

        Args:
            control_type: NFQ or NFQs classes
            neural_structure: Tuple of (num_layers, neurons_per_layer)
            lookback: The lookback
            prioritized: Prioritization method, if desired
            state_channels: The state channels going into the controller
            action: The action type
            action_channel: The name of the action channel being controlled. Do not add _index!
                            This is done for you when necessary.
            double: Whether or not to use double q learning
            dueling: Whether or not to use the dueling neural architecture (Currently experimental wrt sigmoid!)
            scale: Whether or not the scale the q targets into the range [0,1].
            random_action_repeat: How many times a random action can be repeated. Helpful to prevent cycling
                                  actions too fast in some problems.
        """
        if control_type.__name__ == "NFQ":
            print("Creating an NFQ model...")
            # NFQ trains on the action index value
            self.action_channel = action_channel + "_index"
        elif control_type.__name__ == "NFQs":
            print("Creating an NFQs model...")
            # NFQs trains directly on the action value
            self.action_channel = action_channel
        else:
            raise ValueError(f"Invalid control type. ({control_type})")
        print(f"Action channel is {self.action_channel}")
        model = self.make_model(
            control_type,
            neural_structure,
            len(state_channels),
            len(action.legal_values[0]),
            lookback,
            dueling,
        )
        control = control_type(
            model=model,
            state_channels=state_channels,
            action=action,
            action_values=action.legal_values[0],
            lookback=lookback,
            prioritized=prioritized,
            doubleq=double,
            num_repeat=random_action_repeat,
            scale=scale,
        )
        return control

    @abstractmethod
    def calculate_custom_metrics(self, episode: Optional[int] = None) -> None:
        """Problem specific metrics can be calculated with this method.

        Place the metrics directly into `self.metrics`. If the desired metric
        is dependent on the episode, it can be attached to said episode creating a
        new key under the `episode_{episode_number}` key.
        """
        ...

    def run_loop(self, evaluation: bool = False) -> Tuple[float, int, List[State]]:
        """Run the loop for a single episode.

        Args:
            evaluation: Whether or not this is an evaluation episode.
                        Eval episodes' data are not used for training.
        """
        self._check_initialized()
        if evaluation:
            print("Running an evaluation loop...")
            self.eval_loop.run_episode(
                -1, max_steps=self.max_steps, pretty_printer=self.pp
            )
            trajectory = self.eval_loop.trajectory
        else:
            self.ep_num += 1
            self._episodes_before_reset += 1
            print(f"Running episode {self.ep_num}...")
            self.train_loop.run_episode(
                self.ep_num, max_steps=self.max_steps, pretty_printer=self.pp
            )
            trajectory = self.train_loop.trajectory
            self.metrics["completed_episodes"] += 1
            # Extract metrics: total cost, total cycles, wall time
            self.metrics[f"episode_{self.ep_num}"] = self.train_loop.metrics[
                self.ep_num
            ]
            # Return metrics for when this method is run directly
            metrics = self.train_loop.metrics[self.ep_num]
            return metrics["total_cost"], metrics["cycles_run"], trajectory
        return (-1, -1, trajectory)

    def fit_on_batch(self, episode: Optional[int] = None) -> None:
        """Fit on a set batch of data.

        The data must exist first before attempting to fit.
        """
        if episode is None:
            episode = self.ep_num if self.mode == "growing" else None
        batch = Batch.from_hdf5(
            self.sart_dir,
            state_channels=self.state_channels,
            action_channels=(self.action_channel,),
            lookback=self.lookback,
            control=self.control,
            prioritization=self.prioritization,
            only_newest=self.batch_only_newest,
        )
        if self.prioritization is not None:
            # Update prioritization parameters
            batch.alpha = 0.5
            batch.beta = self.beta_scheduler.current_value
            print(f"Prioritization parameters: alpha={batch.alpha}; beta={batch.beta}")
        print(
            f"Batch training on {batch.num_samples} transitions ({batch.num_episodes} episodes)..."
        )
        self.control.fit_normalizer(batch.observations, method=self.norm_method)

        if (
            self.reset_params
            and batch.num_episodes % self.reset_params.every_n_episodes == 0
            and self._episodes_before_reset != 0
        ):  # TODO:this implies at least one episode to run before resetting, not entirely true but better than the deadlock
            print(
                f"Resetting model ({self._episodes_before_reset} episodes passed)..."
            )  # TODO This actually doesn't work because if the batch stays the same size then it will constantly reset and NEVER STOPPPPPPPPPPPPPP
            self._reinit_control(epsilon_decay_scheduler=None)
            self._episodes_before_reset = 0
            if self.reset_params.prefit_after_reset:
                print(f"Prefitting for {self.total_iterations_performed} iterations...")
                self.control.fit(
                    batch,
                    iterations=self.total_iterations_performed,
                    epochs=self.epochs,
                    minibatch_size=self.minibatch_size,
                    gamma=self.gamma,
                    callbacks=self.callbacks,
                    costfunc=self.cost_func,
                    verbose=0,
                )
                # Reset scaling parameters
                if isinstance(self.control, NFQ):  # not yet implemented in NFQs
                    self.control.A = 1
                    self.control.B = 0
        self.control.fit(
            batch,
            iterations=self.iterations,  # 1 * episode//10 + 1
            epochs=self.epochs,
            minibatch_size=self.minibatch_size,
            gamma=self.gamma,
            callbacks=self.callbacks,
            costfunc=self.cost_func,
            verbose=0,
        )
        self.total_iterations_performed += self.iterations

        self.calculate_custom_metrics(episode)
        self.save_current_model(episode_number=episode)
        self.beta_scheduler.update(episode)

        # ###### MEM PLOTTING, REMOVE ######
        # self.mem.append(psutil.Process(os.getpid()).memory_info().rss / 1e9)
        # self.memax.clear()
        # self.memax.plot(self.mem)
        # self.memax.set_title(f"Memory for Process {self.index}")
        # plt.pause(.01)
        # ##################################
        # Collect garbage and clear Keras session due to memory leak in TF 2.0
        gc.collect()
        tf.keras.backend.clear_session()

    def _growing_batch_fit(self):
        """Fit sequentially on a sequence of collected episodes."""
        for i in range(1, (self.num_episodes + 1) // self.eps_between_fit):
            for episode in range(1, self.eps_between_fit + 1):
                episode = i + episode - 1  # keep track in multiple episode per fit case
                self.run_loop()
            print(f"Training after episode {episode}...")
            start_fit = time.time()
            self.fit_on_batch()
            self.metrics[f"episode_{episode}"]["fit_time_s"] = round(
                time.time() - start_fit, 4
            )
            if self.epsilon_scheduler is not None:
                self.control.epsilon = self.epsilon_scheduler.update(i)
                print("New epsilon:", self.control.epsilon)

    def train(self) -> Dict[str, Any]:
        """Train the model with the training mode of the trainer."""
        self._check_initialized()
        start = time.time()
        if self.mode == "growing":
            assert self.num_episodes is not None
            self._growing_batch_fit()
            print("Training complete.")
        else:  # batch mode
            self.fit_on_batch()
            self.metrics["completed_episodes"] = None  # no episodes are run in batch
            print("Batch training complete.")

        self.metrics["wall_time_s"] = round(time.time() - start, 4)
        return self.metrics

    def save_current_model(
        self, path: Optional[str] = None, episode_number: Optional[int] = None
    ) -> None:
        """Saves the current version of the model to the given path."""
        if path is None and self.save_dir is None:
            return  # saving is disabled
        if path is None:
            path = self.save_dir
        name = self.name
        if episode_number is not None:
            name = f"{name}_{episode_number}.zip"
        self.control.save(join(path, name))

    def metrics_to_csv(self, base_path: str) -> None:
        """Convert the metrics dictionary to a csv which saves in the base folder."""
        metrics = deepcopy(self.metrics)
        episode_metrics = []
        for key in reversed(list(metrics.keys())):
            if isinstance(metrics[key], dict) and key.startswith("episode"):
                df = pd.DataFrame(metrics[key], index=[0])
                episode_metrics.append(df)  # TODO: This is reversed?
                # Remove the key so that the global df below does not include it
                metrics.pop(key)
        df = pd.DataFrame(metrics, index=[0])
        df.to_csv(join(base_path, "global_metrics.csv"))
        if self.mode != "batch":
            episode_df = pd.concat(episode_metrics).reset_index(drop=True)
            episode_df.to_csv(join(base_path, "episode_metrics.csv"))


class NeuralTrainerProcess(NeuralTrainer):
    """Neural trainer that can be used in a process and communicates via ZMQ.

    This trainer can be used in a process, for example with concurrent training
    of multiple models and communicates with the main process via zmq. The index
    provided is used in the topic for receiving commands and reporting costs.

    This class is specialized towards the multi armed bandit case, in which
    this process acts as an arm. However, it can still be used separately and
    only requires some boilerplate interaction code.

    Args:
        index: The arm index
        command_port: Port to listen for commands
        report_port: Port to send back run results
        stability_model_dir: Directory to store the model if chosen by the bandit
        *args, **kwargs: Args and kwargs for :class:`NeuralTrainer`
    """

    def __init__(
        self,
        index: int,
        stability_model_dir: str,
        command_port: int = 7900,
        report_port: int = 7901,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        context = zmq.Context()
        self.cmd_socket = context.socket(zmq.SUB)
        self.report_socket = context.socket(zmq.PUB)
        self.index = f"Arm{index}"
        self.cmd_socket.setsockopt_string(zmq.SUBSCRIBE, self.index)
        self.cmd_socket.connect("tcp://localhost:%s" % command_port)
        self.report_socket.connect("tcp://localhost:%s" % report_port)

        # Tracker to properly label saved models
        self._fitted_episodes = 0

        # The trainer, if selected as best model, saves its underlying model
        # to the stability_model_dir, in order for it to be loaded if the
        # stability arm is ever selected.
        self.stability_model_dir = stability_model_dir
        self.stability_model_history_dir = join(
            stability_model_dir, "stability-history"
        )
        os.makedirs(self.stability_model_history_dir, exist_ok=True)

    def sign_log(self, string: str) -> str:
        """Adds the signature of the trainer to the front of a log message."""
        return f"[{self.index}]: {string}"

    def get_command(self) -> Optional[str]:
        """Read any waiting command coming from the main process."""
        topic, msg = self.cmd_socket.recv().decode("utf-8").split(" ")
        if topic != self.index:
            return None
        print(self.sign_log(f"Received command: {msg}"))
        return msg

    def _run_loop(self, evaluation: bool) -> None:
        cost, cycles, trajectory = self.run_loop(evaluation=evaluation)
        if self.epsilon_scheduler is not None:
            self.control.epsilon = self.epsilon_scheduler.update()
            print(
                self.sign_log(
                    f"New epsilon on arm {self.index}: {self.control.epsilon}"
                )
            )
        print(self.sign_log("Done running."))
        topic = f"{self.index}-report"
        print(self.sign_log(f"Sending report on topic {topic}"))
        self.report_socket.send_multipart(
            [
                topic.encode(),
                str(cost).encode(),
                str(cycles).encode(),
                pickle.dumps(trajectory),
            ]
        )

    def update_epsilon(self, epsilon: float) -> None:
        """Update epsilon using a global value provided by the main process."""
        self.control.epsilon = epsilon
        self.send_ready()

    def send_ready(self) -> None:
        """Send a ready notice to the main thread."""
        print(self.sign_log("Sending ready notice..."))
        # Send index back as a signal to continue
        self.report_socket.send_string(f"ready {self.index[3:]}")

    def fit_on_batch(self, episode: Optional[int] = None) -> None:
        """Fit on the batch, and then send a ready message back to the main thread."""
        super().fit_on_batch(self._fitted_episodes)
        self.send_ready()
        self._fitted_episodes += 1

    def dump_model(self) -> None:
        """Saves the model so that it can be loaded by the main process."""
        print(self.sign_log(f"Saving model as current stability model."))
        self.control.save(
            join(self.stability_model_dir, f"stability-model-{self.index}")
        )
        # Save again in the history of stability models, for later analysis
        index = len(os.listdir(self.stability_model_history_dir))
        t = datetime.now().strftime("%y%m%d-%H%M%S")
        self.control.save(
            join(
                self.stability_model_history_dir,
                f"stability-model-{self.index}-{index}-{t}",
            )
        )
        self.send_ready()

    def run(self) -> None:
        """Await for commands and execute them when arrived.

        To end the process, send a 'finish' command.
        """
        while True:
            msg = self.get_command()
            if not msg:  # checks for None and ""
                pass
            elif msg == "finish":
                print(self.sign_log("Received command to finish."))
                break
            elif msg == "run" or msg == "run-eval":
                evaluation = msg[-4:] == "eval"
                print(self.sign_log(f"Running loop...(eval: {evaluation})"))
                self._run_loop(evaluation)
            elif msg == "train":
                print(self.sign_log("Training..."))
                self.fit_on_batch()
                print(self.sign_log("Done training."))
            elif msg == "save":
                self.dump_model()
            elif msg[:3] == "eps":
                epsilon = float(f".{msg[3:]}")
                print(self.sign_log(f"Updating epsilon to {epsilon}"))
                self.update_epsilon(epsilon)
        print(self.sign_log("Shutting down..."))
