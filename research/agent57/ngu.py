"""


```
@article{badia2020never,
  title={Never Give Up: Learning Directed Exploration Strategies},
  author={Badia, Adri{\`a} Puigdom{\`e}nech and Sprechmann, Pablo and Vitvitskyi, Alex and Guo, Daniel and Piot, Bilal and Kapturowski, Steven and Tieleman, Olivier and Arjovsky, Mart{\'\i}n and Pritzel, Alexander and Bolt, Andew and others},
  journal={arXiv preprint arXiv:2002.06038},
  year={2020}
}
```
"""
import logging
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid
from tensorflow.keras import layers as tfkl

from psipy.rl.control import Controller
from psipy.rl.control.agent57.cost_components.episodic_novelty import (
    EpisodicNoveltyModule,
)
from psipy.rl.control.agent57.cost_components.random_network_distillation import (
    RandomNetworkDistillation,
)
from psipy.rl.control.nfq import ObservationStack, enrich_errors
from psipy.rl.control.nfq import NFQ  # TODO: Make rewards work with NFQ
from psipy.rl.plant import State
from psipy.rl.plant.plant import Action, Numeric, Plant
from psipy.rl.preprocessing import StackNormalizer

LOG = logging.getLogger(__name__)

# TODO Tests

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=3)
plottables = defaultdict(list)


class NGUPlantMixin(Plant):
    """Plant Mixin which adds NGU related parameters to the state and maintains novelty modules.

    NGU utilizes extra information extracted from the plant, such as intrinsic
    reward, which this mixin provides. #TODO: Better docstring

    There should be one plant per NGU mixture.
    """

    def __init__(
        self,
        beta: float,
        rnd_epochs: int,
        embedding_size: int,
        k_neighbors: int,
        embedding_epochs: int,
        embedding_batch_size: int,
        callbacks: List,
        max_reward_scale: int = 5,  # parameter L, sec. 2 eq. 1
    ):
        # Does not call super since no parameters should be overwritten
        assert self.state_type.__name__.startswith(
            "NGU"
        ), "NGUPlant must be used with an NGUState!"
        self.beta = beta
        self.lifelong_novelty = RandomNetworkDistillation(  # TODO: Do RND and Episodic get the extra state params?
            len(
                self.state_type.channels()
            ),  # TODO: Actually the betas can't train on other mixture's data because the intrinsic reward is 0ed out, no?
            (2, 100),
            epochs=rnd_epochs,
            ngu_novelty=False,  # see comment in RND regarding this parameter
            callback=callbacks[1],  # TODO: Make this 0, 1 on the callbacks not 1, 0
        )
        self.episodic_novelty = EpisodicNoveltyModule(
            len(self.state_type.channels()),
            self.action_type,
            embedding_size=embedding_size,
            k_neighbors=k_neighbors,
            embedding_epochs=embedding_epochs,
            embedding_batch_size=embedding_batch_size,
            callbacks=callbacks[0],
        )

        self.maximum_reward_scale = max_reward_scale

    def get_next_state(self, state, action):
        next_state = super().get_next_state(state, action)
        state = state.as_array(only_channels=True)
        self.episodic_novelty.append_state(state)
        prev_action = action.as_array()[0]
        extrinsic_reward = next_state.cost
        intrinsic_reward = self.get_intrinsic_reward(state)[0]
        cost = self.get_reward(extrinsic_reward, intrinsic_reward)
        # Full extrinsic + intrinsic reward #TODO: This won't be used during training if cost function is provided! Overwritten!
        next_state.set_ngu_values(
            prev_action=prev_action,
            extrinsic_reward=extrinsic_reward,
            intrinsic_reward=intrinsic_reward,
            beta=self.beta,
        )
        next_state.cost = cost
        self._current_state = next_state
        return self._current_state

    def get_episodic_novelty(self, state):
        return self.episodic_novelty.get_episodic_reward(state)

    def get_lifelong_novelty(self, state):
        return self.lifelong_novelty.get_novelty(state)

    def get_intrinsic_reward(self, state):
        """Returns a mixture of both intrinsic rewards.

        See NGU Sec. 2 Equation 1.
        """
        lifelong_factor = self.get_lifelong_novelty(state)
        episodic_factor = self.get_episodic_novelty(state)
        itr = episodic_factor * np.minimum(
            np.maximum(lifelong_factor, np.ones(len(lifelong_factor))),
            self.maximum_reward_scale,
        )
        plt.figure(fig.number)
        for a in ax:
            a.clear()
        plottables["lf"].append(lifelong_factor)
        plottables["ef"].append(episodic_factor)
        plottables["ir"].append(itr)
        ax[0].plot(plottables["lf"], label="Lifelong")
        ax[1].plot(plottables["ef"], label="Episodic")
        ax[2].plot(plottables["ir"], label="Intrinsic Total")
        ax[0].set_title("Lifelong")
        ax[1].set_title("Episodic")
        ax[2].set_title("Intrinsic Total")
        plt.pause(0.01)
        return itr

    def get_reward(self, extrinsic_reward, intrinsic_reward):
        """Returns a mixture of the extrinsic and intrinsic reward.

        The intrinsic weight scales the importance of the intrinsic rewards on the
        controller. An intrinsic weight of 0 disables intrinsic rewards, and yields
        a controller that, when acting greedily, purely acts based on the task defined
        by the reward function.

        See NGU Sec. 2.
        """
        return extrinsic_reward + self.beta * intrinsic_reward

    def notify_episode_starts(self) -> bool:
        """Reset the episodic novelty module at the beginning of every episode."""
        super().notify_episode_starts()
        self.episodic_novelty.notify_episode_starts()
        # for key in plottables:
        #     plottables[key].clear()
        return True

    def notify_episode_stops(self) -> bool:
        super().notify_episode_stops()
        LOG.warning("MAKE SURE TO CHANGE BETA!")
        return True


class NGUState(State):
    """Observation type specific to NGU-based control.

    NGU requires the rewards, previous action, and a one hot encoding of the
    beta mixture of the current controller to be a part of the state. Thus, this
    state requires those values and will provide them as part of the state when
    being saved to SART or converted to an array.

    Args:
        prev_action: The action taken to get to this state
        extrinsic_reward: The reward determined by the task to be solved
        intrinsic_reward: The reward determined by the novelty of the state
        beta: The integer representation of which NGU mixture this state was generated by
        See the docstring for :class:`State` for information about the other parameters
    """

    __slots__ = [
        "_data",
        "prev_action",
        "extrinsic_reward",
        "intrinsic_reward",
        "beta",
        "terminal",
        "meta",
    ]

    prev_action: Union[float, int]

    #: Immediate, at runtime plant-prescribed reward at the current timestep.
    extrinsic_reward: float

    #: Immediate, at runtime controller-prescribed intrinsic reward at current timestep
    intrinsic_reward: float

    #: Integer representation of the beta of the controller that generated this state.
    beta: int  # TODO To be converted controller side into one hot

    def __init__(
        self,
        values: Union[Dict[str, Union[np.ndarray, Numeric]], np.ndarray],
        cost: float = 0.0,
        terminal: bool = False,
        meta: Optional[Dict[str, Optional[Union[Numeric, str]]]] = None,
        check_unexpectedness: bool = False,
        filler: Optional[Dict[str, Numeric]] = None,
        *,
        prev_action: Union[int, float] = 0,
        extrinsic_reward: float = 0,
        intrinsic_reward: float = 0,
        beta: int = 0,
    ) -> None:
        self.prev_action = prev_action
        self.extrinsic_reward = extrinsic_reward
        self.intrinsic_reward = intrinsic_reward
        self.beta = beta
        self.cost = cost
        super().__init__(values, cost, terminal, meta, check_unexpectedness, filler)

    def as_dict(self, semantic: bool = False):
        if semantic:
            raise NotImplementedError
        values = OrderedDict(
            (channel, self[channel]) for i, channel in enumerate(self.keys())
        )
        values["prev_action"] = self.prev_action
        values["extrinsic_reward"] = self.extrinsic_reward
        values["intrinsic_reward"] = self.intrinsic_reward
        values["beta"] = self.beta
        return OrderedDict(
            values=values,
            cost=self.extrinsic_reward,
            terminal=self.terminal,
            meta=self.meta,
        )

    def as_array(
        self, *channels: Union[Sequence[str], str, None], only_channels: bool = False
    ) -> np.ndarray:
        if len(channels) == 0 or channels[0] is None:
            channels = self.keys()
        if only_channels:
            return np.array([*[self[channel] for channel in channels]])
        return np.array(
            [
                *[self[channel] for channel in channels],
                self.prev_action,
                self.extrinsic_reward,
                self.intrinsic_reward,
                self.beta,
            ]
        )

    def set_ngu_values(self, *, prev_action, extrinsic_reward, intrinsic_reward, beta):
        """Set the NGU values, assuring proper variable assignment regardless of input order."""
        self.prev_action = prev_action
        self.extrinsic_reward = extrinsic_reward
        self.intrinsic_reward = intrinsic_reward
        self.beta = beta


class NGU(Controller):
    def __init__(
        self,
        num_mixtures: int,
        state_channels,
        action,
        network_structure: Tuple[int, int],
        action_values: Optional[Tuple[Union[int, float], ...]] = None,
        lookback: int = 1,
        control_pairs: Optional[Tuple[Tuple[str, str], ...]] = None,
        num_repeat: int = 0,
        doubleq: bool = False,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "RMSProp",
        prioritized: bool = False,
        base_beta: float = 0.5,
        max_gamma: float = 0.997,
        min_gamma: float = 0.99,
        action_channels=None,
        **kwargs,
    ):
        super().__init__(state_channels, action, action_channels, **kwargs)

        # TODO INPUT one hot beta, prev action, prev intrinsic, prev extrinisc

        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        # Beta and gamma are chosen such that smaller discount factors are used for the
        # exploratory policies since the intrinsic reward is dense and the range of
        # values is small, and use the highest possible discount factor for the
        # exploitative policy in order to be as close as possible to optimizing
        # the undiscounted return (Appendix A "Distributed Training").
        self.betas = self._generate_betas(num_mixtures, base_beta)
        self.gammas = self._generate_gammas(num_mixtures)
        assert len(self.betas) == len(self.gammas)

        # Add more channels to the state due to extra state parameters included by NGU
        self._len_state = len(self.state_channels) + 4
        self._len_processed_state = len(self.state_channels) + 3 + num_mixtures
        self.controllers = []
        for i in range(num_mixtures):
            # Add 4 to the state channels since both rewards,
            # prev action, and beta enter the model
            model = self._create_model(
                self._len_processed_state,
                len(action.legal_values[0]),
                lookback,
                network_structure,
            )
            control = NFQ(
                model,
                action_values,  # TODO: Are the beta=0 weights put into each beta>0 occasionally? or is the whole idea just collect data for the beta=0 which then learns the optimal policy?
                lookback,
                None,
                num_repeat,
                doubleq,
                optimizer,
                prioritized,
                state_channels=state_channels,
                action=action,
            )
            control.normalizer = StackNormalizer("identity")
            control.fit_normalizer(
                np.zeros(1)[None, ...]
            )  # Init fit the identity normalizer to prevent warning log
            control.epsilon = 0  # TODO: This might not be correct
            self.controllers.append(control)
        self._mixture_index = 0

        self.control_pairs = control_pairs
        self.normalizer = StackNormalizer("meanstd")

        # Memory for defining states over multiple timesteps, not for generating reward
        self._memory = ObservationStack((self._len_state,), lookback=lookback)

        # Appendix A: Distributed Training (GAMMA VALUES)
        # .5 from NGU Appendix F.3 Table 7 but they had best results with .3

    def _create_model(self, n_inputs, n_outputs, lookback, network_structure):
        # TODO NOTE: Only NFQr is currently supported!
        inp = tfkl.Input((n_inputs, lookback), name="states")
        net = tfkl.Flatten()(inp)
        for layer in range(network_structure[0]):
            net = tfkl.Dense(network_structure[1], activation="tanh")(net)
        net = tfkl.Dense(n_outputs, activation="sigmoid")(net)
        return tf.keras.Model(inp, net)

    def _generate_gammas(self, num_mixtures: int) -> List[float]:
        """Create n gammas, one for each NGU mixture.

        Equation from NGU Appendix A: Distributed Training.
        """
        gammas = []
        for i in range(num_mixtures):
            gamma = 1 - np.exp(
                (num_mixtures - 1 - i) * np.log(1 - self.max_gamma)
                + (i * np.log(1 - self.min_gamma)) / num_mixtures
                - 1
            )
            gammas.append(gamma)
        return gammas

    def _generate_betas(self, num_mixtures: int, base_beta: float = 0.5) -> List[float]:
        """Create n betas, one for each NGU mixture.

        Equation from NGU Appendix A: Distributed Training.
        """
        betas = []
        betas.append(0)
        for i in range(1, num_mixtures - 1):
            betas.append(
                base_beta * sigmoid(10 * 2 * i - (num_mixtures - 2) / num_mixtures - 2)
            )
        betas.append(base_beta)
        return betas

    def fit_normalizer(self, observations, method) -> None:
        """Fit the internal StackNormalizer on all channels except beta.

        Note that all controllers will have the same normalization parameters. This
        means that the normalizer might have seen data that some controller did not
        see. This is assumed to be ok because the normalizer will be closer to the
        true distribution the more data it sees.
        """
        if method and method != self.normalizer.method:
            self.normalizer = StackNormalizer(method)
        # Fit on everything except beta
        self.normalizer.fit(observations[:, :-1])

    def fit(self, *args, **kwargs) -> None:
        """Fits the currently active mixture."""
        self.active_controller.fit(*args, **kwargs)

    def get_active_mixture(self,) -> Tuple[float, float]:
        return self.active_gamma, self.active_beta

    def get_action(self, state: State) -> Action:
        observation = state.as_array(*self.state_channels)
        stack = self._memory.append(observation).stack
        stack = self.preprocess_observations(stack)
        action, meta = self.active_controller.get_actions(stack)
        assert action.shape[0] == 1
        action = action.ravel()

        # Splitting the meta data vectors into the individual action channels.
        individual_meta = dict()
        for key, values in meta.items():
            for channel, value in zip(self.action_channels, values):
                individual_meta[f"{channel}_{key}"] = value.item()

        mapping = dict(zip(self.action_channels, action))
        return self.action_type(mapping, additional_data=individual_meta)

    def _initial_preprocessing(self, stacks: np.ndarray):
        channels = self.state_channels
        if self.control_pairs is not None:
            stacks, channels = enrich_errors(stacks, channels, *self.control_pairs)
        # TODO This is not he same input channels in nfqr and so might break if using it
        self.input_channels = channels
        return stacks

    def preprocess_observations(self, stacks: np.ndarray):
        """One hot encode beta."""
        if len(stacks.shape) == 2:
            stacks = stacks[None, ...]  # Add batch dim
        stacks = self._initial_preprocessing(stacks)
        # Transform everything except beta (since it is preprocessed differently)
        stacks[:, :-1] = self.normalizer.transform(stacks[:, :-1])
        processed_stacks = np.zeros(
            (stacks.shape[0], stacks.shape[1] + len(self.betas) - 1, stacks.shape[2])
        )
        # Fill in all data except for the beta parameter
        processed_stacks[:, :-1] = stacks
        # Extra the recorded betas
        beta = stacks[:, -1]
        # Allocate memory for the one hot encoding
        one_hot = np.zeros((len(beta), len(self.betas)))
        # Create a mapping from betas to ints
        mapping = {k: v for k, v in zip(self.betas, range(len(self.betas)))}
        # Create indices of which column should become a one in the encoding
        indices = np.array([mapping[b] for b in beta.ravel()])
        # Set values and insert into array
        one_hot[np.arange(len(one_hot)), indices] = 1
        processed_stacks[..., -len(self.betas) :, :] = one_hot[..., None]
        return processed_stacks

    def next_mixture(self):
        """Switch the active model mixture to the next."""
        self._mixture_index = (self._mixture_index + 1) % len(self.controllers)
        LOG.info(f"Current NGU mixture: beta={self.active_beta}")

    def notify_episode_starts(self) -> None:
        pass

    def notify_episode_stops(self) -> None:
        self.next_mixture()
        self._memory.clear()

    @property
    def active_controller(self) -> Controller:
        return self.controllers[self._mixture_index]

    @property
    def active_beta(self) -> float:
        return self.betas[self._mixture_index]

    @property
    def active_gamma(self) -> float:
        return self.gammas[self._mixture_index]


if __name__ == "__main__":

    class FakeState(NGUState):
        _channels = ("test", "test2")

    s = FakeState([1, 2], 1, 2, 3, 4)
    print(s.keys())
    print(s.as_dict())
    print(s.as_array())
