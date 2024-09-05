import logging
from typing import Tuple

import numpy as np
import tensorflow.keras as tfk
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import layers as tfkl

from psipy.core.welfords import WelfordsAlgorithm
from psipy.rl.plant import Action

LOG = logging.getLogger(__name__)

# TODO Retrace/Vtrace loss


class EpisodicMemory:
    """Memory of controllable states visited during an episode.

    In order to determine how often a state was visited during an episode, without
    keeping explicit counts and to deal with continuous state spaces, a memory of
    states is kept in order to perform a k-nearest neighbor search to approximate
    a pseudo-count. This count is then used to determine some intrinsic novelty
    to provide as a dense reward to an agent. (NGU Eq. 2 Episodic memory...)

    The states used within the memory are called "controllable states", since they
    are the embeddings created from an inverse dynamics model. The idea is that
    the embedding removes all parts of the state that are unaffected by the action.
    See "Embedding network:..." in the NGU paper for more details or see the
    :class:`EpisodicNoveltyModule`.

    The memory is cleared at the beginning of each episode in order to provide an
    "intraepisodic" intrinsic reward.
    """

    def __init__(self, n_neighbors: int):
        self.controllable_states = []
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)

    def __getitem__(self, item):
        return self.controllable_states[item]

    def __len__(self):
        return len(self.controllable_states)

    def fit(self):
        """Update the distance matrix of the KNN space."""
        self.knn.fit(self.controllable_states)

    def clear(self) -> None:
        """Reset the memory by clearing the states and resetting the KNN space."""
        self.controllable_states.clear()
        self.knn = NearestNeighbors(n_neighbors=self.knn.n_neighbors)

    def append(self, controllable_state: np.ndarray) -> None:
        """Append a single state to the memory."""
        assert len(controllable_state) == 1, "Can only append a single state!"
        # The .ravel() drops the singular batch dim
        self.controllable_states.append(controllable_state.ravel())
        # The KNN is fit here since it needs to be fit before predict;
        # To not have the current state included in the neighbors, an extra neighbor
        # must be requested and the first sliced off, because the first neighbor
        # will be the requested state and have a distance 0.
        self.fit()

    def get_nearest_neighbors_to(
        self, controllable_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the nearest neighbors to the given state along with their distances."""
        # Plus one because the identity state will be sliced off and n_neighbors is
        # still wanted after the slicing.
        n_neighbors = self.knn.n_neighbors + 1
        if len(self) == 1:
            # If there is only one state, return it with 0 distance
            return controllable_state, np.zeros(1)
        if n_neighbors > len(self):
            n_neighbors = len(self)
        distances, indices = self.knn.kneighbors(
            controllable_state, n_neighbors=n_neighbors
        )
        neighbors = np.array(self.controllable_states)[indices].squeeze()
        # Slice off the input state that has distance 0
        return neighbors[1:], distances.T[1:]  # (1,x) -> (x,1)

    def notify_episode_starts(self):
        """Reset to start anew every episode."""
        self.clear()
        LOG.info("Episodic memory cleared")


class EpisodicNoveltyModule:
    """Creates an intraepisodic intrinsic reward based on states seen in an episode.

    The module uses an inverse dynamics model (input state and next state and generate
    the action that joins them) in order to create an embedding of "controllable states".
    These states have theoretically removed all elements from them that are not influenced by the actions
    taken by the agent. In order to compute novelty, nearest neighbors of any given state are put through a kernel
    in order to determine "pseudo-counts" of how "known" the state is.

    Since the memory is cleared at the beginning of every episode, this module provides
    only intraepisodic novelty values. Interepisodic novelty is handled through another
    module (Random Network Distillation in the case of NGU).

    Args:
        state_size: The number of state channels
        action: The action type
        embedding_size: How big the controllable state embedding is. The embedding -> action layer is 4 times this size.
        k_neighbors: How many neighbors to consider for state similarity
        c: A small constant that guarantees a minimum amount of pseudo-counts. Default comes from below Sec. 2 Eq. 2.
        kernel_cluster_distance: Appendix F.2
        kernal_max_similarity: Appendix F.2
    """

    def __init__(
        self,
        state_size: int,
        action: Action,
        embedding_size: int,
        k_neighbors: int = 10,
        c: float = 0.001,
        kernel_cluster_distance: float = 0.008,
        kernel_max_similarity: float = 8,
        embedding_epochs: int = 10,
        embedding_batch_size: int = 1024,
        callbacks=None,
    ):
        self._pseudo_count_constant = 0.001
        self.max_similarity = 8
        self.__epsilon = 0.0001
        self.action_type = action
        # In the discrete action case, the indices need to be recreated, hence the below mapping.
        self._values_mapping = {
            k: v
            for k, v in zip(action.legal_values[0], range(len(action.legal_values[0])))
        }
        self.c = c
        self.kernel_cluster_distance = kernel_cluster_distance
        self.kernel_max_similarity = kernel_max_similarity

        # Embedding network
        models = self._create_siamese_network(state_size, embedding_size)
        self._epochs = embedding_epochs
        self._batchsize = embedding_batch_size
        self._siamese_train_model, self._controllable_state_model = models

        # Memory
        self.memory = EpisodicMemory(k_neighbors)
        self.distance_average = WelfordsAlgorithm()

        self.callbacks = []
        if callbacks is not None:
            self.callbacks = [callbacks]

    def _create_siamese_network(
        self, state_size, embedding_size
    ) -> Tuple[tfk.Model, tfk.Model]:
        state = tfkl.Input((state_size,), name="siam/state")
        state_net = tfkl.Flatten()(state)
        state_net = tfkl.Dense(20, activation="relu")(state_net)
        # This is the embedding
        state_net = tfkl.Dense(embedding_size, activation="relu")(state_net)

        next_state = tfkl.Input((state_size,), name="siam/next_state")
        next_net = tfkl.Flatten()(next_state)
        next_net = tfkl.Dense(20, activation="relu")(next_net)
        next_net = tfkl.Dense(embedding_size, activation="relu")(next_net)

        net = tfkl.Concatenate()([state_net, next_net])
        net = tfkl.Dense(embedding_size * 4, activation="relu")(net)
        # If the action is discrete, solve a categorical cross entropy loss;
        # if continuous, solve a mean squared error on the action value
        if self.action_type.dtype == "discrete":
            legal_values = len(self.action_type.legal_values[0])
            net = tfkl.Dense(legal_values, activation="softmax")(net)
            loss = "categorical_crossentropy"
        else:
            net = tfkl.Dense(1, activation="linear")(net)
            loss = "mse"

        # Action prediction model p(a|f(s),f(s'))
        train_model = tfk.Model([state, next_state], net)
        train_model.compile(optimizer="adam", loss=loss)

        # Controllable state model
        controllable_state = tfk.Model([state], state_net)
        return train_model, controllable_state

    def __maybe_add_batch_dim(self, array: np.ndarray) -> np.ndarray:
        """Expand dims from (x,) to (1, x) for network computation."""
        # Only handles vector and scalar arrays (matrices and up not supported)
        if len(array.shape) == 1:
            array = array[None, ...]
        return array

    def fit_inverse_dynamics_model(
        self, states: np.ndarray, next_states: np.ndarray, actions: np.ndarray
    ) -> bool:
        """Fit the inverse dynamics model."""
        states = self.__maybe_add_batch_dim(states)
        next_states = self.__maybe_add_batch_dim(next_states)
        actions = self.__maybe_add_batch_dim(actions)
        if self.action_type.dtype == "discrete":
            actions = [self._values_mapping[a] for a in actions[:, 0]]
            # Convert to 1-hot
            one_hot_array = np.zeros(
                (len(actions), len(self.action_type.legal_values[0]))
            )
            one_hot_array[np.arange(len(actions)), actions] = 1
            actions = one_hot_array
        self.siamese_train_model.fit(
            [states, next_states],
            actions,
            epochs=self._epochs,
            batch_size=self._batchsize,
            verbose=0,
            callbacks=[self.callbacks],
        )
        LOG.info("Inverse dynamics fitted.")
        return True

    def append_state(self, state: np.ndarray) -> None:
        """Transform state into controllable state and append to memory."""
        state = self.__maybe_add_batch_dim(state)  # needed for prediction
        controllable_state = self.controllable_state_model.predict(state)
        self.memory.append(controllable_state)

    def get_episodic_reward(self, state: np.ndarray) -> float:
        """Calculate the intraepisodic intrinsic reward on a single state.

        See Algorithm 1 in Appendix A for more information.

        Args:
            state: The state to calculate the episodic reward on
        """
        state = self.__maybe_add_batch_dim(state)
        controllable_state = self.controllable_state_model.predict(state)
        neighbors, distances = self.memory.get_nearest_neighbors_to(controllable_state)
        # Compute the squared euclidean distance as per Alg. 1
        distances **= 2
        self.distance_average.update(distances)  # ** 2)
        distances = distances / (self.distance_average.mean + self.__epsilon)
        distances = np.maximum(
            (distances - self.kernel_cluster_distance).ravel(), np.zeros(len(distances))
        )
        kernel = self.__epsilon / (distances + self.__epsilon)
        similarity = np.sqrt(np.sum(kernel)) + self.c
        if similarity > self.kernel_max_similarity:
            return 0
        return 1 / similarity  # TODO: Ensure this works for a batch of states?

    def notify_episode_starts(self):
        """Reset memory at episode start but keep state model."""
        self.memory.notify_episode_starts()

    @property
    def siamese_train_model(self):
        """The inverse dynamics model."""
        return self._siamese_train_model

    @property
    def controllable_state_model(self):
        """The controllable state embedding."""
        return self._controllable_state_model
