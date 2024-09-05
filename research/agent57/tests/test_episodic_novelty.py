import numpy as np
import pytest
from tensorflow.keras.activations import linear, softmax

from psipy.rl.control.agent57.cost_components.episodic_novelty import (
    EpisodicMemory,
    EpisodicNoveltyModule,
)
from psipy.rl.plant import Action, State


class FakeState(State):
    _channels = ("fake", "fake2", "fake3")


class FakeAction(Action):
    channels = ("act1",)
    legal_values = ((1, 2, 3),)


class FakeContinuousAction(Action):
    dtype = "continuous"
    channels = ("act_c",)
    legal_values = ((-10, 10),)


@pytest.fixture
def module():
    return EpisodicNoveltyModule(
        state_size=3, action=FakeAction, embedding_size=5, k_neighbors=2
    )


@pytest.fixture
def memory():
    return EpisodicMemory(n_neighbors=2)


@pytest.fixture
def neighbors():
    return [
        np.array([0.0, 0.0, 0.0])[None, ...],
        np.array([0.0, 0.5, 0.0])[None, ...],
        np.array([1.0, 1.0, 0.5])[None, ...],
    ]


class TestEpisodicNovelty:
    @staticmethod
    def test_can_fit_discrete(module):
        state = FakeState(np.array([1, 2, 3]))
        next_state = FakeState(np.array([4, 5, 6]))
        action = FakeAction(np.array([1]))
        assert module.siamese_train_model.loss == "categorical_crossentropy"
        assert module.siamese_train_model.layers[-1].activation == softmax
        module.fit_inverse_dynamics_model(
            state.as_array(), next_state.as_array(), action.as_array(),
        )
        assert True  # if it gets here, the fit worked

    @staticmethod
    def test_can_fit_continuous():
        module = EpisodicNoveltyModule(
            state_size=3, action=FakeContinuousAction, embedding_size=5, k_neighbors=2
        )
        state = FakeState(np.array([1, 2, 3]))
        next_state = FakeState(np.array([4, 5, 6]))
        action = FakeContinuousAction(np.array([5.5]))
        assert module.siamese_train_model.loss == "mse"
        assert module.siamese_train_model.layers[-1].activation == linear
        module.fit_inverse_dynamics_model(
            state.as_array(), next_state.as_array(), action.as_array(),
        )
        assert True  # if it gets here, the fit worked

    @staticmethod
    def test_can_predict(module):
        state = FakeState(np.array([1, 2, 3]))
        module.append_state(state.as_array())
        assert len(module.memory) != 0
        assert isinstance(module.memory[-1], np.ndarray)

    @staticmethod
    def test_episodic_reward_gets_smaller_with_more_states(module, neighbors):
        for neighbor in neighbors:
            module.append_state(neighbor)
        # Reward for new state is high
        new_state = np.array([1, 1, 1])
        new_reward = module.get_episodic_reward(new_state)
        # See the new state again, reward should be lower
        module.append_state(new_state)
        repeat_new_reward = module.get_episodic_reward(new_state)
        assert new_reward > repeat_new_reward

        # Add another state that was already seen (should be lower than the novel state)
        newer_state = np.array([0, 0, 0])
        module.append_state(newer_state)
        already_seen_reward = module.get_episodic_reward(new_state)
        assert new_reward > already_seen_reward

        # Add another novel state, should be higher than both the
        # before-seen state novelties
        newest_state = np.array([0, 0, 1])
        module.append_state(newest_state)
        newest_reward = module.get_episodic_reward(new_state)
        assert newest_reward > already_seen_reward and newest_reward > repeat_new_reward


class TestEpisodicMemory:
    @staticmethod
    def test_finds_proper_neighbors(memory, neighbors):
        """Replicates the example on the sklearn.neighbors.KNeighborsRegressor page."""
        # Also tests fitting
        for neighbor in neighbors:
            memory.append(neighbor)
        memory.fit()  # fits on all states in memory
        neighbors, distances = memory.get_nearest_neighbors_to(
            np.array([1, 1, 1])[None, ...]
        )
        np.testing.assert_array_equal(np.array([0.5]), distances[:, 0])
        np.testing.assert_array_equal(neighbors[0], np.array([1, 1, 0.5]))
