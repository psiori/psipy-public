import numpy as np

from psipy.rl.control.agent57.cost_components.random_network_distillation import (
    RandomNetworkDistillation,
)
from psipy.rl.plant import State


class FakeState(State):
    _channels = ("fake", "fake2", "fake3")


class TestRandomNetworkDistillation:
    @staticmethod
    def test_can_predict():
        rnd = RandomNetworkDistillation(3, (2, 20))
        state = FakeState(np.array([1, 2, 3]))
        rnd.get_novelty(state.as_array())
        assert True  # if it gets here, the model doesn't throw an error

    @staticmethod
    def test_target_weights_dont_change():
        # Also tests fitting
        rnd = RandomNetworkDistillation(3, (2, 20))
        state = FakeState(np.array([1, 2, 3]))
        state = state.as_array()
        original_weights = rnd.random_target_network.get_weights().copy()
        rnd.fit(state)
        np.testing.assert_equal(
            rnd.random_target_network.get_weights(), original_weights
        )
