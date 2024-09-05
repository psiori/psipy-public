# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from psipy.rl.control.nfq import NFQ, ObservationStack, tanh2
from psipy.rl.plant.tests.mocks import MockDiscreteAction, MockState


@pytest.fixture
def tmpzip(tmpdir):
    return str(tmpdir.join("zipfile.zip"))


def equate_weights(w1, w2):
    return all([np.array_equal(i, j) for i, j in zip(w1, w2)])


def make_model(inputs):
    inp = tfkl.Input((inputs, 1), name="state")
    net = tfkl.Reshape((inputs * 1,))(inp)
    # net = tfkl.BatchNormalization()(net)
    net = tfkl.Dense(2, activation="tanh")(net)
    net = tfkl.Dense(2, activation="tanh")(net)
    net = tfkl.Dense(3, activation="sigmoid")(net)
    return tf.keras.Model(inp, net)


def test_tanh2():
    rand = np.random.random(100) * 2 - 1
    cost = tanh2(rand, C=0.1, mu=0.5)
    assert cost.max().round(3) == 0.1
    assert np.sum(cost <= 0.095) == np.sum(np.abs(rand) <= 0.5)
    rand = np.random.random(100) * 2 - 1
    cost = tanh2(rand, C=0.01, mu=0.1)
    assert cost.max().round(5) == 0.01
    assert np.sum(cost <= 0.0095) == np.sum(np.abs(rand) <= 0.1)


class TestObservationStack:
    @staticmethod
    def test_values():
        stack = ObservationStack((2, 2), lookback=2, dtype=np.uint8)
        assert len(stack) == 2
        assert stack.stack.dtype == np.uint8
        assert np.all(stack.stack == 0)
        stack.append(np.array([[1, 1], [1, 1]]))
        # We backfill everything with the first observation
        assert np.all(stack.stack[..., 1] == 1)
        stack.append(np.array([[2, 2], [2, 2]]))
        assert np.all(stack.stack[..., 0] == 1)
        assert np.all(stack.stack[..., 1] == 2)
        stack.append(np.array([[3, 3], [3, 3]]))
        assert np.all(stack.stack[..., 0] == 2)
        assert np.all(stack.stack[..., 1] == 3)
        assert len(stack) == 2, "Length changed by appending?"


@pytest.mark.usefixtures("tensorflow")
class TestNFQ:
    @staticmethod
    def test_save_load(tmpzip):
        channels = MockState.channels()
        model = make_model(len(channels))
        nfq1 = NFQ(
            model=model,
            state_channels=channels,
            action=MockDiscreteAction,
            action_channels=("channel2",),
            action_values=(1, 10),
        )
        nfq1.save(tmpzip)
        nfq2 = NFQ.load(tmpzip)
        assert nfq1.action_type == nfq2.action_type
        assert nfq1.action_values.tolist() == nfq2.action_values.tolist()

        # Fit on observations
        nfq1.normalizer.fit(np.random.random((10, len(channels))))
        nfq1.save(tmpzip)
        nfq2 = NFQ.load(tmpzip, custom_objects=[MockDiscreteAction])

        # Test on observation stacks
        data = np.random.random((10, len(channels), 3))
        data1 = nfq1.normalizer.transform(data)
        data2 = nfq2.normalizer.transform(data)
        assert not np.allclose(data, data1)
        assert not np.allclose(data, data2)
        assert np.allclose(data1, data2)

    @staticmethod
    def test_max_epsilon():
        channels = MockState.channels()
        model = make_model(len(channels))
        nfq = NFQ(
            model=model,
            state_channels=channels,
            action=MockDiscreteAction,
            action_channels=("channel2",),
            action_values=(1, 10),
        )
        nfq.epsilon = 1
        actions, meta = nfq.get_actions(np.random.random((10, len(channels), 1)))
        assert actions.shape == (10, 1)
        assert meta["nodoe"].shape == (10, 1)
        assert meta["index"].shape == (10, 1)

        state = MockState(np.random.random((len(channels),)))
        action = nfq.get_action(state)
        assert action.keys() == ("channel2",)
        assert set(action.keys(with_additional=True)) == {
            "channel2",
            "channel2_index",
            "channel2_nodoe",
        }
