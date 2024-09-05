# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from psipy.rl.controllers.nfqca import NFQCA
from psipy.rl.plants.tests.mocks import MockSingleChannelAction, MockState


@pytest.fixture
def tmpzip(tmpdir):
    return str(tmpdir.join("zipfile.zip"))


def make_models(inputs):
    def make_actor(inputs):
        inp = tfkl.Input((inputs, 1), name="state_actor")
        net = tfkl.Reshape((inputs * 1,))(inp)
        # net = tfkl.Flatten()(inp)
        net = tfkl.Dense(3, activation="tanh")(net)
        net = tfkl.Dense(3, activation="tanh")(net)
        net = tfkl.Dense(1, activation="tanh")(net)
        return tf.keras.Model(inp, net, name="actor")

    def make_critic(inputs):
        inp = tfkl.Input((inputs, 1), name="state_critic")
        act = tfkl.Input((1,), name="act_in")
        net = tfkl.Reshape((inputs * 1,))(inp)
        # net = tfkl.Flatten()(inp)
        net = tfkl.Concatenate()([net, act])
        net = tfkl.Dense(3, activation="tanh")(net)
        net = tfkl.Dense(3, activation="tanh")(net)
        net = tfkl.Dense(1, activation="sigmoid")(net)
        return tf.keras.Model([inp, act], net, name="critic")

    return make_actor(inputs), make_critic(inputs)


@pytest.mark.usefixtures("tensorflow")
class TestNFQCA:
    @staticmethod
    @pytest.mark.slow
    def test_save_load(tmpzip):
        channels = MockState.channels()
        actor, critic = make_models(len(channels))
        nfqca1 = NFQCA(
            actor=actor,
            critic=critic,
            state_channels=channels,
            action=MockSingleChannelAction,
        )
        nfqca1.save(tmpzip)
        nfqca2 = NFQCA.load(tmpzip, custom_objects=[MockSingleChannelAction])
        assert nfqca1.action_type == nfqca2.action_type
        nfqca2 = NFQCA.load(tmpzip)
        assert nfqca1.action_type == nfqca2.action_type
        # Compare weights for each layer in each actor/critic combination
        for original, loaded in zip(
            (nfqca1._actor.get_weights(), nfqca1._critic.get_weights()),
            (nfqca2._actor.get_weights(), nfqca2._critic.get_weights()),
        ):
            for i in range(len(original)):
                assert np.array_equal(original[i], loaded[i])

        assert nfqca1.get_config() == nfqca2.get_config()
        assert "lookback" in nfqca2.get_config()
        assert "td3" in nfqca2.get_config()
        assert "drop_pvs" in nfqca2.get_config()
        assert "disable_terminals" in nfqca2.get_config()
        assert "optimizer" in nfqca2.get_config()

        # Fit on observations
        nfqca1.normalizer.fit(np.random.random((3, len(channels))))
        nfqca1.save(tmpzip)
        nfqca2 = NFQCA.load(tmpzip, custom_objects=[MockSingleChannelAction])

        # Test on observation stacks
        data = np.random.random((3, len(channels), 3))
        data1 = nfqca1.normalizer.transform(data)
        data2 = nfqca2.normalizer.transform(data)
        assert not np.allclose(data, data1)
        assert not np.allclose(data, data2)
        assert np.allclose(data1, data2)
