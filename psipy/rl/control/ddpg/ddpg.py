from typing import Tuple

import numpy as np
import tensorflow as tf

from psipy.rl.control.controller import Controller


class Actor:
    def __init__(self, input_shape: Tuple[int, ...]):
        self._input_shape = input_shape

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = self.make_model(self.input_shape, name="actor")
        return self._model

    @property
    def target(self):
        if not hasattr(self, "_target"):
            self._target = self.make_model(self.input_shape, name="target_actor")
        return self._target

    @staticmethod
    def make_model(self, input_shape: Tuple[int, ...], name: str):
        inp = tf.keras.Input(input_shape=input_shape)
        net = tf.keras.layers.Dense(40, activation="tanh")(inp)
        return tf.keras.Model(inputs=inp, outputs=net, name=name)


class Critic:
    def __init__(self, name: str, input_shape: Tuple[int, ...]):
        self._name = name
        self._input_shape = input_shape
        self._layers = Critic.make_layers()

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = self.make_model(self._input_shape, self._name)
        return self._model

    @staticmethod
    def make_layers():
        return [
            tf.keras.layers.Dense(40, activation="tanh"),
            tf.keras.layers.Dense(40, activation="tanh"),
        ]

    def __call__(self, act: tf.keras.layers.Layer, name="critic"):
        inp = tf.keras.Input(input_shape=self._input_shape)
        net = tf.layers.concat([inp, act])
        for layer in self._layers:
            net = layer(net)
        return tf.keras.Model(inputs=[inp, act], outputs=net, name=name)


class DDPG(Controller):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def _get_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def notify_episode_starts(self):
        ...

    def notify_episode_stops(self):
        ...
