"""Random Network Distillation.

This module calculates a measure of novelty per state which decreases the more
the state is visited. It consists of two networks, one which is trained on the
outputs of the other. The novelty is then calculated as the MSE between the
two network outputs, turning the predictive error novelty into a deterministic
problem.

The target generation network is never trained and is initialized randomly.
This randomness is claimed to prevent running into the issues of getting stuck
on stochastic states or high computational effort.

The paper refers to the output of the network as both novelty and intrinsic reward.
Both terms refer to the same thing.

Although the term `reward` is used, the notion of reward or cost has no effect on
the RND. It only returns the normalized MSE between the two networks, and it is up
to the user to use it in either a reward or cost context.

See the class docstring for more information.
```
@article{burda2018exploration,
  title={Exploration by random network distillation},
  author={Burda, Yuri and Edwards, Harrison and Storkey, Amos and Klimov, Oleg},
  journal={arXiv preprint arXiv:1810.12894},
  year={2018}
}
```
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow.keras as tfk
from tensorflow.keras import layers as tfkl

from psipy.core.welfords import WelfordsAlgorithm

LOG = logging.getLogger(__name__)

# TODO: Implement updated prio rule from Kubowski or whoever

# TODO: Do I send in the batch normalized data or just the batch raw data
#  and use only this normalization? Question is, do I do both normalizations, or does it even matter?


class RandomNetworkDistillation:
    """Random Network Distillation Network for life long intrinsic rewards.

    Provides a measure of novelty of each state that decreases the more the state is visited.
    This network can be used to provide an exploration bonus/intrinsic reward to a controller
    interacting with its environment to better explore the state space.

    The RND takes in the states from a batch and computes the MSE between
    the random frozen target network and the trained predictor network. Training
    attempts to bring the predictor closer to the random network. With more states
    seen, it predicts closer to the target, and thus "novelty" decreases.

    Running stats are used to normalize the states going into the networks, as
    well as the reward values coming out. Separate parameters are used for each. In the
    original RND paper, the state normalization is warm started by running a random
    agent in the environment for a few steps. However, this is not required for this
    implementation because it can fit its normalizer on all data contained in
    the :class:`Batch`.

    This network generates the "life long novelty" part of the NGU "intrinsic reward".
    The output of this network eventually modulates the intraepisodic intrinsic reward
    after being put through a (1, L) clipping (NGU eq. 2.1).

    Original Paper:
        https://arxiv.org/abs/1810.12894

    Args:
        len_state: The length of the state, i.e. how many channels
        network_structure: A tuple consisting of (num_layers, num_neurons) that will
                           be the structure of both the prediction and target networks
        epochs: How many epochs to train the RND network per fit
        ngu_novelty: Whether or not to calculate novelty according to the Never Give Up
                     agent (Badia et. al., 2020).
    """

    def __init__(
        self,
        len_state: int,
        network_structure: Tuple[int, int],
        epochs=10,
        ngu_novelty: bool = False,
        callback=None,
    ):
        self.n_layers, self.n_neurons = network_structure
        self.epochs = epochs
        self._len_state = len_state
        self._ngu = ngu_novelty

        self._random_target_network = self._create_network(len_state)
        self._predictor = self._create_network(len_state)

        # Running stats trackers
        self.state_stats = WelfordsAlgorithm()
        self.reward_stats = WelfordsAlgorithm()

        self.callbacks = []
        if callback is not None:
            self.callbacks = [callback]

    def _create_network(self, len_state: int) -> tfk.Model:
        """Create equal architecture networks.

        The predictor and target networks are the same, except that the target
        network is never trained. Each output n_neurons values.
        """
        state = tfkl.Input((len_state,), name="rnd/state_in")
        net = tfkl.Dense(self.n_neurons)(state)
        for _ in range(self.n_layers - 1):
            net = tfkl.Dense(self.n_neurons)(net)
        model = tfk.Model(state, net)
        model.compile(
            optimizer="adam", loss="mse", metrics=[tfk.metrics.MeanSquaredError()]
        )
        return model

    def update_state_stats(self, states: np.ndarray) -> None:
        """Update state running average and stddev according to Welford's Algorithm.

        See RND paper Section 2.4.
        """
        self.state_stats.update(states)

    def update_reward_stats(self, reward: np.ndarray) -> None:
        """Update the running standard deviation of rewards (sec. 2.4)."""
        self.reward_stats.update(reward)

    def whiten_state(self, states: np.ndarray) -> np.ndarray:
        """Whitens and clips states for prediction (sec. 2.4)."""
        # Input shapes are either (1, state_channels) or (batch, state_channels, 1);
        # the last dimension is removed if equal to 1
        if len(states.shape) == 3 and states.shape[-1] == 1:
            states = states.squeeze()
        # TODO: Replace this with a stack normalizer and constantly update mean and scale?
        normalized = (states - self.state_stats.mean) / (
            # Prevent division by 0 by adding a small epsilon to the std
            self.state_stats.std
            + 1e-8
        )
        assert normalized.shape == states.shape
        return np.clip(normalized, -5, 5)  # 5 from RND paper

    def fit(self, states: np.ndarray) -> None:
        """Fit the predictor on the random network's target (sec. 2.2)."""
        if len(states.shape) == 1:
            states = states[None, ...]
        states = self.whiten_state(states)
        target = self.random_target_network.predict(states)
        # TODO: Callbacks?
        self.predictor.fit(
            states,
            target,
            epochs=self.epochs,
            verbose=0,
            batch_size=len(states),
            callbacks=[self.callbacks],
        )
        LOG.info("RND fitted.")

    def get_novelty(self, states: np.ndarray) -> np.ndarray:
        """Calculate the intrinsic reward for the batch of states (sec. 2.2).

        The intrinsic reward is the error between the predictor network and the
        output of the random target network. The more a state is seen, the closer
        the output of the predictor to the target network will be, and thus a
        lower error. Lower errors correspond to less novelty, and so more novel
        (read: higher MSE) states will receive more intrinsic reward.

        The RND paper divides the returned MSE by its running standard deviation.
        NGU properly whitens the MSE.
        """
        if len(states.shape) == 1:
            states = states[None, ...]
        # Note: if RND is ever used on multiple values in a batch, the state stats will be
        # messed up; to counter this, create a separate "step" function which updates the stats
        # and is only called when controlling.
        self.update_state_stats(states)
        states = self.whiten_state(states)
        target = self.random_target_network.predict(states)
        prediction = self.predictor.predict(states)  # TODO: State channels
        mse = np.mean((prediction - target) ** 2, axis=1)
        self.update_reward_stats(mse)
        if not self._ngu:
            return mse / self.reward_stats.std  # RND Paper (sec. 2.4)
        # NGU (sec. 2 Integrating...)
        # Note that in the paper they claim the RND paper used this method, when
        # they did not. Be wary of using this, since the mean can be higher than
        # the mse, and thus get a negative novelty value as a result.
        return 1 + (mse - self.reward_stats.mean) / self.reward_stats.std

    @property
    def random_target_network(self) -> tfk.Model:
        return self._random_target_network

    @property
    def predictor(self) -> tfk.Model:
        return self._predictor


if __name__ == "__main__":
    """Recreate Figure 2 from Section 2.3 (MNIST)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow.keras.datasets.mnist import load_data

    # from tensorflow.keras.utils import to_categorical

    def pick_subset(target, num_target):
        """Pick n numbers of target from MNIST, and fill in the rest with 0s."""

        def take_target_out(target, num_target, y_train, y_test):
            """Take first n of M targets and first M-n zeros."""
            train_target_indices = np.where(y_train == target)[0]
            test_target_indices = np.where(y_test == target)[0]
            train_zero_indices = np.where(y_train == 0)[0]

            total = len(train_target_indices)
            if target == -1:
                num_target = total
            if num_target > total:
                raise ValueError(
                    f"Number of targets can not be greater than the"
                    f" number of available targets! ({num_target}!>{total})"
                )

            return (
                train_target_indices[:num_target],
                train_zero_indices[: total - num_target],
                test_target_indices,
            )

        (x_train, y_train), (x_test, y_test) = load_data(path="mnist.npz")
        target_index, zero_index, test_index = take_target_out(
            target, num_target, y_train, y_test
        )

        training_data = np.ones((len(target_index) + len(zero_index), 28, 28))

        training_data[: len(target_index)] = x_train[target_index]
        training_data[len(target_index) :] = x_train[zero_index]

        test_data = x_test[test_index]

        # The labels aren't used in this experiment, but kept here if needed later
        # training_labels = np.ones(len(target_index) + len(zero_index))
        # training_labels[: len(target_index)] = y_train[target_index] - target + 1
        # training_labels[len(target_index) :] = y_train[zero_index]
        # training_labels = to_categorical(training_labels, num_classes=2)
        # test_labels = to_categorical(y_test[test_index] - target + 1, num_classes=2)

        return (
            training_data.reshape((len(training_data), 28 * 28)),
            test_data.reshape((len(test_data), 28 * 28)),
        )

    def run_mnist_test(target, num_target, epochs, init_tgt_weights, init_pred_weights):
        """Demo novelty/intrinsic reward by fitting RND on a subset of MNIST.

        A target class, e.g. 5, is sampled with n samples. The rest of the
        samples (up to len(target)) are filled with the class 0. The RND is
        then fit on the [0, target] dataset, and predicts for all targets in
        the test dataset. The idea is that the 0 class acts as known states
        and the target class as novel states. The more the target class exists
        in the training set, the less novel it is. In this case, the MSE is
        the measure of novelty/intrinsic reward. It is expected that as more
        targets are included, the end MSE is lower than for less targets.

        The initial weights are kept constant for each different target class.
        """

        training_data, test_data = pick_subset(target, num_target)

        size = 28 * 28
        rnd = RandomNetworkDistillation(size, (2, 20), epochs=epochs)
        if init_tgt_weights is None and init_pred_weights is None:
            # Set initial weights to reapply at the start of every experiment
            init_tgt_weights, init_pred_weights = (
                rnd.random_target_network.get_weights(),
                rnd.predictor.get_weights(),
            )
        else:
            rnd.random_target_network.set_weights(init_tgt_weights)
            rnd.predictor.set_weights(init_pred_weights)

        rnd.fit(training_data)

        test_data = rnd.whiten_state(test_data)
        target = rnd.random_target_network.predict(test_data)
        predictions = rnd.predictor.predict(test_data)

        mse = np.mean((target - predictions) ** 2)

        return mse, init_tgt_weights, init_pred_weights

    figure, ax = plt.subplots(figsize=(20, 10))

    final = {}
    experimental_amounts = [0, 10, 50, 100, 500, 1000, 3000, -1]
    epochs = 10
    for target in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"Testing {target}...")
        init_tgt_weights, init_pred_weights = None, None
        results = []
        for amount in experimental_amounts:
            print(f"\t...{amount}", "samples")
            mse, init_tgt_weights, init_pred_weights = run_mnist_test(
                target, amount, epochs, init_tgt_weights, init_pred_weights
            )
            results.append(mse)
        final[target] = results

    for k, v in final.items():
        ax.plot(v, label=k)
    ax.legend()
    ax.set_xlabel("Number of Samples of Target Class")
    ax.set_ylabel("Test MSE on Target Class")
    ax.set_title(
        f"MSE as a Function of Target Class Balance\n(MNIST Dataset; {epochs} epochs)"
    )
    experimental_amounts[-1] = "All"
    experimental_amounts = [0] + experimental_amounts  # prepend for proper labelling
    ax.set_xticklabels(experimental_amounts)

    ax.grid(True, axis="x")
    plt.savefig("mnist_rnd_test.png", bbox_inches="tight")
    plt.show()
