# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os
import time
from collections import OrderedDict

import numpy as np
import pytest

from psipy.rl.io.batch import Batch, Episode
from psipy.rl.io.sart import SARTWriter
from psipy.rl.plant.tests.mocks import MockAction, MockPlant


@pytest.fixture
def test_data():
    plant = MockPlant(with_meta=True)
    action = MockAction(np.array([1, 2]))
    plant.notify_episode_starts()
    state = plant.check_initial_state(None)  # Supply a state for cost
    return dict(state=state.as_dict(), action=action.as_dict())


@pytest.fixture
def eps1():
    obs = [[1]] * 4 + [[2]]
    act = [[1]] * 4 + [[0]]
    term = [False] * 4 + [True]
    cost = [0] * 5
    eps = Episode(obs, act, term, cost)
    return eps


@pytest.fixture
def batch():
    obs = np.random.random((100, 4))
    act = np.random.random((100, 1))
    term = np.random.random((100, 1)) > 0.5
    cost = np.random.random((100, 1))
    cost[-1] = 100
    eps = Episode(obs, act, term, cost, lookback=2)

    obs = np.random.random((100, 4))
    act = np.random.random((100, 1))
    term = np.random.random((100, 1)) > 0.5
    cost = np.random.random((100, 1))
    cost[-1] = 100
    eps2 = Episode(obs, act, term, cost, lookback=2)

    return Batch([eps, eps2])


@pytest.fixture
def prioritized_batch():
    obs = np.array([1, 2, 3, 4])
    act = np.ones(4)
    term = np.ones(4)
    cost = np.ones(4)
    eps = Episode(obs, act, term, cost)

    obs = np.array([5, 6, 7, 8])
    eps2 = Episode(obs, act, term, cost)
    return Batch([eps, eps2], prioritization="proportional")


def custom_cost(states):
    """Create a cost array that equals [0->100, 777, -100->-1, -777]"""
    costs = []
    for _, state in enumerate(states):
        if state < 300:
            costs.append(state[0] - 200)
        elif state == 300:
            costs.append(777)
        elif state >= 2000 and state < 2100:
            costs.append(state[0] - 2100)
        else:
            costs.append(-777)
    return np.array(costs)


#: Fake bdi options to test splitting a batch on them (has ``len(9)``)
options = [
    "one",
    "two",
    "one",
    "one",
    "two",
    "two",
    "two",
    "one",
    "one",
]


class TestEpisode:
    @staticmethod
    def test_stacks(eps1):
        eps1.lookback = 2
        stack, act = eps1.get_transitions([0, 2])
        assert stack.shape[0] == 2
        assert stack.shape[-1] == eps1.lookback + 1 == 3
        assert act.shape[0] == 2
        assert np.array_equal(stack[1, ..., :-1], [[1.0, 1.0]])
        assert np.array_equal(stack[1, ..., 1:], [[1.0, 2.0]])

    @staticmethod
    def test_indices(eps1):
        assert len(eps1.indices) == 4

    @staticmethod
    def test_observations(eps1):
        assert np.array_equal(eps1.observations, np.array([[1], [1], [1], [1], [2]]))

    @staticmethod
    def test_terminals(eps1):
        assert np.array_equal(eps1.terminals, np.array([False, False, False, True]))

    @staticmethod
    def test_is_valid(eps1):
        eps1._observations = np.array([1, np.nan, 2])
        assert not eps1.is_valid()

    @staticmethod
    def test_from_hdf5(temp_dir, eps1):
        # Data will come in from the loop as dict of scalars hence list
        obs = [1] * 4 + [2]
        act = [1] * 4 + [0]
        term = [False] * 4 + [True]
        reward = [0] * 5
        writer = SARTWriter(temp_dir, "Test", 1)
        for i in range(len(obs)):
            data = OrderedDict(
                state=OrderedDict(
                    {
                        "cost": reward[i],
                        "terminal": term[i],
                        "values": OrderedDict({"obs1": obs[i]}),
                        "meta": {"meta1": 0},
                    }
                ),
                action=OrderedDict({"act1": act[i]}),
            )
            writer.append(data)
        time.sleep(0.5)
        writer.notify_episode_stops()
        f = os.listdir(temp_dir)[0]
        eps = Episode.from_hdf5(os.path.join(temp_dir, f))
        assert eps == eps1

    @staticmethod
    def test_equality(batch, eps1):
        eps2 = batch._episodes[0]
        assert not eps1 == eps2
        assert not eps1 == batch
        assert eps1 == eps1

    @staticmethod
    def test_split_on_action_key(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        for i in range(10):
            try:
                test_data["action"].update(dict(option=options[i]))
            except IndexError:
                pass
            writer.append(test_data)
        writer.notify_episode_stops()

        episodes = Episode.multiple_from_key_hdf5(
            filepath=os.path.join(temp_dir, os.listdir(temp_dir)[0]),
            key_source="action",
            key="option",
        )
        assert len(episodes["one"]) == 3
        assert len(episodes["two"]) == 2

        one_lengths = [1, 2, 3]
        two_lengths = [1, 3]
        for i, ep in enumerate(episodes["one"]):
            assert len(ep._observations) == one_lengths[i]
        for i, ep in enumerate(episodes["two"]):
            assert len(ep._observations) == two_lengths[i]

    @staticmethod
    def test_split_on_action_key_value(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        for i in range(10):
            try:
                test_data["action"].update(dict(option=options[i]))
            except IndexError:
                pass
            writer.append(test_data)
        writer.notify_episode_stops()

        episodes = Episode.multiple_from_key_hdf5(
            filepath=os.path.join(temp_dir, os.listdir(temp_dir)[0]),
            key_source="action",
            key="option",
            value="one",
        )
        assert list(episodes.keys()) == ["one"]
        assert len(episodes["one"]) == 3

    @staticmethod
    def test_split_on_meta_key(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        for i in range(10):
            if i % 2 == 0:
                test_data["state"]["meta"]["meta3"] = "One"
            else:
                test_data["state"]["meta"]["meta3"] = "Two"
            writer.append(test_data)
        writer.notify_episode_stops()

        episodes = Episode.multiple_from_key_hdf5(
            filepath=os.path.join(temp_dir, os.listdir(temp_dir)[0]),
            key_source="meta",
            key="meta3",
        )
        assert len(episodes["One"]) == 5
        assert len(episodes["Two"]) == 5

        for ep in episodes["One"]:
            assert len(ep._observations) == 1
        for ep in episodes["Two"]:
            assert len(ep._observations) == 1

    @staticmethod
    def test_split_on_meta_key_value(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        for i in range(10):
            if i % 2 == 0:
                test_data["state"]["meta"]["meta3"] = "One"
            else:
                test_data["state"]["meta"]["meta3"] = "Two"
            writer.append(test_data)
        writer.notify_episode_stops()

        episodes = Episode.multiple_from_key_hdf5(
            filepath=os.path.join(temp_dir, os.listdir(temp_dir)[0]),
            key_source="meta",
            key="meta3",
            value="One",
        )
        assert list(episodes.keys()) == ["One"]
        assert len(episodes["One"]) == 5

    @staticmethod
    def test_split_nonexistent_key(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        for _ in range(2):
            writer.append(test_data)
        writer.notify_episode_stops()

        with pytest.raises(KeyError):
            Episode.multiple_from_key_hdf5(
                filepath=os.path.join(temp_dir, os.listdir(temp_dir)[0]),
                key_source="meta",
                key="meta3",
                value="unknown",
            )

    @staticmethod
    def test_remove_string_axes():
        # Arrays come through with object dtype
        action = np.array([[1, 2, 3, "p"], [1, 2, 3, "op"]], dtype="object")
        action = Episode.remove_string_axes(action)
        solution = np.array([[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(action, solution)


class TestBatch:
    # Sort and shuffle implicitly tested via other tests
    @staticmethod
    def test_len(batch):
        num_samples = 200 - (batch.lookback * batch.num_episodes)
        assert batch.num_samples == num_samples
        batch.set_minibatch_size(num_samples // 2)
        assert len(batch) == 2
        batch.set_minibatch_size(-1)
        assert len(batch) == 1

    @staticmethod
    def test_targets(batch):
        targets = np.random.random((batch.num_samples, 1))
        batch.set_targets(targets).set_minibatch_size(-1).sort()
        _, t, _ = batch.statesactions_targets[0]
        assert np.allclose(targets, t)
        # Handles missing singleton dimension, remembers minibatch_size and sort.
        targets = np.random.random(batch.num_samples)
        batch.set_targets(targets)
        _, t, _ = batch.statesactions_targets[0]
        assert np.allclose(targets[..., None], t)
        # Handles additional singleton dimension and remembers sort through
        # minibatch_size change.
        targets = np.random.random((batch.num_samples, 1, 1))
        batch.set_targets(targets).set_minibatch_size(batch.num_samples // 2)
        _, t1, _ = batch.statesactions_targets[0]
        _, t2, _ = batch.statesactions_targets[1]
        t = np.vstack((t1, t2))
        assert np.allclose(targets, t)

    @staticmethod
    def test_targets_shuffle(batch):
        """Assert that shuffled minibatches retain state <-> target relationships."""
        # Miss-using the targets as indices in order to be able to match
        # shuffled states and actions to sorted states and actions.
        targets = np.arange(0, batch.num_samples, dtype=np.int32)
        batch.set_targets(targets).set_minibatch_size(-1)
        (s1, a1), t1, _ = batch.sort().statesactions_targets[0]
        (s2, a2), t2, _ = batch.shuffle().statesactions_targets[0]
        permutation = t2.flatten().astype(np.uint32)
        assert np.array_equal(t1[permutation, ...], t2), "Wat?"
        assert np.array_equal(a1[permutation, ...], a2)
        assert np.array_equal(s1[permutation, ...], s2)

    @staticmethod
    def test_costs(batch):
        """Assert that shuffled minibatches retain costs <-> terminal relationships."""

        prev = 0

        def compute_costs(states: np.ndarray) -> np.ndarray:
            nonlocal prev
            val = (np.arange(len(states)) + prev) - (batch.lookback - 1)
            # Since we are using the costs as indices, we need to roll them in the
            # opposite direction as the costs are shifted.  Otherwise we will be
            # missing an index on both edges.
            # c(s, u) == c(s')
            val = np.roll(val, 1)
            # len(states) - 1 since we exclude the
            # terminal state (no c(s, u)==c(s') for that state)
            prev += val.max()
            return val

        # Miss-using the targets as indices in order to be able to match
        # shuffled states and actions to sorted states and actions.
        # Splitting Batch into two minibatches in order to not run into the problem
        # of batch internally sorting all minibatches.
        batch.compute_costs(compute_costs).set_minibatch_size(batch.num_samples // 2)
        batch.sort()
        c11, t11 = batch.costs_terminals[0]
        c12, t12 = batch.costs_terminals[1]

        batch.shuffle()
        c21, t21 = batch.costs_terminals[0]
        c22, t22 = batch.costs_terminals[1]

        # Recollect minibatches in full stacks in order to compare sorted and
        # shuffled data. In the shuffled instance, the first and second batch
        # will contain different data than the first and second batch in the
        # sorted case.
        c1, t1 = np.vstack((c11, c12)), np.vstack((t11, t12))
        c2, t2 = np.vstack((c21, c22)), np.vstack((t21, t22))
        permutation = c2.flatten().astype(int)
        assert np.array_equal(c1[permutation, ...], c2), "Wat?"
        assert np.array_equal(t1[permutation, ...], t2)

    @staticmethod
    def test_use_episode_costs(batch):
        costs = []
        for e in batch._episodes:
            # Terminals get dropped via batch indices, which is based on len(episode)
            costs.extend(e.costs[:-1])
        costs = np.array(costs)
        batch.set_minibatch_size(-1)
        batch_costs = batch.costs_terminals[0][0]
        assert np.array_equal(batch_costs, costs)
        # Note: the below test should be used if we shift
        # episode costs to include terminal costs.
        # assert batch_costs[-1] == costs[-1] == 100

    @staticmethod
    def test_mode_error(batch):
        with pytest.raises(AssertionError):
            batch[0]

    @staticmethod
    def test_costs_properly_assigned_to_states():
        obs = np.arange(200, 301)[..., None]
        act = np.arange(100, 201)[..., None]
        term = np.array([False] * 101)[..., None]
        term[-1] = True

        obs2 = np.arange(2000, 2101)[..., None]
        act2 = np.arange(1000, 1101)[..., None]
        term2 = np.array([False] * 101)[..., None]
        term2[-1] = True

        calculated_cost = custom_cost(np.array([*obs, *obs2]))
        cost = calculated_cost[: len(calculated_cost) // 2]
        cost2 = calculated_cost[len(calculated_cost) // 2 :]
        eps = Episode(obs, act, term, cost, lookback=2)
        eps2 = Episode(obs2, act2, term2, cost2, lookback=2)

        # Original cost is altered because we drop lookback-1 states and terminal state
        original_costs_trimmed = np.append(cost[1:-1], cost2[1:-1])
        ordered_batch = Batch([eps, eps2])
        states, costs = ordered_batch.sort().set_minibatch_size(-1).states_costs[0]
        _, terminals = ordered_batch.costs_terminals[0]

        assert len(states) == len(costs)
        assert terminals[0] == False  # noqa
        assert costs[0] == 1
        # Index on [0,1] because lookback is 2
        assert states[0, ..., 1] == 201

        # We exclude terminal states
        assert costs[-1] == -1
        assert states[-1, ..., 1] == 2099
        # Terminal is defined as t(s,a) := True iff s' == terminal
        assert terminals[-1] == True  # noqa

        # Assert everything
        assert np.array_equal(
            costs[: len(costs) // 2],
            original_costs_trimmed[: len(original_costs_trimmed) // 2],
        )
        assert np.array_equal(
            costs[len(costs) // 2 :],
            original_costs_trimmed[len(original_costs_trimmed) // 2 :],
        )

        # Now test custom cost
        ordered_batch.compute_costs(custom_cost)
        states1, costs = ordered_batch.states_costs[0]
        assert np.array_equal(states, states1)
        # Costs are shifted, so even though the cost function is the same, it is shifted
        assert costs[0] == 2
        # Terminal state is included, otherwise the array would be 1 too short
        assert costs[-1] == -777
        assert len(costs) == len(states)

    @staticmethod
    def test_append(eps1):
        b = Batch([eps1]).set_minibatch_size(1)
        assert b.num_episodes == 1
        b.append([eps1])
        assert b.num_episodes == 2
        assert b._episodes[0] == b._episodes[1]

    @staticmethod
    def create_fake_hdf5_file(dir: str, episode: int):
        writer = SARTWriter(dir, "Testing", episode)
        action_meta = ["one", "one", "two", "two", "one", "one"]
        for i in range(6):
            data = dict(
                state={
                    "values": OrderedDict({"state1": 1.0}),
                    "cost": 0,
                    "terminal": False,
                    "meta": {"meta1": "hello"},
                },
                action=OrderedDict({"action1": 0.0, "action_meta": action_meta[i]}),
            )
            writer.append(data)
        writer.notify_episode_stops()
        time.sleep(0.1)

    @staticmethod
    def test_load_from_hdf5(temp_dir):
        TestBatch.create_fake_hdf5_file(temp_dir, 1)
        TestBatch.create_fake_hdf5_file(temp_dir, 2)
        TestBatch.create_fake_hdf5_file(temp_dir, 3)
        TestBatch.create_fake_hdf5_file(temp_dir, 4)
        batch = Batch.from_hdf5(temp_dir)
        assert batch.num_episodes == 4

    @staticmethod
    def test_load_only_new_from_hdf5(temp_dir):
        TestBatch.create_fake_hdf5_file(temp_dir, 1)
        TestBatch.create_fake_hdf5_file(temp_dir, 2)
        TestBatch.create_fake_hdf5_file(temp_dir, 3)
        TestBatch.create_fake_hdf5_file(temp_dir, 4)
        batch = Batch.from_hdf5(temp_dir, only_newest=2)
        assert batch.num_episodes == 2
        loaded = batch._loaded_sart_paths
        trimmed = set(path.split(os.sep)[-1].split("-")[3] for path in loaded)
        assert trimmed == {"3", "4"}

    @staticmethod
    def test_load_only_latest_hdf5(temp_dir):
        TestBatch.create_fake_hdf5_file(temp_dir, 1)
        batch = Batch.from_hdf5(temp_dir, only_newest=1)
        assert batch.num_episodes == 1
        loaded = batch._loaded_sart_paths
        trimmed = set(path.split(os.sep)[-1].split("-")[3] for path in loaded)
        assert trimmed == {"1"}
        TestBatch.create_fake_hdf5_file(temp_dir, 2)
        batch = Batch.from_hdf5(temp_dir, only_newest=1)
        assert batch.num_episodes == 1
        loaded = batch._loaded_sart_paths
        trimmed = set(path.split(os.sep)[-1].split("-")[3] for path in loaded)
        assert trimmed == {"2"}
        TestBatch.create_fake_hdf5_file(temp_dir, 3)
        batch = Batch.from_hdf5(temp_dir, only_newest=1)
        assert batch.num_episodes == 1
        loaded = batch._loaded_sart_paths
        trimmed = set(path.split(os.sep)[-1].split("-")[3] for path in loaded)
        assert trimmed == {"3"}

    @staticmethod
    def test_load_more_new_than_begins_with(temp_dir):
        TestBatch.create_fake_hdf5_file(temp_dir, 1)
        batch = Batch.from_hdf5(temp_dir, only_newest=2)
        assert batch.num_episodes == 1
        loaded = batch._loaded_sart_paths
        print(loaded)
        trimmed = set(path.split(os.sep)[-1].split("-")[3] for path in loaded)
        assert trimmed == {"1"}
        TestBatch.create_fake_hdf5_file(temp_dir, 2)
        batch = Batch.from_hdf5(temp_dir, only_newest=2)
        assert batch.num_episodes == 2
        loaded = batch._loaded_sart_paths
        trimmed = set(path.split(os.sep)[-1].split("-")[3] for path in loaded)
        assert trimmed == {"1", "2"}
        TestBatch.create_fake_hdf5_file(temp_dir, 3)
        batch = Batch.from_hdf5(temp_dir, only_newest=2)
        assert batch.num_episodes == 2
        loaded = batch._loaded_sart_paths
        trimmed = set(path.split(os.sep)[-1].split("-")[3] for path in loaded)
        assert trimmed == {"2", "3"}

    @staticmethod
    def test_append_hdf5(temp_dir):
        TestBatch.create_fake_hdf5_file(temp_dir, 1)
        TestBatch.create_fake_hdf5_file(temp_dir, 2)
        batch = Batch.from_hdf5(temp_dir)
        assert batch.num_episodes == 2
        TestBatch.create_fake_hdf5_file(temp_dir, 3)
        batch.append_from_hdf5(temp_dir)
        assert batch.num_episodes == 3

    @staticmethod
    def test_append_only_new_hdf5(temp_dir):
        TestBatch.create_fake_hdf5_file(temp_dir, 1)
        TestBatch.create_fake_hdf5_file(temp_dir, 2)
        batch = Batch.from_hdf5(temp_dir, only_newest=1)
        assert batch.num_episodes == 1
        TestBatch.create_fake_hdf5_file(temp_dir, 3)
        batch.append_from_hdf5(temp_dir)
        assert batch.num_episodes == 2

    @staticmethod
    def test_load_multiple_from_key_hdf5(temp_dir):
        num_eps = 3
        for i in range(1, num_eps + 1):
            TestBatch.create_fake_hdf5_file(temp_dir, i)
        batches = Batch.multiple_from_key_hdf5(
            temp_dir, key_source="action", key="action_meta"
        )
        # In total there should be 2 "one" episodes and 1 "two" episode per hdf5
        assert list(batches.keys()) == ["one", "two"]
        assert batches["one"].num_episodes == 2 * num_eps
        assert batches["two"].num_episodes == 1 * num_eps

    @staticmethod
    def test_append_multiple_from_key_hdf5(temp_dir):
        TestBatch.create_fake_hdf5_file(temp_dir, 1)
        batches = Batch.multiple_from_key_hdf5(
            temp_dir, key_source="action", key="action_meta"
        )
        batch = batches["one"]
        TestBatch.create_fake_hdf5_file(temp_dir, 2)
        batch.append_multiple_from_key_hdf5(
            temp_dir, key_source="action", key="action_meta", value="one"
        )
        assert batch.num_episodes == 4

    @staticmethod
    def test_proper_probs_and_weights(prioritized_batch):
        from psipy.rl.io.batch import e

        # Proportional
        delta = np.array([1, 1, 1, 2, 2, -2])
        p_answer = np.abs(delta) + e
        p_answer = p_answer / np.sum(p_answer)
        w_answer = prioritized_batch.num_samples * p_answer
        w_answer = w_answer / np.max(w_answer)

        prioritized_batch.set_delta(delta)
        prioritized_batch.alpha = 1  # make the math easier
        prioritized_batch.beta = -1  # make the math easier (== 1 exponent)
        p, w = prioritized_batch.get_sample_distribution()

        np.testing.assert_allclose(p, p_answer)  # because floats
        np.testing.assert_allclose(w, w_answer)

        # Rank
        # Test with repeated deltas
        p_answer = np.array([1 / 2, 1 / 2, 1 / 2, 1 / 1, 1 / 1, 1 / 1])
        p_answer = p_answer / np.sum(p_answer)
        prioritized_batch.prioritization = "rank"
        p, _ = prioritized_batch.get_sample_distribution()
        np.testing.assert_allclose(p, p_answer)

        # Test without repeated deltas
        delta = np.array([1, 2, 3, 5, 6, 7])
        prioritized_batch.set_delta(delta)
        p_answer = np.array([1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1 / 1])
        p_answer = p_answer / np.sum(p_answer)
        p, _ = prioritized_batch.get_sample_distribution()
        np.testing.assert_allclose(p, p_answer)

    @staticmethod
    def test_weights_properly_sliced(prioritized_batch):
        from psipy.rl.io.batch import e

        # Proportional
        delta = np.array([1, 1, 1, 2, 2, -2])
        sum_ = np.sum(np.abs(delta))
        p_answer = (np.abs(delta) + e) / sum_  # asserted above to be correct
        w_answer = prioritized_batch.num_samples * p_answer
        w_answer = w_answer / np.max(w_answer)

        prioritized_batch.set_delta(delta)
        prioritized_batch.alpha = 1  # make the math easier
        prioritized_batch.beta = -1  # make the math easier (== 1 exponent)
        # Miss-using the targets as indices in order to be able to match
        # shuffled states and actions to sorted states and actions.
        prioritized_batch.set_minibatch_size(-1).sort()
        targets = np.arange(0, prioritized_batch.num_samples, dtype=np.int32)
        prioritized_batch.set_targets(targets)
        prioritized_batch.shuffle()

        _, t, w = prioritized_batch.statesactions_targets[0]
        permutation = t.flatten().astype(np.uint32)
        np.testing.assert_allclose(
            w, w_answer[permutation][..., None]
        )  # should be all w's

        prioritized_batch.set_minibatch_size(2)
        _, t, w = prioritized_batch.statesactions_targets[0]
        permutation = t.flatten().astype(np.uint32)
        np.testing.assert_allclose(w, w_answer[permutation][:2][..., None])
