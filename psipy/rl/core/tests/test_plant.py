# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

from collections import OrderedDict

import numpy as np
import pytest

from psipy.rl.core.plant import Action, State, TerminalStates
from psipy.rl.plants.tests.mocks import MockAction, MockState


@pytest.fixture
def terminal_state_simple():
    class Terminal(TerminalStates):
        def cond1(self, state):
            return False

        def cond2(self, state):
            return True

    return Terminal


@pytest.fixture
def terminal_state_with_state():
    class Terminal2(TerminalStates):
        count = 0

        def cond1(self, state):
            return False

        def counter(self, state):
            """Only hits when something occurs 2 times"""
            self.count += 1
            if self.count == 2:
                return True
            return False

    return Terminal2


@pytest.fixture
def terminal_state_with_property():
    class Terminal3(TerminalStates):
        @property
        def true(self):
            return True

        def false(self, state):
            return False

    return Terminal3


class TestAction:
    @staticmethod
    def test_init():
        act = MockAction(np.array([1, None]))
        assert act.is_partial()
        act = MockAction(np.array([1, 2]))
        assert not act.is_partial()
        with pytest.raises(ValueError):
            MockAction(np.array([1, np.nan]))

    @staticmethod
    def test_equality():
        action1 = MockAction(np.array([1, 2]))
        action2 = MockAction(np.array([1, 2]))
        action3 = MockAction(np.array([2, 3]))
        assert action1 == action2
        assert action1 != action3

    @staticmethod
    def test_edit_dict_not_edit_action():
        action1 = MockAction(np.array([1, 2]))
        action_dict = action1.as_dict()
        action_dict["430PP41FIC006/PID_CV_MAN"] = 5
        assert action_dict != action1.as_dict()

    @staticmethod
    def test_semantic_dict():
        action1 = MockAction(np.array([1, 2]))
        semantic_1 = action1.as_semantic_dict()
        assert list(semantic_1.keys()) == ["Action1", "Action2"]

    @staticmethod
    def test_merge():
        action1 = MockAction(np.array([2, None]))
        action2 = MockAction(np.array([None, 1]))
        merged = Action.merge(action1, action2)
        assert merged == MockAction(np.array([2, 1]))

        action1 = MockAction({"430PP41FIC006/PID_CV_MAN": 2})
        action2 = MockAction({"430PP42FIC006/PID_CV_MAN": 1})
        merged = Action.merge(action1, action2)
        assert merged == MockAction(np.array([2, 1]))
        assert merged.keys() == ("430PP41FIC006/PID_CV_MAN", "430PP42FIC006/PID_CV_MAN")

        # Test merge and .keys() with additional metadata channels.
        action1 = MockAction(
            {"430PP41FIC006/PID_CV_MAN": 2}, additional_data=dict(some_channel="abc")
        )
        merged = Action.merge(action1, action2)
        assert merged.keys() == ("430PP41FIC006/PID_CV_MAN", "430PP42FIC006/PID_CV_MAN")
        assert merged.keys(with_additional=True) == (
            "430PP41FIC006/PID_CV_MAN",
            "430PP42FIC006/PID_CV_MAN",
            "some_channel",
        )

        action2 = MockAction({"430PP42FIC006/PID_CV_MAN": 1})
        with pytest.raises(ValueError):  # single partial action
            Action.merge(action2)

    @staticmethod
    def test_invalid_merge():
        class MockAction2(Action):
            channels = ("a", "b")
            legal_values = ((0, 1000), (0, 1000))

        # All actions are of the same class
        with pytest.raises(ValueError):
            action1 = MockAction2(np.array([2, None]))
            action2 = MockAction(np.array([None, 1]))
            Action.merge(action1, action2)
        # All actions are partial
        with pytest.raises(ValueError):
            action1 = MockAction(np.array([2, 5]))
            action2 = MockAction(np.array([None, 1]))
            Action.merge(action1, action2)
        # No actions overlap
        with pytest.raises(ValueError):

            class ThreeAction(Action):
                channels = ("one", "two", "three")
                legal_values = ((0, 10), (0, 10), (0, 10))

            action1 = ThreeAction(np.array([1, None, 4]))
            action2 = ThreeAction(np.array([None, 8, 5]))
            Action.merge(action1, action2)
        # There are no unfilled action values""
        with pytest.raises(ValueError):
            action1 = MockAction(np.array([2, 5]))
            action2 = MockAction(np.array([None, None]))
            Action.merge(action1, action2)

    @staticmethod
    def test_merge_one_action():
        with pytest.raises(ValueError):
            Action.merge(MockAction(np.array([2, None])))
        action = MockAction(np.array([2, 3]))
        merged = Action.merge(action)
        assert merged == action

    @staticmethod
    def test_array_dict_duality():
        array_repr = np.array([2, 3])
        dict_repr = {k: v for (k, v) in zip(MockAction.channels, array_repr)}
        action = MockAction(array_repr)
        assert np.array_equal(action.as_array(), array_repr)
        assert action.as_dict() == dict_repr
        action2 = MockAction(dict_repr)
        assert action2.as_dict() == dict_repr
        assert np.array_equal(action2.as_array(), array_repr)
        assert action == action2

    @staticmethod
    def test_invalid_action_values():
        class DAction(Action):
            legal_values = (range(1, 11), range(-5, 6))  # 1-10, -5,5
            channels = ("one", "two")
            dtype = "discrete"

        # Test the discrete action case
        a = DAction(np.array([1, -5]))
        a = DAction(np.array([10, 5]))
        a = DAction(np.array([1, 0]))
        with pytest.raises(ValueError):
            a = DAction(np.array([0, 0]))
        with pytest.raises(ValueError):
            a = DAction(np.array([5, -10]))

        # Test the continuous action case
        a = MockAction(np.array([1, 10]))  # upper limit is ok
        a = MockAction(np.array([5.2, 5.2]))
        with pytest.raises(ValueError):
            a = MockAction(np.array([-10, 5.5]))
        with pytest.raises(ValueError):
            a = MockAction(np.array([5.5, 11]))

        del a  # satisfy mypy :)


class TestState:
    @staticmethod
    def test_annotate():
        class S(State):
            _channels = ("hello",)

        s = S(np.array([1]))
        assert isinstance(s, S)
        assert s.cost == 0
        s.annotate(1, True)
        assert isinstance(s, S)
        assert s.cost == 1

    @staticmethod
    def test_keys():
        assert MockState.channels() == MockState._channels
        dct = dict(zip(MockState.channels(), range(len(MockState.channels()))))
        mockstate = MockState(dct)
        assert mockstate.keys() == MockState.channels()
        assert "430PP42FIC006/PID_PV" in mockstate
        assert "nope" not in mockstate

    @staticmethod
    def test_creation():
        array_repr = np.arange(len(MockState.channels()))
        dict_repr = dict(zip(MockState.channels(), array_repr))
        dict_repr_with_more = OrderedDict(values=dict_repr)
        dict_repr_with_more["cost"] = 0
        dict_repr_with_more["terminal"] = False
        dict_repr_with_more["meta"] = {}
        state = MockState(array_repr)
        assert state.as_array() is not array_repr
        assert np.array_equal(state.as_array(), array_repr)
        assert state.as_dict() is not dict_repr_with_more
        assert state.as_dict() == dict_repr_with_more

        # Test creation with too many values in a dict -- superset' values
        # should be ignored
        dict_repr["another_key"] = 42
        state = MockState(dict_repr)
        del dict_repr["another_key"]
        assert state.as_dict() is not dict_repr
        assert state.as_dict() == dict_repr_with_more
        assert np.array_equal(state.as_array(), array_repr)
        assert state == state

        # Test creation with too many values in a numpy array
        array_repr = np.arange(len(MockState.channels()) + 1)
        with pytest.raises(AssertionError):
            MockState(array_repr)

    @staticmethod
    def test_none_replacement():
        class S(State):
            _channels = ("one", "two")

        # Make sure we can write normal channels
        s_dict_good = S({"one": 1, "two": 2})
        s_good = S(np.array([1, 2]))
        assert s_dict_good == s_good
        # Check writing None channels
        s_none = S(np.array([1, None]))
        s_dict_none = S({"one": 1, "two": None})

        assert s_none["two"] == 0.0
        assert s_dict_none["two"] == 0.0

    @staticmethod
    def test_coerce_dtype():
        assert State._coerce_dtype(1.0) == 1.0
        assert State._coerce_dtype(1) == 1.0
        assert State._coerce_dtype(-1) == -1.0
        assert State._coerce_dtype("123") == 123.0

        assert State._coerce_dtype(None) == 0.0
        assert State._coerce_dtype("non-coerceable") == 0.0
        assert State._coerce_dtype(float("inf")) == 0.0
        assert State._coerce_dtype(float("nan")) == 0.0

        assert State._coerce_dtype(None, default=23) == 23.0
        assert State._coerce_dtype("non-coerceable", default="123") == 123.0
        assert State._coerce_dtype(float("nan"), default="123") == 123.0
        assert State._coerce_dtype(float("inf"), default="123") == 123.0

        # Arguments `name` and `default` are keyword only.
        with pytest.raises(TypeError):
            State._coerce_dtype(float("inf"), "123")

    @staticmethod
    def test_semantic_channels():
        # Check for the tuple channel case which does nothing
        state = MockState(np.arange(len(MockState.channels())))
        assert state.semantic_channels() is state.channels()

        class S(State):
            _channels = {
                "One": {"desc": "First"},
                "Two": {"desc": "Second"},
                "Three": {"skipme": "skip!"},
            }

        s = S(np.array([1, 2, 3]))
        assert s.semantic_channels() == ("First", "Second", "Three")

    @staticmethod
    def test_semantic_dict():
        # Check the tuple backwards compatible case
        state = MockState(np.arange(len(MockState.channels())))
        assert state.as_dict(semantic=True) == state.as_dict()

        class S(State):
            _channels = {
                "One": {"desc": "First"},
                "Two": {"desc": "Second"},
                "Three": {"skipme": "skip!"},
            }

        s = S(np.array([1, 2, 3]))
        assert s.as_dict(semantic=True)["values"] == {
            "First": 1,
            "Second": 2,
            "Three": 3,
        }

    @staticmethod
    def test_copy():
        array_repr = np.arange(len(MockState.channels()))
        dict_repr = dict(zip(MockState.channels(), array_repr))
        dict_repr_with_more = OrderedDict(values=dict_repr)
        dict_repr_with_more["cost"] = 0
        dict_repr_with_more["terminal"] = False
        dict_repr_with_more["meta"] = {}
        state = MockState(array_repr)
        new_state = state.copy()
        assert new_state == state

        # Test with custom meta
        state.meta = {"This is a test": True}
        new_state2 = state.copy()
        assert new_state2 == state

        # Assure we weren't creating references to original state
        assert new_state2 != new_state


class TestTerminalStates:
    @staticmethod
    def test_cond2(terminal_state_simple):
        assert terminal_state_simple.determine_if_terminal(None)

    @staticmethod
    def test_counter(terminal_state_with_state):
        # Counter limit not yet reached
        assert not terminal_state_with_state.determine_if_terminal(None)
        # Counter limit reached
        assert terminal_state_with_state.determine_if_terminal(None)

    @staticmethod
    def test_property_has_no_effect(terminal_state_with_property):
        assert not terminal_state_with_property.determine_if_terminal(None)

    @staticmethod
    def test_disabled(terminal_state_simple):
        terminal_state_simple.disable()
        assert not terminal_state_simple.determine_if_terminal(None)
