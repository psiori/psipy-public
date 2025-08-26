# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os
import re
import tempfile
import numpy as np
import pytest

from psipy.rl.core.controller import ContinuousRandomActionController
from psipy.rl.core.controller import DiscreteRandomActionController
from psipy.rl.core.plant import Action, State


@pytest.fixture
def disc_action():
    class DiscreteAction(Action):
        dtype = "discrete"
        channels = ("channel1", "channel2", "channel3")
        legal_values = (range(11), range(-10, -4), np.linspace(100.1, 200.5, 3))

    return DiscreteAction


@pytest.fixture
def cont_action():
    class ContinuousAction(Action):
        dtype = "continuous"
        channels = ("channel1", "channel2", "channel3")
        legal_values = ((0, 100), (-10, 10), (-10, -5))

    return ContinuousAction


@pytest.fixture
def negative_and_positive_range():
    class ContinuousAction(Action):
        dtype = "continuous"
        channels = ("channel1",)
        legal_values = ((-1, 1),)

    return ContinuousAction


@pytest.fixture
def custom_state():
    class CustomState(State):
        _channels = ("state1", "state2")

    data = np.ones(len(CustomState.channels()))
    state = CustomState(data)
    return state


class TestContinuousRandomController:
    @staticmethod
    def test_sampling(cont_action, custom_state):
        crc = ContinuousRandomActionController(custom_state.channels(), cont_action)
        action = crc.get_action(custom_state)
        assert action["channel1"] >= 0
        assert action["channel3"] < 0
        assert len(action) == 3

    @staticmethod
    def test_fail_on_misuse_with_discrete_action(disc_action, custom_state):
        with pytest.raises(ValueError):
            ContinuousRandomActionController(custom_state.channels(), disc_action)


class TestDiscreteRandomController:
    @staticmethod
    def test_discrete(disc_action, custom_state):
        drc = DiscreteRandomActionController(custom_state.channels(), disc_action)
        action = drc.get_action(custom_state)
        # The first two channels are drawn from a list of ints
        assert action["channel1"] == int(action["channel1"])
        assert action["channel2"] == int(action["channel2"])
        # Third channel is drawn from a list of floats
        assert action["channel3"] != int(action["channel3"])
        assert action["channel1"] >= 0
        assert action["channel2"] < 0
        assert action["channel3"] > 100
        assert len(action) == 3

    @staticmethod
    def test_fail_on_misuse_with_continuous_action(cont_action, custom_state):
        with pytest.raises(ValueError):
            DiscreteRandomActionController(custom_state.channels(), cont_action)

    @staticmethod
    def test_repeating_action(cont_action, custom_state):
        crc = ContinuousRandomActionController(
            custom_state.channels(), cont_action, num_repeat=2
        )
        action = crc.get_action(custom_state)
        # Must make copy or action will always be the same
        saved_action = cont_action(action.as_array())
        assert crc.get_action(custom_state) == saved_action
        assert crc.get_action(custom_state) != saved_action


class TestPartialControllers:
    @staticmethod
    def test_partial_fill(disc_action, custom_state):
        first2 = DiscreteRandomActionController(
            custom_state.channels(),
            disc_action,
            action_channels=("channel1", "channel2"),
        )
        last = DiscreteRandomActionController(
            custom_state.channels(), disc_action, action_channels=("channel3",)
        )
        assert first2.is_partial()
        assert last.is_partial()
        act1 = first2.get_action(custom_state)
        act2 = last.get_action(custom_state)
        with pytest.raises(ValueError):  # cannot convert partial to dense
            act1.as_array()
        with pytest.raises(ValueError):  # cannot convert partial to dense
            act2.as_array()
        assert act1.keys() == ("channel1", "channel2")
        assert act2.keys() == ("channel3",)

    @staticmethod
    def test_convert_partial_to_dense(disc_action, custom_state):
        first2 = DiscreteRandomActionController(
            custom_state.channels(),
            disc_action,
            action_channels=("channel1", "channel2"),
        )
        action = first2.get_action(custom_state)
        with pytest.raises(ValueError):  # Cannot convert partial action to dense array
            action.as_array()
        assert tuple(sorted(action.as_dict().keys())) == ("channel1", "channel2")

    @staticmethod
    def test_default_not_partial(disc_action, custom_state):
        c = DiscreteRandomActionController(custom_state.channels(), disc_action)
        assert not c.is_partial()


class TestControllerIDMixin:
    """Test IDMixin functionality integrated into Controller classes."""

    @staticmethod
    def test_controller_has_unique_id(cont_action, custom_state):
        """Test that controllers get unique IDs on creation."""
        controller1 = ContinuousRandomActionController(custom_state.channels(), cont_action)
        controller2 = ContinuousRandomActionController(custom_state.channels(), cont_action)
        
        # Both should have valid IDs
        assert hasattr(controller1, 'id')
        assert hasattr(controller2, 'id')
        
        # IDs should be different
        assert controller1.id != controller2.id

    @staticmethod
    def test_generation_increment(cont_action, custom_state):
        """Test generation increment functionality."""
        controller = ContinuousRandomActionController(custom_state.channels(), cont_action)
        
        # Should start at generation 0
        assert controller.id_generation == 0
        assert controller.id.endswith('_0000')
        
        # Increment generation
        original_strand = controller.id_strand
        controller.increment_generation()
        
        assert controller.id_generation == 1
        assert controller.id.endswith('_0001')
        assert controller.id_strand == original_strand  # Strand unchanged

    @staticmethod
    def test_id_parsing(cont_action, custom_state):
        """Test ID string parsing and setting."""
        controller = ContinuousRandomActionController(custom_state.channels(), cont_action)
        
        # Set ID from valid string
        test_id = "20241201143022-A3F2E_0005"
        controller.set_id_from_string(test_id)
        
        assert controller.id == test_id
        assert controller.id_strand == "20241201143022-A3F2E"
        assert controller.id_generation == 5

    @staticmethod
    def test_get_default_basename_uses_id(cont_action, custom_state):
        """Test that default basename includes the controller ID."""
        controller = ContinuousRandomActionController(custom_state.channels(), cont_action)
        basename = controller.get_default_basename()
        
        assert basename.startswith("controller-")
        assert controller.id in basename
