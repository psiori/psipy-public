# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Mock components for plant, action, state etc."""

import time
from collections import OrderedDict

import numpy as np

from psipy.rl.core.plant import Action, Plant, State
from psipy.rl.core.plant import TerminalStates

__all__ = [
    "MockAction",
    "MockPlant",
    "MockState",
    "MockTerminal",
    "MockDiscreteAction",
    "MockSingleChannelAction",
]


class MockState(State):
    _channels = (
        "430PP42FIC006/PID_PV",
        "430PP42DIC007/PID_PV",
        "430PP41DIC007/PID_PV",
        "430SL612WIX001_PV/PV",
        "430TH01PT007_PV/PV",
        "430SL612FIX001_PV/PV",
        "430PP41FIC006/PID_PV",
        "430TH01AT027_PV",
        "430SL612FIX001_PV",
        "430TH01LT026_PV",
        "430TH01XS004A_PV",
        "430TH01XS004B_PV",
        "830PP20FIC001/PID_SP",
    )


class MockAction(Action):
    channels = ("430PP41FIC006/PID_CV_MAN", "430PP42FIC006/PID_CV_MAN")
    dtype = "continuous"
    semantic_channels = ("Action1", "Action2")
    legal_values = ((1, 10), (1, 10))


class MockDiscreteAction(Action):
    channels = ("channel1", "channel2")
    dtype = "discrete"
    legal_values = ((1, 10), (1, 10))


class MockSingleChannelAction(Action):
    channels = ("move",)
    dtype = "continuous"
    legal_values = ((1, 10),)


class MockTerminal(TerminalStates):
    def terminal1(self, state):
        if state["430PP42FIC006/PID_PV"] == 102:
            return True
        return False


class MockPlant(Plant[MockState, MockAction]):
    state_type = MockState
    action_type = MockAction
    terminal_states = MockTerminal
    meta_keys = ("meta1", "meta2", "met/a3")

    def __init__(self, with_meta=False, **kwargs):
        super().__init__(**kwargs)
        self.i = 0
        state = [
            101,
            1.337,
            22,
            34.6,
            67,
            0.1,
            1587.7,
            97,
            1.0111,
            1.1,
            34.3,
            84.25,
            1.9,
        ]
        state2 = [
            1,
            2.447,
            21,
            100,
            76,
            0.25,
            300.7,
            44,
            -1.0111,
            1.12,
            77.3,
            10.25,
            1.99999,
        ]
        if with_meta:
            state = {key: state[i] for i, key in enumerate(MockState.channels())}
            state2 = {key: state2[i] for i, key in enumerate(MockState.channels())}
            meta1 = OrderedDict({"meta1": 1, "meta2": 123, "met/a3": "Start"})
            meta2 = OrderedDict(
                {
                    "meta1": 2,
                    "meta2": 456,
                    "met/a3": "Thisisastringthatislongerthan10characters",
                }
            )
            self.metas = [meta1, meta2]
        else:
            state = np.array(state)
            state2 = np.array(state2)
            self.metas = [OrderedDict(), OrderedDict()]
        self.states = [state, state2]

    def _get_next_state(self, state: MockState, action: MockAction) -> MockState:
        # this might be us waiting on network communication or some hardware
        time.sleep(0.1)
        self.i += 1
        state = self.state_type(self.states[self.i % 2], meta=self.metas[self.i % 2])
        state.cost = 1
        state.terminal = self.terminal_states.determine_if_terminal(state)
        self._current_state = state
        return state

    def check_initial_state(self, state: MockState) -> MockState:
        self._current_state = self.state_type(self.states[0])
        self._current_state.cost = 1
        self._current_state.terminal = False
        self._current_state.meta = self.metas[0]
        return self._current_state

    def notify_episode_stops(self) -> bool:
        pass
