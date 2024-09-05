# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

from functools import partial

import pytest
from numpy.testing import assert_almost_equal

from psipy.core.utils import busy_sleep
from psipy.rl.core.cycle_manager import CM, Timer, setfield

callback = partial(setfield, {}, "Test")


@pytest.fixture
def timer():
    return Timer(callback)


@pytest.fixture
def cm():
    """Manages the global cycle manager to work in tests in atomic fashion.

    Ensures it is properly started before and cleanup after the test. This only
    works in non-concurrent tests tho!
    """
    CM.setup(10000)
    CM.notify_episode_starts(0, "testplant", "testcont")
    yield CM
    CM.notify_episode_stops()


class TestCycleManager:
    @staticmethod
    def test_create_new_timer(cm):
        cm["test"].tick()
        assert ["test"] == cm.available_timers

    @staticmethod
    def test_get_timer(cm):
        cm["test"].tick()  # create
        cm["test"].tock()  # retrieve
        assert isinstance(cm["test"].time, float)
        assert ["test"] == list(cm._stats.keys())
        assert len(cm._stats["test"]) == 2

    @staticmethod
    def test_singleton_reset(cm):
        cm["test"].tick()
        with pytest.raises(ValueError):
            cm["test"].tick()
        cm.notify_episode_stops()
        cm["test"].tick()


class TestTimer:
    @staticmethod
    def test_tick_tock_once(timer):
        timer.tick()
        busy_sleep(0.01)
        timer.tock()
        assert_almost_equal(timer.time, 0.01, decimal=2)

    @staticmethod
    def test_tick_tock_twice(timer):
        for _ in range(2):
            timer.tick()
            busy_sleep(0.01)
            timer.tock()
        assert_almost_equal(timer.time, 0.01, decimal=2)

    @staticmethod
    def test_context_manager(timer):
        with timer:
            busy_sleep(0.01)
        assert_almost_equal(timer.time, 0.01, decimal=2)

    @staticmethod
    def test_ticktock(timer):
        timer.ticktock()
        busy_sleep(0.01)
        timer.ticktock()
        busy_sleep(0.01)
        timer.ticktock()
        assert_almost_equal(timer.time, 0.01, decimal=2)
