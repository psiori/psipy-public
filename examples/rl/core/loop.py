# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Example script of how a basic RL Loop works.

The Loop is the main component that ties plants, controllers,
states, and actions together.  Running this script with the TUI
will show how the controller sends random actions from the
:class:`ContinuousRandomActionController` and states from the
:class:`MockPlant` are received back.
"""

from psipy.rl.control import ContinuousRandomActionController
from psipy.rl.loop import Loop
from psipy.rl.plant.tests.mocks import MockAction, MockPlant, MockState


def run(name: str):
    controller = ContinuousRandomActionController(MockState.channels(), MockAction)
    plant = MockPlant(with_meta=True)
    loop = Loop(plant, controller, name, "ExampleLoop")
    loop.run(10, max_episode_steps=-1)


if __name__ == "__main__":
    run("TestLoop")
