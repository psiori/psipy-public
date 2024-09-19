# PSIORI PACT
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Script to control the cartpole hardware with a joystick."""

import numpy as np
import pygame

from psipy.rl import Loop
from psipy.rl.control import Controller

from cartpole_control.plant.cartpole_plant import (
    SwingupContinuousAction,
    SwingupPlant,
    SwingupState,
)


class Joystick(Controller):
    def __init__(self):
        super().__init__(
            state_channels=SwingupState.channels(), action=SwingupContinuousAction,
        )
        pygame.init()
        pygame.joystick.init()
        self.stick = pygame.joystick.Joystick(0)
        self.stick.init()

    def notify_episode_starts(self) -> None:
        pass

    def notify_episode_stops(self) -> None:
        pass

    def _get_action(self, state) -> np.ndarray:
        pygame.event.get()
        speed = int(self.stick.get_axis(0) * 1000)
        print(f"STICK SPEED: {speed}")
        return np.array([speed])


if __name__ == "__main__":
    loop = Loop(
        SwingupPlant(
            "pact-one.localdomain",
            5555,
            5556,
            speed_values=[200],
            angle_terminals=False,
            continuous=True,
        ),
        Joystick(),
        "Joystick",
        "Joystick-SART",
    )

    loop.run(episodes=-1, max_episode_steps=-1)
