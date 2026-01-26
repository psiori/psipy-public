# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""The Loop is the main connection between the plant and the controller.

The loop operates in episodes, starting and stopping all components cleanly
between episodes and on process shutdown. SART information is recorded, and
progress through the episode can be visualized in a separate CLI process.

The loop takes a plant and a controller. The controller acts on the state
coming from the plant. A controller within itself can be made up of individual
controllers, each acting on its respective action channels. The produced action
is passed to the plant by the controller, resulting in a changed state in the
next cycle. This repeats until a terminal state occurs or max_episode_steps are
reached.

"""

import logging.handlers
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from psipy.rl.core.controller import Controller
from psipy.rl.core.cycle_manager import CM
from psipy.rl.core.exceptions import NotNotifiedOfEpisodeStart
from psipy.rl.core.plant import Action, Plant, State, TAction, TState
from psipy.rl.io.sart import SARTLogger

__all__ = ["Loop", "LoopPrettyPrinter"]

LOG = logging.getLogger(__name__)


class LoopPrettyPrinter:
    def __init__(
        self,
        costfunc: Optional[Callable] = None,
        state_channels: Optional[Tuple[str, ...]] = None,
        action_channels: Optional[Tuple[str, ...]] = None,
    ):
        self._costfunc = costfunc
        self._state_channels = state_channels
        self._action_channels = action_channels
        self.total_cost: float = 0.0
        self.total_transitions: int = 0

    def print_transition(
        self,
        state: TState,
        action: TAction,
        next_state: TState,
        cost: float,
    ) -> None:
        state = state.as_array(self._state_channels)
        next_state = next_state.as_array(self._state_channels)
        if self._costfunc is not None:
            cost = self._costfunc(next_state[None, :].copy()).tolist()[0]
        with np.printoptions(precision=2, suppress=True, floatmode="fixed"):
            if self._action_channels is not None:
                action = action.as_array(self._action_channels)
            else:
                action = action.as_array()
            LOG.info(f"{CM.cycle:3d}: {state} + {action} -> {next_state} = {cost:.04f}")
        self.total_cost += cost
        self.total_transitions += 1

    def print_total_cost(self) -> None:
        LOG.info(f"Total cost: {self.total_cost}")
        self.total_cost = 0
        self.total_transitions = 0


class Loop:
    """Uses the given controller to loop through episodes of the given plant."""

    def __init__(
        self,
        plant: Plant,
        control: Controller,
        name: str = "DefaultName",
        logdir: str = "sart-logs",
        render: bool = False,
        cm_port: int = 5556,
        cm_log_level: str = "INFO",
        sart_rollover: Optional[str] = None,
        initial_time: Optional[datetime] = None,
        single_sart: bool = False,
    ):
        # try:
        #     if not CM.is_setup():
        #         CM.setup(cm_port, cm_log_level)
        # except Exception as e:
        #     print("CM failed with exception", e)
        self.plant = plant
        self.control = control
        self.name = name
        self.render = render
        self.logdir = logdir
        self.initial_time = initial_time
        self.sart = SARTLogger(
            self.logdir,
            self.name,
            self.initial_time,
            sart_rollover,
            single_sart,
        )
        self.single_sart = single_sart

        # Dictionary of episode statistics as they are run.
        # Contains the total cost and cycles taken per episode, indexed
        # by episode number. Remember when running single episodes to pass
        # in the proper episode number, or else the first episode's metrics
        # will constantly be overwritten.
        self.metrics: Dict[int, Dict[str, Any]] = dict()

    def __str__(self):
        return f"Loop({repr(self.control)} controlling {repr(self.plant)})"

    def __del__(self):
        CM.reset()

    def shutdown(self):
        """Cleanup things which (maybe) were not cleanup in run_episode's finally."""
        self.sart.shutdown()
        CM.reset()

    def run(
        self,
        episodes: int = 1,
        max_episode_steps: int = -1,
        max_writer_steps: Optional[int] = None,
        step_callback: Optional[
            Callable[[int, State, Action, State, float], None]
        ] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Runs the loop for "episodes" many episodes

        By passing a negative number for episodes, the loop runs forever.

        Args:
            episodes: Number of episodes to run (-1 for infinite).
            max_episode_steps: Maximum steps per episode.
            max_writer_steps: Steps before SART rollover.
            step_callback: Optional callback(step, state, action, next_state, cost) called each step.

        Returns:
            Metrics of the completed episodes
        """

        episodes = sys.maxsize - 1 if episodes < 0 else episodes
        for episode in range(1, episodes + 1):
            break_all = self.run_episode(
                episode,
                max_steps=max_episode_steps,
                max_writer_steps=max_writer_steps,
                step_callback=step_callback,
            )
            if break_all:
                break
        return self.metrics

    def update_actor(self, actor: Controller) -> None:
        self.control = actor

    def run_episode(
        self,
        episode_number: int = 1,
        max_steps: int = -1,
        max_writer_steps: Optional[int] = None,
        initial_state: Optional[State] = None,
        pretty_printer: Optional[LoopPrettyPrinter] = None,
        step_callback: Optional[
            Callable[[int, State, Action, State, float], None]
        ] = None,
    ) -> bool:
        """Runs a single episode.

        Args:
            episode_number: Episode number.
            max_steps: Maximum steps per episode (-1 for unlimited).
            max_writer_steps: Steps before SART rollover.
            initial_state: Optional initial state.
            pretty_printer: Optional pretty printer for transitions.
            step_callback: Optional callback(step, state, action, next_state, cost) called each step.
        """

        self.trajectory: List[State] = []

        LOG.info(f"Loop starts, episode {episode_number}...")
        step: int = 0
        total_cost: Union[int, float] = 0
        start_time: float = time.time()
        try:
            CM.notify_episode_starts(
                episode_number,
                self.plant.__class__.__name__,
                self.control.__class__.__name__,
            )

            self.plant.notify_episode_starts()
            self.control.notify_episode_starts()
            self.sart.notify_episode_starts()

            # Forces an initial state if one is provided.
            state = self.plant.check_initial_state(initial_state)
            # self.trajectory.append(state) # TODO(SL): this was missing. Please check and leave a comment if it was intentional.

            while True:
                step += 1
                CM["loop"].tick()
                self.plant.cycle_started()

                with CM["get-action"]:
                    action = self.control.get_action(state)
                with CM["get-state"]:
                    next_state = self.plant.get_next_state(state, action)
                    cost = self.plant.get_cost(next_state)
                    terminal = self.plant.is_terminal(next_state)
                self.trajectory.append(next_state)
                action_dict = action.as_dict(with_additional=True)
                data = dict(state=state.as_dict(), action=action_dict)
                CM.step(data)  # Increments step counter.
                with CM["sart-append"]:
                    self.sart.append(data)

                # Call cycle callback if provided
                if step_callback is not None:
                    step_callback(step, state, action, next_state, cost)

                # TODO: Remove all plant side costs?
                if pretty_printer is not None:
                    pretty_printer.print_transition(state, action, next_state, cost)
                state = next_state
                total_cost += cost

                if self.render:
                    with CM["render"]:
                        self.plant.render()

                if terminal:
                    LOG.info(f"Total cost: {total_cost}")
                    data = dict(
                        state=next_state.as_dict(),
                        action=OrderedDict({k: np.nan for k in action.channels}),
                    )
                    self.sart.append(data)
                    CM.step(data, increment_step_counter=False)
                    break

                # User initiated episode stop / loop exit or max_steps reached.
                if CM.should_stop(max_steps=max_steps):
                    # BUG? the last observation has not been added to the SART, yet! Thus, the last transition will never be used for learning??
                    break

                if (
                    max_writer_steps is not None
                    and step > 0
                    and step % max_writer_steps == 0
                ):
                    LOG.info(
                        f"Max writer steps reached. Rolling over SART file within episode after {step} steps..."
                    )
                    print(
                        f"Max writer steps reached. Rolling over SART file within episode after {step} steps..."
                    )
                    self.sart.do_rollover()

                CM["loop"].tock()
            LOG.info("Loop finished...")
        except KeyboardInterrupt:
            LOG.info("Interrupt received.")
            return True  # exit all episodes, if many
        except Exception as e:
            CM.handle_exception(e)
            raise e  # Loop may crash, crashes should be handled one level up.
        finally:  # Cleanup local resources.
            # print("\n>>>> START OF TRAJECTORY:")
            # pprint(self.trajectory)
            # print("<<<< END OF TRAJECTORY")
            # print("Trajectory length: ", len(self.trajectory))

            if pretty_printer is not None:
                pretty_printer.print_total_cost()
            try:
                self.plant.notify_episode_stops()
                self.control.notify_episode_stops()
                self.sart.notify_episode_stops()
                CM.notify_episode_stops()
            except NotNotifiedOfEpisodeStart:
                pass
            self.metrics[episode_number] = {
                "total_cost": total_cost,
                "cycles_run": step,
                "wall_time_s": round(time.time() - start_time, 4),
            }

        return False  # exit only current episode
