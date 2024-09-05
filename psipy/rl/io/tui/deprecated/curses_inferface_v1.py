# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Terminal User Interface for Reinforcement Learning projects.

The TUI receives via zmq information from the RL :class:`Loop` and displays
it in a readable format. TUIs will display the current plant and
controller, state, action, meta, and cycle times. Optionally, the TUI can
display information in #TODO another tab, internally called the "Custom Tab",
because the information displayed here is dependent on the customer. This
information can take any format, for example bar graphs or array representations.
All that needs to be done is implement a create of :class:`Tab` and implement its
:meth:`print`.

Information is sent via the Cycle Manager's (:class:`CycleManager`) :meth:`step` and
#TODO :meth:`step_custom` methods.  Step is called once and only once during each step
in the loop, while :meth:`step_custom` can be called multiple times within one loop.
All information sent over the custom step is conglomerated into one message that is
then printed in the custom tab.

#TODO: Errors in the TUI close the TUI gracefully and print the stacktrace to the
#      terminal.
Errors within the loop open a popup with error and stacktrace within the TUI, and must
be acknowledged by the user by hitting `Enter`. The loop can continue to run while this
error window is open. For an example error, hit `d` in the TUI. A generic error will
appear.

The TUI is implemented in curses.
"""

import curses
import logging
import traceback
from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple, cast

import numpy as np
import zmq

from psipy.core.io import json_decode
from psipy.core.threading_utils import StoppableThread

__all__ = ["TerminalInterface"]
__version__ = "4.0"

LOG = logging.getLogger(__name__)


# To access internal variables, load global DEBUG_LOG into function
# and set it to str(var). This will be printed at the bottom of the screen.
DEBUG_LOG = ""
_number_dtypes = (
    np.generic,
    float,
    int,
)
# Global heights
NUM_LOGS = 5
HEIGHT_LOGS = NUM_LOGS + 2
HEIGHT_HEADER = 5
HEIGHT_FOOTER = 3


def format_date(date):
    return date.strftime("%H:%M:%S %Y-%m-%d")


def format_timer(ts: float) -> str:
    if ts < 100:
        return f"{ts:.2f}s"
    ds = datetime.fromtimestamp(ts)
    if ts < 3600:
        return ds.strftime("%M:%S.%f")[:-3]
    return ds.strftime("%H:%M:%S.%f")[:-3]


def _add_subwindow_title(window, y, title):
    """Adds title over border of subwindow"""
    window.border()
    window.addstr(0, y // 2 - len(title) // 2, title, curses.A_STANDOUT)


def _print_aligned_dict(window, dict_obj: Dict, start_i: int = 0):
    """Prints a dict where the colons are aligned"""
    # Find the longest key
    max_len = max([len(key) for key in dict_obj])
    max_num_len = 8  # e.g. 1234.567

    # Only do decimal alignment if all values are numbers
    align_decimal = all([isinstance(val, _number_dtypes) for val in dict_obj.values()])

    space_colon = " " * 2  # to increase spacing between colons if desired
    for i, (key, value) in enumerate(dict_obj.items()):
        i += start_i
        extra_space = " " * (max_len - len(key))
        decimal_space = ""
        if isinstance(value, (float, int)):
            if isinstance(value, int):
                value = str(value)
            else:
                value = f"{value:3.3f}"
            if align_decimal:
                len_num = len(value)
                decimal_space = " " * (max_num_len - len_num)

        window.addstr(
            i + 1, 1, f"{key}{extra_space}:{space_colon}{decimal_space}{value}"
        )


def print_header(
    header, plant: str, controller: str, ports: Tuple[int, int], start_time: datetime
):
    """Prints header centered around '|' """
    _, max_y = header.getmaxyx()
    header.border()
    pin, pout = ports
    header.addstr(0, 2, f" PSIORI Loop Monitor v{__version__} | i/o {pin}/{pout} ")

    header_string = f" Plant: {plant} | Controller: {controller} "
    time_string = (
        f"Start Time: {format_date(start_time)} "
        f"| Current Time: {format_date(datetime.now())}"
    )
    elapsed_string = f"Elapsed: {str(datetime.now() - start_time)}"

    midpoint = header_string.find("|") + 1  # +1 to shift to proper center
    midpoint_time = time_string.find("|") + 1
    midpoint_elapsed = (len(elapsed_string) // 2) + 1

    header.addstr(1, 1 + max_y // 2 - midpoint, header_string, curses.A_BOLD)
    header.addstr(2, 1 + max_y // 2 - midpoint_time, time_string)
    header.addstr(3, 1 + max_y // 2 - midpoint_elapsed, elapsed_string)
    header.noutrefresh()


def print_status_window(status_win, state: Dict, actions: Dict, color: bool):
    window_title = " Status "
    if color:
        status_win.bkgd(" ", curses.color_pair(3))
    x, y = status_win.getmaxyx()

    _add_subwindow_title(status_win, y, window_title)
    try:
        # Print the "values"
        status_win.addstr(1, y // 2 - len("State") // 2, "State", curses.A_UNDERLINE)
        status_win.addstr(2, 1, "Values", curses.A_DIM)
        _print_aligned_dict(status_win, state["values"], 2)

        # Print the terminal and reward
        cur_y = len(state["values"]) + 2
        state = {
            key: value
            for (key, value) in state.items()
            if key != "values"
            if key != "meta"
        }
        status_win.addstr(cur_y + 2, 1, "Returns", curses.A_DIM)
        _print_aligned_dict(status_win, state, start_i=cur_y + 2)

        # Print the actions
        cur_y += len(state) + 2
        status_win.addstr(
            cur_y + 3, y // 2 - len("Action") // 2, "Action", curses.A_UNDERLINE
        )
        _print_aligned_dict(status_win, actions, start_i=cur_y + 3)
    except curses.error:
        status_win.erase()
        _add_subwindow_title(status_win, y, window_title)
        status_win.addstr(1, 1, "Please enlarge the window to see all info.")
    except Exception as e:  # noqa F841
        status_win.addstr(2, 1, "Awaiting data...")
    status_win.noutrefresh()


def print_meta_window(meta_win, meta: Dict, color: bool):
    window_title = " Meta Information "
    if color:
        meta_win.bkgd(" ", curses.color_pair(1))
    x, y = meta_win.getmaxyx()

    _add_subwindow_title(meta_win, y, window_title)
    try:
        _print_aligned_dict(meta_win, meta)
    except curses.error:
        meta_win.erase()
        _add_subwindow_title(meta_win, y, window_title)
        meta_win.addstr(1, 1, "Please enlarge the window to see all info.")
    except Exception as e:  # noqa F841
        pass
    meta_win.noutrefresh()


def print_cycle_window(cycle_win, cycle: Dict[str, Tuple[float, float]], color: bool):
    window_title = " Cycle Manager Info "
    if color:
        cycle_win.bkgd(" ", curses.color_pair(2))
    x, y = cycle_win.getmaxyx()

    _add_subwindow_title(cycle_win, y, window_title)
    # Create dictionary to be displayed
    cycle_dict = dict()
    try:
        for name, timer in cycle.items():
            avg, std = timer
            avg *= 1000
            std *= 1000
            cycle_dict[name] = f"{avg:7.2f}ms ±{std:5.2f}"
        _print_aligned_dict(cycle_win, cycle_dict)
    except curses.error:
        cycle_win.erase()
        _add_subwindow_title(cycle_win, y, window_title)
        cycle_win.addstr(1, 1, "Please enlarge the window to see all info.")
    except Exception as e:
        LOG.error(f"Error while printing cycles: {e}")
    cycle_win.noutrefresh()


def print_logs(log_win, logs: List[Dict[str, str]]):
    window_title = " Logs "
    x, y = log_win.getmaxyx()

    _add_subwindow_title(log_win, y, window_title)
    try:
        for i, log in enumerate(reversed(logs)):
            log_win.addstr(i + 1, 1, log["msg"])
    except curses.error:
        # log_win.erase()
        # _add_subwindow_title(log_win, y, window_title)
        # log_win.addstr(1, 1, "Please enlarge the window to see all info.")
        pass
    except Exception as e:  # noqa F841
        pass
    log_win.noutrefresh()


def print_footer(footer):
    _, max_y = footer.getmaxyx()
    debug = "" if DEBUG_LOG == "" else f" | DEBUG LOG: {DEBUG_LOG}"
    footer_string = f"q: quit tui | s: stop episode | e: exit loop {debug}"
    footer_string += " " * (max_y - len(footer_string))
    footer.addstr(1, 1, footer_string, curses.A_STANDOUT)
    footer.noutrefresh()


def print_error_window(window, error: Exception):
    x, y = window.getmaxyx()
    window.addstr(1, y // 2 - 3, "Error", curses.A_BLINK)
    try:
        error_lines = str(error).split("\n")
        formatted_error = ""
        for error_line in error_lines:
            spacing = " " * 2
            formatted_error += spacing + error_line + "\n"
        window.addstr(2, 2, formatted_error, curses.A_BOLD)
    except curses.error:
        pass
    window.border()


def create_windows(max_x: int, max_y: int, meta_height: int = 10):
    """Create the windows for each cli section.
    Window parameters are: (nlines, ncols, begin_y, begin_x)
    """
    meta_height += 4  # Include episode and step plus border
    log_height = NUM_LOGS + 2  # To include borders
    header = curses.newwin(HEIGHT_HEADER, max_y, 0, 0)
    status_win = curses.newwin(
        max_x - (HEIGHT_HEADER + HEIGHT_FOOTER + log_height),
        max_y // 2,
        HEIGHT_HEADER,
        0,
    )
    meta_win = curses.newwin(meta_height, max_y // 2, HEIGHT_HEADER, max_y // 2)
    cycle_win = curses.newwin(
        max_x - (HEIGHT_FOOTER + meta_height + HEIGHT_HEADER + log_height),
        max_y // 2,
        meta_height + HEIGHT_HEADER,
        max_y // 2,
    )
    logs = curses.newwin(log_height, max_y, max_x - (HEIGHT_FOOTER + log_height), 0)
    footer = curses.newwin(HEIGHT_FOOTER, max_y, max_x - HEIGHT_FOOTER, 0)
    return header, status_win, meta_win, cycle_win, logs, footer


def terminal_screen(
    screen,
    plant: str,
    controller: str,
    ports: Tuple[int, int],
    stats: Dict[str, Tuple[float, float]],
    state: Dict,
    actions: Dict,
    meta: Dict,
    logs: List[Dict[str, str]],
    start_time: datetime,
    color=False,
):
    """Paints all sub windows to the screen."""
    max_x, max_y = screen.getmaxyx()
    try:
        head_win, status_win, meta_win, cycle_win, log_win, foot_win = create_windows(
            max_x, max_y, len(meta)
        )
        print_status_window(status_win, state, actions, color)
        print_meta_window(meta_win, meta, color)
        print_cycle_window(cycle_win, stats, color)
        print_header(head_win, plant, controller, ports, start_time)
        print_logs(log_win, logs)
        print_footer(foot_win)
        curses.doupdate()
    except curses.error:
        screen.erase()
        screen.addstr(0, 0, "Please enlarge window to see all info.")
        screen.noutrefresh()
        curses.doupdate()


def error_screen(
    screen,
    plant: str,
    controller: str,
    ports,
    error: Exception,
    start_time: datetime,
    color=False,
):
    """Paints an error message over the screen"""
    max_x, max_y = screen.getmaxyx()
    try:
        head_win, status_win, meta_win, cycle_win, log_win, foot_win = create_windows(
            max_x, max_y
        )
        print_header(head_win, plant, controller, ports, start_time)
        print_footer(foot_win)

        # Create error window
        err_win = curses.newwin(
            max_x - (HEIGHT_HEADER + HEIGHT_FOOTER), max_y, HEIGHT_HEADER, 0
        )
        if color:
            err_win.bkgd(" ", curses.color_pair(4))
        print_error_window(err_win, error)

        status_win.noutrefresh()
        meta_win.noutrefresh()
        cycle_win.noutrefresh()
        err_win.noutrefresh()

        curses.doupdate()
    except curses.error:
        screen.erase()
        screen.addstr(0, 0, "Please enlarge window to see all info.")
        screen.noutrefresh()
        curses.doupdate()


class TUISubscriber(StoppableThread):
    """The subscriber manages incoming messages in its own thread.

    Keeps track of the latest message per topic, making them available to the
    terminal interface loop. It may happen that individual messages are skipped
    if the terminal interface loop does not read them from this Subscriber
    fast enough. For messages on the ``step`` channel this is on purpose (as the
    primary control loop may run faster than the terminal interface), for other
    topics (like ``exception`` or ``lifecycle``) the terminal interface has to
    be able to degrade accordingly.

    Also it is to note, that messages can only be read once! Once a message is
    read from the subscriber, it is deleted. This is implemented in order to not
    have an event happen twice in quick succession.
    """

    #: Latest conflated messages for individual topics.
    _data: Dict[str, str]

    #: Message buffers for individual topics.
    _buffers: Dict[str, List[Dict[str, str]]]

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port

        self._data = dict()
        self._buffers = dict(logs=[])

        self.zmq = zmq.Context()
        self.sub = self.zmq.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
        self.sub.setsockopt(zmq.RCVTIMEO, 1000)  # wait 1 sec at most per iter
        self.sub.connect(f"tcp://localhost:{port}")

    def __del__(self):
        self.stop()
        self.sub.close()
        self.zmq.destroy()

    def __getitem__(self, topic: str) -> Optional[Dict[str, Any]]:
        if topic == "logs":
            if len(self._buffers[topic]) == 0:
                return None
            return self._buffers[topic].pop()
        msg = self._data.pop(topic, "null")  # json_decode("null") == None
        return cast(Optional[Dict[str, Any]], json_decode(msg))

    def run(self):
        while not self.stopped():
            try:
                temp = self.sub.recv_multipart()
            except zmq.ZMQError:
                continue
            try:
                topic, msg = temp
            except ValueError:  # workaround for handling messed up messages
                continue
            topic = topic.decode()
            if topic.startswith("logs."):
                topic, level = topic.split(".", maxsplit=1)
                self._buffers[topic].append(dict(level=level, msg=msg.decode()))
            else:
                self._data[topic] = msg.decode()


class TerminalInterface:
    """CLI interface for monitoring Plant<->Controller interaction.
    If for some reason your terminal becomes messed up (shouldn't happen),
    run the command ``reset``!
    """

    def __init__(
        self, port_sub: int, port_pub: int, update_ms: int = 1, use_color: bool = False
    ):
        self.start_time = datetime.now()
        self.is_colored = use_color
        self.stopped = False
        self.update_ms = min(1, update_ms)

        self.last_data: Optional[Dict[str, Any]] = None
        self.log_buffer: Deque[Dict[str, str]] = deque(maxlen=10)

        # Received from loop on episode start.
        self.plant_name = "N.A."
        self.controller_name = "N.A."
        self.episode_number = 0
        self.ports = (port_sub, port_pub)

        # zmq setup
        LOG.info("Connect zmq...")
        self.zmq = zmq.Context()
        self.pub = self.zmq.socket(zmq.PUB)
        self.pub.connect(f"tcp://localhost:{port_pub}")
        self.sub = TUISubscriber(port_sub)
        self.sub.start()
        LOG.info(f"Connected zmq to ports pub {port_pub} and sub {port_sub}.")

    def start(self):
        self.screen = curses.initscr()

        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_WHITE)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_RED)

        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.screen.nodelay(True)
        self.screen.keypad(1)

        self.screen.erase()
        self.screen.addstr(0, 0, f"TUI ready, awaiting loop on port {self.ports[0]}..")
        self.screen.noutrefresh()
        curses.doupdate()

        while self.print():
            curses.napms(self.update_ms)

        self.stop()

    def print(self) -> bool:
        """Update curses screen, returning whether to continue or exit."""
        try:
            user_input = self.screen.getch()

            # Stop current episode, maybe start next
            if user_input == ord("s"):
                self.pub.send_multipart([b"cmd", b"episode_stop"])
                return True

            # Exit loop
            if user_input == ord("e"):
                self.pub.send_multipart([b"cmd", b"exit"])
                return True

            # Quit terminal ui
            if user_input == ord("q"):
                return False

            # Allow forcing a 'debug exception' to test error modes
            if user_input == ord("D"):
                raise RuntimeError("Forced debug exception!")

            data = self.sub["exception"]
            if data is not None:
                raise ValueError(f"Exception {data['exception']} occurred in loop.")

            data = self.sub["lifecycle"]
            if data is not None:
                event = data.pop("event")
                if event == "episode_starts":
                    self.plant_name = data["plant"]
                    self.controller_name = data["control"]
                    self.episode_number = data["episode"]
                    return True
                if event == "episode_stop":
                    return True
                if event == "exit":
                    return True

            data = self.sub["step"]
            if data is not None:
                self.last_data = data

            # Collect oldest log message available. Should easily empty the
            # log message buffer from the subscriber as the tui refreshes so
            # frequently.
            log_msg = self.sub["logs"]
            if log_msg is not None:
                self.log_buffer.append(log_msg)

            # If no data was received, do not refresh screen.
            if (data is None and log_msg is None) or self.last_data is None:
                return True

            data = self.last_data.copy()
            data["state"] = data["state"].copy()
            meta = data["state"].pop("meta")
            meta["Episode"] = self.episode_number
            meta["Step"] = data["step"]
            meta["Interaction Time"] = format_timer(data["interaction_time"])

            terminal_screen(
                self.screen,
                self.plant_name,
                self.controller_name,
                self.ports,
                self.last_data["stats"],
                self.last_data["state"],
                self.last_data["action"],
                meta,
                list(self.log_buffer),
                self.start_time,
                self.is_colored,
            )
        except Exception as e:  # noqa F841
            if data is not None and "traceback" in data:
                trace = data["traceback"]
            else:
                trace = traceback.format_exc()
            error_screen(
                self.screen,
                self.plant_name,
                self.controller_name,
                self.ports,
                trace,
                self.start_time,
                self.is_colored,
            )
        return True

    def __del__(self):
        if not self.stopped:
            self.stop()

    def stop(self):
        self.zmq.destroy()
        self.sub.stop()
        self.stopped = True
        curses.nocbreak()
        curses.curs_set(1)
        self.screen.nodelay(False)
        self.screen.keypad(0)
        curses.echo()
        curses.endwin()
        return self.stopped


def run(port, update_ms: int = 10):  # TODO: Update ms
    tui = TerminalInterface(port, update_ms)
    tui.start()


def main():
    parser = ArgumentParser(description="PSIORI RL Toolbox Terminal UI")
    parser.add_argument(
        "port",
        type=int,
        default=5556,
        help="I/O port to subscribe to, will publish at port+1.",
    )
    args = parser.parse_args()
    port_sub = args.port
    port_pub = port_sub + 1
    run(port_sub, port_pub)


if __name__ == "__main__":
    main()
