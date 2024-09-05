# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Terminal User Interface for Reinforcement Learning projects.

The TUI receives via zmq information from the RL :class:`Loop` and displays
it in a readable format. TUIs will always display the current plant and
controller, state, action, meta, and cycle times. Optionally, the TUI can
display information in another tab, internally called the "Custom Tab",
because the information displayed here is dependent on the customer. This
information can take theoretically take any format, but is limited to expertise
with the ``curses`` library. All that needs to be done is implement a create of
:class:`Tab`, any :class:`Window`s that are needed, and implement their
:meth:`_print`.

Information is sent via the Cycle Manager's (:class:`CycleManager`) :meth:`step` and
:meth:`step_custom` methods.  Step is called once and only once during each step
in the loop, while :meth:`step_custom` can be called multiple times within one loop.
All information sent over the custom step is conglomerated into one message that is
then printed in the custom tab.

The TUI looks different depending on whether it is opened on Windows or POSIX systems.
POSIX systems have a warning log window, information about the plant such as the SART
and meta, and then another log window below that is relatively sized to fill all
remaining space. If there is no space, this window is not seen. On Windows, this
relative sizing does not work. Therefore, Windows forgoes this second log window in
favor of a log tab, which can be accessed to pressing ``l`` and fills the entire
screen.

Errors in the TUI close the TUI gracefully and print the stacktrace to the terminal.
Curses errors are caught, and the terminal displays a "..." to denote the current
display is throwing a curses error. Usually resizing the window solves this. To know
the exact error, pass the --debug flag, which will cause a crash upon curses errors.
Any stdout will be lost upon crashing. Errors within the loop open a popup with error
and stacktrace within the TUI, and must be acknowledged by the user by hitting `Enter`.
The loop can continue to run while this error window is open. For an example error,
hit `D` in the TUI. A generic error will appear.

Components used by the TUI are organized into functions, structures, tabs, and
windows. See the file ``tui/tui_interface.py`` for more information.

The TUI is implemented in curses. More information here:
https://docs.python.org/3/howto/curses.html
"""

import curses
import json
import logging
import platform
from argparse import ArgumentParser
from collections import deque
from copy import deepcopy
from curses import ascii
from datetime import datetime
from queue import Queue
from typing import Any, Deque, Dict, List, Optional, Tuple, cast

import zmq

from psipy.rl.io.tui import HEIGHT_FOOTER, HEIGHT_HEADER
from psipy.rl.io.tui.tui_comm import TUISubscriber
from psipy.rl.io.tui.tui_functions import format_timer, parse_user_input
from psipy.rl.io.tui.tui_interface import Tab
from psipy.rl.io.tui.tui_tabs import TAB_REGISTER, LogsTab, MainTab
from psipy.rl.io.tui.tui_windows import Footer, Header

__all__ = ["TerminalInterface"]

LOG = logging.getLogger(__name__)


class TerminalInterface:
    # TODO: Overlay disappears on resize.
    # TODO: Whaleback tab.
    # TODO: Start up screen no longer works but used to
    # TODO: CM window moves downward when hitting "m"
    """Terminal user interface for monitoring Plant<->Controller interaction.

    Args:
        port: Subscribe to ``port``, publish to ``port + 1``. This is the
              inverse of the :class:`~psipy.rl.cycle_manager.CycleManager`, as
              the two interact through these two ports in a pub/sub fashion.
        state_channels: Subset of channels to display of all received state channels
        state_channels: Subset of channels to display of all received action channels
        update_ms: How many ms to sleep before updating the screen again. Faster
              is more accurate but uses up more resources. Default is 100
              times per second.
        raises_curses_error: True when debugging; will raise curses errors. Otherwise,
                             the TUI will continue running but will display "..." if/
                             until the error is rectified.
    """

    def __init__(
        self,
        port: int,
        state_channels: Optional[List[str]] = None,
        action_channels: Optional[List[str]] = None,
        update_ms: int = 10,
        raises_curses_error: bool = False,
        exception_queue: Optional["Queue[Dict[str, Any]]"] = None,
    ):
        self.start_time = datetime.now()
        self.stopped = False
        self.update_ms = update_ms
        self.raises = raises_curses_error
        if exception_queue is None:
            exception_queue = Queue()
        self.exception_queue: "Queue[Dict[str, Any]]" = exception_queue

        self.last_data: Dict[str, Any] = {}
        self.prev_cust_data: Dict[str, Any] = dict()
        self.log_buffer: Deque[str] = deque(maxlen=200)

        self.typing = False
        self.typed = ""

        # Set TUI basic layout vars
        self.header = Header()
        self.footer = Footer()
        self.active_tab: Optional[Tab] = None
        self.custom_tab: Optional[Tab] = None
        self.custom_tab_set = False

        # zmq setup
        LOG.info("Connect zmq...")
        port_sub, port_pub = port, port + 1
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://localhost:{port_pub}")
        self.sub = TUISubscriber(port_sub)
        self.sub.start()
        self.ports = (port_sub, port_pub)
        LOG.info(f"Connected zmq to ports pub {port_pub} and sub {port_sub}.")

        # Received from loop on episode start.
        self.plant_name = "N.A."
        self.controller_name = "N.A."
        self.episode_number = 0

        # Set filtering variables
        self.state_channels = state_channels
        self.action_channels = action_channels

    def _check_enter_key_hit(self, user_input) -> bool:
        """Different systems represent 'ENTER' differently; check all here."""
        return user_input == curses.KEY_ENTER or user_input == 10 or user_input == 13

    def setup_curses(self) -> None:
        """Prepares the terminal to accept curses control."""
        # # Initialize colors
        # curses.start_color()
        # curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_RED)

        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.screen.nodelay(True)
        self.screen.keypad(True)

    def _get_custom_info(self) -> Optional[Dict[str, Any]]:
        """Gets and updates custom information as it comes in.

        Previous information sent over the custom topic is merged with the
        given information, with the more recent information overwriting old
        keys. In this way, :meth:`step_custom` of :class:`CycleManager` can
        be called multiple times per loop. This allows sending information
        from multiple areas of the loop.
        """
        latest_data = self.sub["custom"]
        cust_info = self.prev_cust_data
        if latest_data is not None:
            cust_info.update(latest_data)
            return cust_info
        return None

    def _set_custom_tab(self, classname: Optional[str] = None) -> None:
        """Set the custom tab class once if requested."""
        if classname is not None and not self.custom_tab_set:
            tab_class = TAB_REGISTER[classname]
            self.custom_tab = tab_class(self.screen)
            self.custom_tab_set = True

    def force_relative_dims(
        self, main_package: Dict[str, Any]
    ) -> Optional[Tuple[int, int]]:
        """Provides max height of each row for the main two rows.

        Depending on the length of SART and CM/Meta, the upper row which
        contains these windows will resize and potentially, if the terminal
        is large enough, leave some room below for the logs.

        Args:
            main_package: The main package dictionary being sent
                          to the TUI objects.

        Returns:
            (Height of top row, height of bottom row)
        """
        try:
            state = main_package["status"]["values"]
            cost = main_package["status"].get("cost", dict())
            terminal = main_package["status"].get("terminal", dict())
            returns = {"Cost": cost, "Terminal": terminal}
            action = main_package["status"]["action"]
            height_sart = len(state) + len(returns) + len(action) + 9

            height_meta = len(main_package["meta"])
            height_cycle = len(main_package["cycle"])

            y, x = self.screen.getmaxyx()
            y -= HEIGHT_FOOTER + HEIGHT_HEADER
            height_top_row = min(max(height_sart, height_meta + height_cycle), y)
            return height_top_row, y - height_top_row
        except KeyError:
            return None

    def set_active_tab(self, tab: Optional[Tab] = None) -> None:
        """Sets and creates the tab to be displayed between the header and footer."""
        if tab is not None:
            self.active_tab = tab
            self.active_tab.create(HEIGHT_HEADER, HEIGHT_FOOTER)

    def create(self) -> None:
        """Create the TUI object infrastructure on screen."""
        y, x = self.screen.getmaxyx()
        self.header.create(HEIGHT_HEADER, x, 0, 0)
        self.footer.create(HEIGHT_FOOTER, x, 0, y - HEIGHT_FOOTER)
        self.log_tab.create(HEIGHT_HEADER, HEIGHT_FOOTER)
        self.set_active_tab(self.active_tab)

    def start(self) -> None:
        """Initialize and start the TUI."""
        try:
            self.setup_curses()
            self.main_tab = MainTab(self.screen)
            self.log_tab = LogsTab(self.screen)
            self.active_tab = self.main_tab
            self.create()

            # Set startup text
            self.screen.erase()
            self.screen.addstr(
                0, 0, f"TUI ready, awaiting loop on port {self.ports[0]}..."
            )
            self.screen.noutrefresh()
            curses.doupdate()

            while self.print():
                curses.napms(self.update_ms)
        finally:  # Shutdown properly, even if an error occurred.
            self.stop()

    def print(self) -> bool:
        """Update curses screen, returning whether to continue or exit."""
        self.active_tab = cast(Tab, self.active_tab)
        try:
            user_input = self.screen.getch()
            error_package: Optional[Dict[str, Any]] = None

            # Leave typing mode.
            if user_input in (ascii.LF, ascii.CR):
                try:
                    topic, msg = parse_user_input(self.typed)
                except Exception:
                    pass
                else:
                    self.pub.send_multipart([topic.encode(), json.dumps(msg).encode()])
            if user_input in (ascii.ESC, ascii.LF, ascii.CR):
                self.typing = False

            # Handle input in typing mode.
            if self.typing:
                if user_input in (ascii.BS, ascii.DEL):
                    self.typed = self.typed[:-1]
                elif -1 < user_input < 255:
                    self.typed += chr(user_input)

            # Enter typing mode
            elif user_input == ord("i"):
                self.typed = ""  # start fresh
                self.typing = True

            # Stop current episode, maybe start next
            elif user_input == ord("s"):
                self.pub.send_multipart([b"cmd", b"episode_stop"])
                return True

            # Exit loop and TUI
            elif user_input == ord("q"):
                self.pub.send_multipart([b"cmd", b"exit"])
                return False

            # Exit TUI
            elif user_input == ord("e"):
                return False

            # Allow forcing a 'debug exception' to test error modes
            elif user_input == ord("D"):
                error_package = {
                    "exception": "User generated error.",
                    "traceback": "No traceback.",
                }

            # Switch to Main Tab
            elif user_input == ord("m") or user_input == ord("M"):
                self.set_active_tab(self.main_tab)

            # Switch to Second Tab
            elif user_input == ord("a") or user_input == ord("A"):
                if not self.custom_tab_set:
                    error_package = {
                        "exception": "CustomTabNotSetWarning",
                        "traceback": "This warning appears if "
                        "there is no custom tab for this plant.",
                    }
                else:
                    self.set_active_tab(self.custom_tab)

            # Switch to Logs Tab
            elif user_input == ord("l") or user_input == ord("L"):
                self.screen.erase()
                self.set_active_tab(self.log_tab)

            # Close any error popups if they exist
            if (
                self._check_enter_key_hit(user_input)
                and not self.active_tab.overlay.hidden
            ):
                self.active_tab.overlay.clear()
                self.screen.erase()

            # Terminal window was resized, throw everything away and recalculate sizes
            if user_input == curses.KEY_RESIZE:
                curses.resize_term(0, 0)
                prev_active = self.active_tab.__class__
                self.main_tab = MainTab(self.screen)
                if self.custom_tab is not None:
                    custom_class = self.custom_tab.__class__
                    self.custom_tab = custom_class(self.screen)  # type:ignore
                if prev_active == MainTab:
                    self.active_tab = self.main_tab
                elif prev_active == LogsTab:
                    self.active_tab = self.log_tab
                elif self.custom_tab is not None:
                    self.active_tab = self.custom_tab
                self.header = Header()
                self.footer = Footer()
                self.screen.erase()

            # Check for exceptions first, if none, continue
            data = self.sub["exception"]
            if not self.exception_queue.empty():
                data = self.exception_queue.get_nowait()
            if data is not None:
                error_package = {
                    "exception": data["exception"],
                    "traceback": data["traceback"],
                }

            # Check if turning the loop off was received from the CM
            data = self.sub["lifecycle"]
            if data is not None:
                event = data.pop("event")
                if event == "episode_starts":
                    self.start_time = datetime.now()
                if event == "episode_stop":
                    return True
                if event == "exit":
                    return True

            # Get log information
            log_entry = self.sub["logs"]
            if log_entry is not None:
                self.log_buffer.append(log_entry["msg"])

            # Extract data if it exists
            data = self.sub["step"]
            if data is not None:
                self.last_data = data

            data = deepcopy(self.last_data)

            # Set the header names
            controller = data.get("control", "-")
            plant = data.get("plant", "-")
            # Extract relevant data (from the copy)
            meta = data.get("state", {}).pop("meta", {})
            meta["episode"] = data.get("episode", 0)
            meta["interaction_time"] = format_timer(data.get("interaction_time", 0))
            status = data.get("state", dict(values={}))
            status["action"] = data.get("action", {})

            # Filter data if necessary
            if self.state_channels is not None:
                status["values"] = {
                    k: v
                    for k, v in status["values"].items()
                    if k in self.state_channels
                }
            if self.action_channels is not None:
                status["action"] = {
                    k: v
                    for k, v in status["action"].items()
                    if k in self.action_channels
                }

            # Create the data packages
            header_package = dict(
                plant=plant,
                controller=controller,
                start=self.start_time,
                ports=self.ports,
            )
            main_package = dict(
                status=status,
                meta=meta,
                cycle=data.get("stats", {}),
                logs=self.log_buffer,
            )
            logs_package = dict(logs=self.log_buffer)

            # Override the heights of the rows if on POSIX systems
            # This is necessary since Windows curses behaves differently
            # from POSIX curses. Apparently trying to force relative sizes
            # makes everything blow up, so here the format of the TUI is
            # changed to not have relative dimensions in the Windows case.
            if platform.system() != "Windows":
                # Below does not work on Windows!
                dims = self.force_relative_dims(main_package)
                if dims is not None:
                    top_row_y, bottom_row_y = dims
                    self.main_tab.directory["Columns1"].fixed_height = top_row_y
                    self.main_tab.directory["SARTWindow1"].fixed_height = top_row_y
                    self.main_tab.directory["CMWindow1"].fixed_height = top_row_y // 2
                    self.main_tab.directory["MetaWindow1"].fixed_height = top_row_y // 2
                    self.main_tab.directory["Columns1"].height = top_row_y
                    # Set the LogsTab to fill the empty space
                    unused = self.main_tab.empty_height
                    fixed_y = self.screen.getmaxyx()[0] - unused + HEIGHT_HEADER
                    self.log_tab.directory["LogWindow1"].fixed_y = fixed_y
                    bottom_row_y = unused - (HEIGHT_HEADER + HEIGHT_FOOTER)
                    self.log_tab.directory["LogWindow1"].fixed_height = bottom_row_y
            else:
                # If on windows, force the informational windows to
                # fill the entire screen and make the log window fill the whole tab
                y, _ = self.screen.getmaxyx()
                available_space = y - (HEIGHT_HEADER + HEIGHT_FOOTER + 5)
                self.main_tab.directory["Columns1"].fixed_height = available_space
                self.main_tab.directory["SARTWindow1"].fixed_height = available_space
                self.main_tab.directory["CMWindow1"].fixed_height = available_space // 2
                self.main_tab.directory["MetaWindow1"].fixed_height = (
                    available_space // 2
                )
                self.main_tab.directory["Columns1"].height = available_space
                self.log_tab.directory["LogWindow1"].fixed_y = HEIGHT_HEADER

            self.create()

            # Set the text
            self.header.print(header_package)
            self.log_tab.print(logs_package)
            self.active_tab.print(main_package)
            self.footer.print({"typing": self.typing, "typed": self.typed})

            # Extract custom info if it exists
            # Set custom tab, and print if it is active
            cdata = self._get_custom_info()
            if cdata is not None:
                self._set_custom_tab(cdata.get("tab"))
            if self.active_tab == self.custom_tab:
                if cdata is not None:
                    self.last_cust_data = cdata
                    cdata = self.last_cust_data.copy()
                    self.custom_tab.print(cdata["info"])

            # Display the overlay if triggered. This must come last,
            # otherwise it will be printed over and disappear.
            if error_package is not None:
                self.active_tab.overlay.create(0, 0, 0, 0)
                self.active_tab.set_error_package(error_package)

        except curses.error as ce:
            if self.raises:
                self.stop()
                raise ce
            self.screen.erase()
            self.screen.addstr(0, 0, "...")
            self.screen.noutrefresh()
        finally:
            curses.doupdate()
        return True

    def __del__(self):
        if not self.stopped:
            self.stop()

    def breakdown_curses(self) -> None:
        """Release terminal from curses control.

        If this fails and the terminal is all wonky, run `reset` in the terminal.
        """
        curses.nocbreak()
        curses.curs_set(1)
        self.screen.nodelay(False)
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()

    def stop(self) -> None:
        """Shut down all components gracefully."""
        self.ctx.destroy()
        self.sub.stop()
        self.stopped = True
        self.breakdown_curses()


def run(
    port,
    state_channels: Optional[List[str]] = None,
    action_channels: Optional[List[str]] = None,
    update_ms: int = 100,
    debug: bool = False,
) -> None:
    """Start the TUI."""
    tui = TerminalInterface(
        port,
        state_channels,
        action_channels,
        update_ms,
        raises_curses_error=debug,
    )
    tui.start()


def main():
    parser = ArgumentParser(description="PSIORI RL Toolbox Terminal UI")
    parser.add_argument(
        "port",
        type=int,
        nargs="?",
        default=5556,
        help="I/O port to subscribe to, will publish at port+1.",
    )
    parser.add_argument("--debug", action="store_const", const=True, default=False)
    args = parser.parse_args()
    run(args.port, debug=args.debug)


if __name__ == "__main__":
    main()
