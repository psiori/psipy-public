# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""TUI Windows and Overlays.

All :class:`Window`s and :class:`Overlay`s that the TUI can represent.
The :class:`Header` and :class:`Footer` behave different from the other Windows in the
sense that they receive a different input.  All other windows receive the same data
package. This is so the data can trickle down and be used by all windows without
required individual window references from needed to be publicly available.

See ``tui_interface.py`` for more information on the interface.
"""

import curses
import platform
from datetime import datetime
from typing import Optional, Tuple

from psipy.rl.io.tui import MIN_HEIGHT_WARNINGS
from psipy.rl.io.tui.tui_functions import format_date, print_centered
from psipy.rl.io.tui.tui_functions import print_centered_on_char, print_dict
from psipy.rl.io.tui.tui_interface import Overlay, Window


class Header(Window):
    """The header for the TUI, displaying CM ports and control setup."""

    def __init__(self):
        # Header has a special title, thus we send no title to super
        super().__init__()
        self.ports: Optional[Tuple[int, int]] = None
        self.title_string = " PSIORI Act TUI | i/o {pin}/{pout} "

    def _print_title(self, style=curses.A_STANDOUT) -> None:
        """Print the header title not centered."""
        pin, pout = 0, 0
        if self.ports is not None:
            pin, pout = self.ports
        self.window.addstr(0, 2, self.title_string.format(pin=pin, pout=pout))

    def _print(self, data_package) -> None:
        self.ports = data_package["ports"]
        plant = data_package["plant"]
        controller = data_package["controller"]
        start = data_package["start"]
        _, width = self.dims
        header_string = f" Plant: {plant} | Controller: {controller} "
        time_string = (
            f"Start Time: {format_date(start)} "
            f"| Current Time: {format_date(datetime.now())}"
        )
        elapsed_string = f"Elapsed: {str(datetime.now() - start)}"

        print_centered_on_char(self.window, header_string, "|", width, 1, curses.A_BOLD)
        print_centered_on_char(self.window, time_string, "|", width, 2)
        print_centered(self.window, elapsed_string, width, 3)


class Footer(Window):
    """The footer of the TUI, displaying keyboard interaction and tab options."""

    def __init__(self):
        super().__init__(border=False)

    def _print(self, data_package) -> None:
        _, width = self.dims
        if width is None:
            raise RuntimeError("Received a None where an int was expected!")

        if not data_package.get("typing", False):
            footer_options = "q: quit (stops agent) | e: exit tui (agent remains)"
            # TODO: Can also do < and > for left/right
            footer_tabs = "Tabs: [m]ain"
            if platform.system() == "Windows":
                footer_tabs += " | [l]ogs"
            gap = " " * (width - len(footer_tabs) - len(footer_options) - 2)
            text = footer_options + gap + footer_tabs
        else:
            text = (
                "Input mode. Leave using ESC, submit using ENTER. "
                f"Input: {data_package['typed']}"
            )

        self.window.addstr(1, 1, text, curses.A_STANDOUT)


class SARTWindow(Window):
    """Displays information from the SART."""

    def __init__(self):
        super().__init__(title=" Status ")

    def _print(self, data_package) -> None:
        state = data_package["status"]["values"]
        _, width = self.dims

        # Print the state
        print_centered(self.window, "State", width, 1, curses.A_UNDERLINE)
        print_dict(self.window, 2, 1, state, align_on_decimal=True)

        # cost = data_package["status"]["cost"]
        # terminal = data_package["status"]["terminal"]
        # returns = {"Cost": cost, "Terminal": terminal}
        # y, x = self.window.getyx()
        # y += 2
        # print_centered(self.window, "Returns", width, y, curses.A_UNDERLINE)
        # print_dict(self.window, y + 1, 1, returns)

        # Get the current position at the end of the above dict and add a space
        action = data_package["status"]["action"]
        y, x = self.window.getyx()
        y += 2
        print_centered(self.window, "Actions", width, y, curses.A_UNDERLINE)
        print_dict(self.window, y + 1, 1, action)


class MetaWindow(Window):
    """Displays the meta information"""

    def __init__(self):
        super().__init__(title=" Meta Information ")

    def _print(self, data_package) -> None:
        meta = data_package["meta"]
        print_dict(self.window, 1, 1, meta, align_on_decimal=True)


class CMWindow(Window):
    """Displays cycle manager info including mean and stdev times."""

    def __init__(self):
        super().__init__(title=" Cycle Manager Info ")

    def _print(self, data_package):
        cycle_dict = dict()
        for name, timer in data_package["cycle"].items():
            avg, std = timer
            avg *= 1000
            std *= 1000
            cycle_dict[name] = f"{avg:7.2f}ms ±{std:5.2f}"
        print_dict(self.window, 1, 1, cycle_dict, align_on_decimal=True)


class LogWindow(Window):
    """Displays logs.

    This window has two forms: warning and normal. The warning window
    will only show the N latest warning logs, where N is defined by a
    constant in the tui's ``__init__.py``. The normal window lives inside
    a :class:`Box` that hides behind the :class:`MainTab`, and is resized
    to fill any unused space. Hence, if there is no unused space, the
    window will not appear.

    Args:
        warning: Turn on the warning mode (see above).
    """

    def __init__(self, warning: bool = False):
        self.warning = warning
        if warning:
            self.fixed_height = MIN_HEIGHT_WARNINGS
        title = " Logs "
        if warning:
            title = " High Level Logs "
        super().__init__(title=title)

    def _print(self, data_package):
        _, width = self.dims
        logs = list(data_package["logs"])
        if self.warning:
            logs = [
                log
                for log in logs
                if (
                    log.startswith("WARNING")
                    or log.startswith("STATUS")
                    or log.startswith("ERROR")
                    or log.startswith("CRITICAL")
                )
            ]
        for i, log in enumerate(logs[-(self.height - 2) :]):
            # TODO: Handle log messages which contain new lines?
            text = log[:width]
            self.window.addstr(i + 1, 1, text)


class BlankWindow(Window):
    """A placeholder blank window."""

    def _print(self, data_package) -> None:
        string = "This space intentionally left blank."
        if data_package is not None:
            string += f"\n\n{data_package}"
        self.window.addstr(1, 1, string)


class ErrorOverlay(Overlay):
    """Overlay that displays an error and traceback."""

    def _print(self, data_package) -> None:
        # return
        exception = data_package["exception"]
        traceback = data_package["traceback"]
        # All subsequent lines of the traceback are set back one space;
        # force them forward again one space
        traceback = traceback.replace("\n", "\n ")
        text = (
            f"An error occurred in the interaction loop! "
            f"({exception})\n\n {traceback}"
        )
        height, width = self.dims
        if width is None or height is None:
            raise RuntimeError("Received a None where an int was expected!")
        acknowledge = "Press Enter to Acknowledge.   "
        spaces = " " * (width - len(acknowledge) - 3)  # - to move out of border
        self.window.addstr(1, 1, text)
        self.window.addstr(height - 2, 1, spaces + acknowledge, curses.A_STANDOUT)
