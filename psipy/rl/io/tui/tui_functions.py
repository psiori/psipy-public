# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""TUI Helper Functions.

Functions here help print desired text in a certain format or format datatypes into
a TUI parseable representation.

.. autosummary::

    format_date
    format_timer
    print_centered
    print_centered_on_char
    print_dict

See ``tui_interface.py`` for more information on the interface.
"""

import curses
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from psipy.core.utils import guess_primitive

_number_dtypes = (np.generic, float, int)


def format_date(date):
    """Pretty print a date."""
    return date.strftime("%Y-%m-%d %H:%M:%S")


def format_timer(ts: float) -> str:
    """Pretty print timestamps."""
    if ts < 100:
        return f"{ts:.2f}s"
    ds = datetime.fromtimestamp(ts)
    if ts < 3600:
        return ds.strftime("%M:%S.%f")[:-3]
    return ds.strftime("%H:%M:%S.%f")[:-3]


def print_centered(
    window,
    text: str,
    total_width: Optional[int],
    y: Optional[int],
    style=curses.A_NORMAL,
) -> None:
    """Print centered text.

    Args:
        window: The curses window to add the text to
        total_width: Total width of the text area, usually the width of the window
        y: The vertical position of the text within the window
        style: A curses text style, e.g. curses.A_BOLD
    """
    if total_width is None or y is None:
        raise RuntimeError("Received a None where an int was expected!")
    window.addstr(y, total_width // 2 - len(text) // 2, text, style)


def print_centered_on_char(
    window,
    text: str,
    char: str,
    total_width: Optional[int],
    y: Optional[int],
    style=curses.A_NORMAL,
) -> None:
    """Print centered text around a character.

    Args:
        window: The curses window to add the text to
        char: The character to center around
        total_width: Total width of the text area, usually the width of the window
        y: The vertical position of the text within the window
        style: A curses text style, e.g. curses.A_BOLD
    """
    assert char in text
    if total_width is None or y is None:
        raise RuntimeError("Received a None where an int was expected!")
    midpoint = text.find(char) + 1  # +1 to shift to proper center
    window.addstr(y, 1 + total_width // 2 - midpoint, text, style)


def print_dict(
    window,
    y: int,
    x: int,
    dic: Dict[str, Any],
    tab_size: int = 1,
    align_on_decimal: bool = False,
) -> None:
    """Prints k,v of a dictionary line by line, aligning on decimals if desired.

    Args:
        window: The curses window to print the dict in
        y: The vertical position to start the dict in the window
        x: The horizontal position to start the dict in the window
        dic: The dictionary to print
        tab_size: How big spaces are between keys and the colon (:)
        align_on_decimal: If true, try to align numbers on their decimal points
                          (works only for numbers up to a string length of 9)
    """
    string = ""
    tab_size = " " * tab_size

    # To align on decimal, all values must be numeric (string numbers don't count!)
    if align_on_decimal:
        if not all([isinstance(val, _number_dtypes) for val in dic.values()]):
            # Bad state return; check for non-numbers!
            align_on_decimal = False

    max_len_key = max([len(key) for key in dic])
    # Max displayable number length before decimals become unaligned, e.g. 12345.678
    max_len_value = 9

    for key, value in dic.items():
        value = f"{value:3.3f}" if align_on_decimal else value
        tab = " " * (max_len_key - len(str(key)))
        alignment = (
            " " * (max_len_value - len(value.split(".")[0])) if align_on_decimal else ""
        )
        string += f"{key}{tab}:{tab_size}{alignment}{value}\n "

    window.addstr(y, x, string)


id = random.randint(0, 500)


def log_to_file(msg, name: str = "LOG.txt") -> None:
    """Log to a file.

    When developing in the TUI, any prints will be hidden. To get around
    this, use this function to write to a file whatever is to be printed.
    Each run of the TUI will receive a random number, to be able to
    distinguish between runs.
    """
    with open(name, "a") as f:
        f.write(f"{id}: {msg}\n")


def parse_user_input(txt: str):
    """Parse user input styled ``topic: key1=val1 key2=val2``.

    Example::

        >>> parse_user_input("plant sp=62.2")
        ('plant', {'sp': 62.2})

    """
    topic, querystring = txt.split(" ", maxsplit=1)
    kvs = querystring.split(" ")
    result = {}
    for kv in kvs:
        k, v = kv.split("=")
        result[k] = guess_primitive(v)
    return topic, result
