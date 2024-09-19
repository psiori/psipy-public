# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""TUI.

Global constants related to the TUI are stored here. Anywhere else
and circular import errors are thrown. If you have a better idea,
please let me know! :D

Version history (rough record):

    * 1.0 - Basic single tab interface in curses
        * 1.0c - Renamed ``reward`` to ``cost``
    * 2.0 - Added keyboard interaction and log window
    * 3.0 - Rewrote to robust tabbed interface in urwid
    * 4.0 - Rewrote to tabbed interface in curses
        * 4.1 - Allowed filtering of state/action channels at init
        * 4.2 - Fixed flickering in data caused by printing when no
                data existed.
        * 4.3 - Made logs fill unused space at bottom and added warning
                logs window at top of the main tab.

"""

__version__ = "4.3"

# Global heights
MIN_NUM_WARNINGS = 3
MIN_HEIGHT_WARNINGS = MIN_NUM_WARNINGS + 2
HEIGHT_HEADER = 5
HEIGHT_FOOTER = 3
