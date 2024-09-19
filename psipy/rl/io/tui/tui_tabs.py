# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""TUI Tabs.

All :class:`Tab`s that the TUI can represent. All tabs must be entered into the
``TAB_REGISTER`` that the TUI knows which tab to load when being requested by the
:class:`CycleManager` to do so.

All windows within the body of tabs must be instantiated!
See ``tui_interface.py`` for more information on the interface.
"""

from psipy.rl.io.tui.tui_interface import Tab
from psipy.rl.io.tui.tui_structures import Box, Columns, Rows
from psipy.rl.io.tui.tui_windows import BlankWindow, CMWindow, ErrorOverlay
from psipy.rl.io.tui.tui_windows import LogWindow, MetaWindow, SARTWindow


class MainTab(Tab):
    def __init__(self, screen):
        overlay = ErrorOverlay(screen)
        super().__init__(screen, overlay)
        self.body = Rows(
            LogWindow(warning=True),
            Columns(SARTWindow(), Rows(MetaWindow(), CMWindow())),
        )


class TestTab(Tab):
    def __init__(self, screen):
        overlay = ErrorOverlay(screen)
        super().__init__(screen, overlay)
        self.body = Rows(SARTWindow(), SARTWindow())


class BlankTab(Tab):
    def __init__(self, screen):
        overlay = ErrorOverlay(screen)
        super().__init__(screen, overlay)
        self.body = Box(BlankWindow())


class LogsTab(Tab):
    """Quasi-Tab to fill in the unused space with logs."""

    def __init__(self, screen):
        overlay = ErrorOverlay(screen)
        super().__init__(screen, overlay)
        self.body = Box(LogWindow())


# class WhalebackTab(Tab):
#     def __init__(self, screen):
#         overlay = ErrorOverlay(screen)
#         super().__init__(screen, overlay)
#         self.body = None #TODO
#
#     def _print(self):
#         raise NotImplementedError


TAB_REGISTER = dict(blank=BlankTab, main=MainTab, test=TestTab, logs=LogsTab)
