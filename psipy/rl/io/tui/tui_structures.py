# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""TUI Structures.

See ``tui_interface.py`` for more information on the interface.
"""
from psipy.rl.io.tui.tui_interface import Structure


class Rows(Structure):
    """Organizes windows into rows.

    :class:`Rows` takes multiple windows and tiles them on top of each other
    so that they generate rows. The middle row can be slightly larger than
    the other rows, as sometimes the size allocated to the row may not divide
    evenly into the number of windows. In practice, this is barely (if at all)
    noticeable.
    """

    def create(self, height: int, width: int, x: int, y: int) -> None:
        # Alter the given width based on any forced widths in the child windows
        # This will resize the structure in relation to neighboring structures.
        widths = [w.width for w in self.windows]
        if any(w is None for w in widths) or sum(widths) < width:
            widths = [width]
        width = max(widths)
        super().create(height, width, x, y)

        num_windows = len(self.windows)
        window_height = height // num_windows
        remainder_height = height % num_windows
        heights = [window_height] * num_windows
        # Middle window (if exists) gets the remainder height
        heights[num_windows // 2] += remainder_height
        # Check if there are any fixed height windows and alter the
        # heights to accommodate them
        set_heights = [w.height for w in self.windows]
        heights = self.accomodate_fixed_dims(set_heights, heights)

        for i, window in enumerate(self.windows):
            window.create(heights[i], width, x, y)
            # Slide the starting y position down
            y += window.height

    def _print(self, data_package) -> None:
        for window in self.windows:
            window.print(data_package)


class Columns(Structure):
    """Organizes windows into columns.

    :class:`Columns` takes multiple windows and tiles them one next to another
    so that they generate columns. The middle column can be slightly larger than
    the other columns, as sometimes the size allocated to the column may not divide
    evenly into the number of windows. In practice, this is barely (if at all)
    noticeable.
    """

    def create(self, height: int, width: int, x: int, y: int) -> None:
        # Alter the given height based on any forced heights in the child windows
        # This will resize the structure in relation to neighboring structures.
        heights = [w.height for w in self.windows]
        if any(h is None for h in heights) or sum(heights) < height:
            heights = [height]
        height = max(heights)
        super().create(height, width, x, y)

        num_windows = len(self.windows)
        window_width = width // num_windows
        remainder_width = width % num_windows
        widths = [window_width] * num_windows
        # Middle window (if exists) gets the remainder width
        widths[num_windows // 2] += remainder_width
        # Check if there are any fixed width windows and alter the
        # widths to accommodate them
        set_widths = [w.width for w in self.windows]
        widths = self.accomodate_fixed_dims(set_widths, widths)

        for i, window in enumerate(self.windows):
            window.create(height, widths[i], x, y)
            # Slide the starting x position to the right
            x += widths[i]

    def _print(self, data_package) -> None:
        for window in self.windows:
            window.print(data_package)


class Box(Structure):
    """A structure that contains only one :class:`Window`.

    This is the identity structure. It has no effect except to wrap a
    :class:`Window` into a :class:`Structure`, without altering its placement.
    """

    def create(self, height: int, width: int, x: int, y: int) -> None:
        assert len(self.windows) == 1, "Can only put one window into a Box."
        super().create(height, width, x, y)
        self.windows[0].create(height, width, x, y)

    def _print(self, data_package) -> None:
        self.windows[0].print(data_package)
