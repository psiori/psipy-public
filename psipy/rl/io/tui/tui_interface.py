# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""TUI Interfaces for Window, Overlay, Structure, and Tab.

The TUI has a required interface which allows all components to nest within
each other and create the TUI layout as well as print the required information.

There are three basic classes, one derived interface which behaves differently,
and one mixin.
The basic classes are:

    1. :class:`Window`
    2. :class:`Structure`
    3. :class:`Tab`

the derived class:

    4. :class:`Overlay`

and the mixin:

    5. :class:`DirectoryMixin`

A brief summary of each follows. For more information, see each class' individual
docstring.

    * :class:`Window`: The basic building block. Can display text.
    * :class:`Structure`: Textless window that can hold multiple windows and display
                          them in a certain pattern (e.g. in columns).
    * :class:`Tab`: A wrapper that holds the upmost window/structure of the desired
                   information to display, as well as an optional :class:`Overlay`.
    * :class:`Overlay`: A special window type which always appears in the middle of
                        the screen and can only be hidden by hitting Enter. Most
                        commonly used for error messages.
    * :class:`DirectoryMixin`: A mixin which creates a recursive directory of child
                               windows. These children can be accessed through
                               dictionary notation. :class:`Tab` and :class:`Structure`
                               have this mixin.

Classes on this page are organized in a hierarchical manner, such that the "smallest"
components are on the top, and "largest" on the bottom.
"""

import curses
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from psipy.rl.io.tui.tui_functions import print_centered


class Window:
    """The basic building block in the TUI curses environment.

    Windows are the building blocks in the TUI tree.
    All text displayed on screen arise from :class:``Window`` classes.
    Windows can optionally have a title and a border, and each require
    a specific implementation of :meth:`_print` to tell the Window how
    to display the information given to it.

    Curses follows a cartesian coordinate system, such that x is columns and
    y is rows, opposite of what might be considered standard. Printing a window
    requires the following parameters: (nlines, ncols, begin_y, begin_x)

    Windows get their size from their parent, which gets its size from its parent,
    all the way up to the main screen, whose size is the size of the Terminal. Thus,
    Windows are self sizing and will expand to fit the size of the :class:`Structure`
    or screen space they are intended to fill.

    Error modes: If the text flows outside the window, the curses error thrown
    will be caught and a warning text placed in the Window saying that it needs
    to be resized. If there is an error in parsing the data provided and printing
    it to the screen, the text "Error in parsing data..." will appear.

    Args:
        title: Optionally provided title that will be printed in the center of the
               window
        border: True if a lined border should be printed around the Window
    """

    #: The curses window where text is displayed
    window: Any

    #: The upper left x coordinate (starting column)
    x: int

    #: The upper left y coordinate (starting row)
    y: int

    #: The height of the Window
    height: Optional[int] = None

    #: The width of the Window
    width: Optional[int] = None

    #: The fixed height of the Window
    fixed_height: Optional[int] = None

    #: The fixed width of the Window
    fixed_width: Optional[int] = None

    #: The fixed x coordinate of the Window
    fixed_x: Optional[int] = None

    #: The fixed y coordinate of the Window
    fixed_y: Optional[int] = None

    def __init__(
        self,
        title: Optional[str] = None,
        border: bool = True,
    ):
        self.title = title
        self.has_border = border
        self._created = False

    def __str__(self) -> str:
        """Class name of Window or a subclass of."""
        return self.__class__.__name__

    def _print_title(self, style=curses.A_STANDOUT) -> None:
        """Print the title centered on the top of the window."""
        if self.title is not None:
            height, width = self.dims
            print_centered(self.window, self.title, width, 0, style)

    def create(self, height: int, width: int, x: int, y: int) -> None:
        """Create the window with the given size and position.

        Height and width are overwritten by the subclass level height and width if
        provided in the subclass. If they are None, height and width will be
        determined by the program to fill all available space (thus, set as provided
        to this method).
        """
        self.x = x
        self.y = y
        if self.fixed_x is not None:
            self.x = self.fixed_x
        if self.fixed_y is not None:
            self.y = self.fixed_y
        # Set to dynamic dims or fixed dims if set
        self.width = width
        self.height = height
        if self.fixed_height is not None:
            self.height = self.fixed_height
        if self.fixed_width is not None:
            self.width = self.fixed_width

        try:
            self.window = curses.newwin(
                max(self.height, 3), max(self.width, 3), self.y, self.x
            )
        except curses.error:
            pass
        self._created = True

    def print(self, data_package: Dict[str, Any]) -> None:
        """Print the window.

        Displays an error message if printing failed or if there is an exception.
        """
        if not self._created:
            raise ValueError(f"{self} was not created, can not print!")
        try:
            self._print(data_package)
        except curses.error:
            self.window.erase()
            self._print_title()
            try:
                self.window.addstr(1, 1, "Please enlarge window to see all info.")
            except curses.error:
                pass
        except Exception:
            self.window.addstr(2, 1, "Error in parsing data...")
        if self.has_border:
            self.window.border()
        self._print_title()
        self.window.noutrefresh()

    @abstractmethod
    def _print(self, data_package: Dict[str, Any]) -> None:
        """Custom print method for individual windows."""
        raise NotImplementedError

    @property
    def dims(self) -> Tuple[Optional[int], Optional[int]]:
        """Height and width of window."""
        return (self.height, self.width)

    @property
    def anchor(self) -> Tuple[int, int]:
        """Upper left corner coordinates."""
        return (self.y, self.x)


class Overlay(Window):
    """Window that pops up and overlays other windows under certain conditions."""

    def __init__(self, screen):
        super().__init__(title="ERROR", border=True)
        self._hidden = True
        self.screen = screen

    def _print_title(self, style=curses.A_BLINK) -> None:
        super()._print_title(style)

    def create(self, height: int, width: int, x: int, y: int):
        """Creates the error window in the relative center."""
        y, x = self.screen.getmaxyx()
        height = int(y * 0.70)
        width = int(x * 0.85)
        self.width = width
        self.height = height

        right_shift = (x - width) // 2
        bottom_shift = (y - height) // 2
        self.window = curses.newwin(height, width, bottom_shift, right_shift)

        self._hidden = False
        self._created = True

    def clear(self):
        """Removes reference to window."""
        self.window = None
        self._hidden = True

    @property
    def hidden(self) -> bool:
        "Whether or not the overlay is hidden."
        return self._hidden


class DirectoryMixin:
    """Mixin to create a dictionary of child windows."""

    def __getitem__(self, item):
        return self.directory[item]

    @staticmethod
    def _generate_unique_name(window: Window, _dir: Dict[str, Window]) -> str:
        """Generates a unique window name by appending increasing integers."""
        i = 1
        name = str(window) + str(i)
        while True:
            if name in _dir.keys():
                i += 1
                name = str(window) + str(i)
            else:
                break
        return name

    def _recurse_windows(self, windows) -> Dict[str, Window]:
        """Recursively creates a dictionary of window name and window."""
        directory: Dict[str, Window] = dict()
        for window in windows:
            name = self._generate_unique_name(window, directory)
            if isinstance(window, Structure):
                directory.update(self._recurse_windows(window.windows))
            directory[name] = window
        return directory

    @property
    def directory(self):
        if hasattr(self, "windows"):
            return self._recurse_windows(self.windows)
        return self._recurse_windows(self.body.windows)


class Structure(DirectoryMixin, Window):
    """A textless :class:`Window` that can contain multiple Windows.

    Structures are used to organize multiple windows in space. Normally,
    they contain no text themselves, although they could be forced to have
    titles and borders if desired (this would clash with the underlying windows'
    titles and borders, however). The structure will automatically divy out
    proper sizing information to its child windows depending on what type of
    structure is trying to be created.

    The size of a structure is either defined by its children or its parent.
    If the children have fixed sizes, the Structure will prioritize the
    maximum dimensions found among its children. Otherwise, it will use
    the equally divied out sizing from its parent.

    Args:
        windows: A tuple of one or more INSTANTIATED :class:`Window`s
    """

    def __init__(self, *windows, height=None, width=None):
        super().__init__(title=None, border=False)
        self.windows = tuple(windows)
        self.height = height
        self.width = width

    def accomodate_fixed_dims(
        self, fixed_dims: List[Optional[int]], generated_dims: List[int]
    ) -> List[int]:
        """Alter equally generated dimensions to accommodate windows with set sizes."""
        left_over = 0
        for i, fixed in enumerate(fixed_dims[:-1]):
            if fixed is not None:
                old = generated_dims[i]
                left_over += old - fixed
                generated_dims[i] = fixed

        generated_dims[-1] -= left_over
        return generated_dims

    @abstractmethod
    def create(self, height: int, width: int, x: int, y: int) -> None:
        """Creates the structure by setting its properties."""
        self.x = x
        self.y = y
        self.width = self.get_width(width)
        self.height = self.get_height(height)

    def get_height(self, given_height: int):
        heights = [given_height]
        for _name, window in self.directory.items():
            height = window.height
            if height is not None:
                heights.append(height)
        for child in self.windows:
            child.height = None
        return max(heights)

    def get_width(self, given_width: int):
        widths = [given_width]
        for _name, window in self.directory.items():
            width = window.width
            if width is not None:
                widths.append(width)
        for child in self.windows:
            child.width = None
        return max(widths)

    def print(self, data_package: Dict[str, Any]) -> None:
        """Print the structure."""
        self._print(data_package)


class Tab(DirectoryMixin):
    """The body of the TUI curses environment.

    Tabs go in between the header and footer and display any information
    required. Subclasses of :class:`Tab` contain a structural tree of
    :class:`Structure` and :class:`Window` to create the desired layout
    inside the ``body`` self variable.

    Tabs can have an overlay, which is only displayed when some condition
    is met. See the docstring for :class:`Overlay`.
    """

    #: The highest level structure or window to display
    body: Union[Window, Structure]

    def __init__(self, screen, overlay: Overlay):
        self.screen = screen
        self.overlay = overlay
        self._error_package: Optional[Dict[str, Any]] = None

    def set_error_package(self, error_package: Dict[str, Any]) -> None:
        self._error_package = error_package

    def create(self, header_height: int, footer_height: int) -> None:
        max_y, max_x = self.screen.getmaxyx()
        max_y -= header_height + footer_height
        # Send the whole size of the screen and upper left corner to the body
        self.body.create(max_y, max_x, 0, header_height)

    def print(self, data_package: Dict[str, Any]) -> None:
        self.body.print(data_package)
        if not self.overlay.hidden and self._error_package is not None:
            self.overlay.print(self._error_package)

    @property
    def empty_height(self):
        """Returns any unused height in the window that the body does not include."""
        try:
            heights = [w.height for w in self.body.windows]
            return self.screen.getmaxyx()[0] - sum(heights)
        except TypeError:  # None in heights
            return 0
