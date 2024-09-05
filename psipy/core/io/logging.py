# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Custom logging functionality.

Usage::

    # In `PACKAGE.__init__.py`
    from psipy.core.io import logging
    logging.register_logger()

    # Anywhere you want to use `LOG.status`
    from psipy.core.io import logging
    LOG = logging.getLogger(__name__)
    LOG.status("This is an important status message.")

    # Everywhere else
    import logging
    LOG = logging.getLogger(__name__)
    LOG.info("This is an important status message.")

.. autosummary::

    getLogger
    register_logger
    RollingFileHandler

"""

import logging
import os
from datetime import datetime
from typing import Optional, cast

from psipy.core.io.zip import zip_file_mp

__all__ = ["getLogger", "register_logger", "RollingFileHandler"]


#: Custom ``STATUS`` log level, which is just above the ``WARNING`` log level.
#: To be used for status messages which should be visible in production, e.g.
#: to which host and port the software is about to connect.
LOG_LEVEL_STATUS_NUM = 31

#: Mapping of string identifiers to datefmt literals. Implements a subset of
#: `timefmt <https://docs.python.org/3/library/time.html#time.strftime>`_.
CALENDAR = {"h": "%H", "d": "%d", "m": "%m", "y": "%Y"}


class Logger(logging.Logger):
    def status(self, message, *args, **kws):
        """Logs ``message`` at level :attr:`LOG_LEVEL_STATUS_NUM`.

        ``STATUS`` messages are logged at level :attr:`LOG_LEVEL_STATUS_NUM`.
        Use them sparsley.
        """
        if self.isEnabledFor(LOG_LEVEL_STATUS_NUM):
            self._log(LOG_LEVEL_STATUS_NUM, message, args, **kws)


def getLogger(name: str) -> Logger:
    """Returns a properly type-hinted instance of :class:`Logger`.

    Make sure to use :meth:`register_logger` before using this method.
    """
    return cast(Logger, logging.getLogger(name))


def register_logger():
    """Registers :class:`Logger` as python's std logging facilities logger.

    Call this method as early as possible within your sub-package, for example
    in the top-level ``__init__.py``, before ever calling :meth:`logging.getLogger`.

    Usage::

        >>> register_logger()
        >>> LOG = getLogger(__name__)  # psipy.core.io.logging.getLogger
        >>> LOG.status("This is an important status message.")

    """
    logging.addLevelName(LOG_LEVEL_STATUS_NUM, "STATUS")
    logging.setLoggerClass(Logger)
    logging.STATUS = LOG_LEVEL_STATUS_NUM
    logging.__all__ += ["STATUS"]


class RollingFileHandler(logging.FileHandler):
    """Logging FileHandler that rolls over after a specific time.

    Log files will roll over at the beginning of the requested rollover
    period, see ``CALENDAR`` in ``psipy.core.io.logging``. Format of the
    filename is:
        ``{name}-Optional{initial-time}-{creation-time}-{file-number}.log``
    where the initial time is the time the script is started, and is optional.

    The :class:`psipy.rl.SARTLogger` follows the same naming format to provide
    unified and easy to match filenames.

    Args:
        filepath: Path to save the log files to
        name: Name of project/script/run such that it is easily identifiable
        initial_time: The initial time the script started, optional
        rollover: At what point to rollover the *.log files.
                  One of ``["d", "w", "m", "y"]``.
        zip: Whether to zip and delete closed files.
        kwargs: Keyword arguments going to the logging :class:`FileHandler`
    """

    #: Number of log files created by this handler.
    file_count: int

    def __init__(
        self,
        filepath: str,
        name: str,
        initial_time: Optional[datetime],
        rollover: Optional[str] = None,
        zip: bool = False,
        **kwargs,
    ):
        self.filepath = filepath
        self.project_name = name
        self.initial_time = initial_time
        self.rollover = rollover
        self.file_rollover_date = self._get_now()
        self.file_count = 1
        self.zip = zip
        super().__init__(self._get_filepath(), **kwargs)

    def _get_now(self):
        """Gets :meth:`datetime.now`'s representation in :attr:`rollover`'s format.

        - In case :attr:`rollover` is ``None``, this method returns `None` and the
          appended data will be stored in a single file.
        - In other cases, this method returns an integer representation of the
          current date, e.g. 15 for ``self.rollover == 'h'`` at 3pm current hour.
        """
        if self.rollover is None:
            return -1
        return int(datetime.now().strftime(CALENDAR[self.rollover.lower()]))

    def _get_filepath(self) -> str:
        creation_time = datetime.now().strftime("%y%m%d-%H")
        if self.initial_time is not None:
            initial = self.initial_time.strftime("%y%m%d-%H%M%S")
            filename = f"{self.project_name}-{initial}-{{num:02d}}-{{creation}}.log"
        else:
            filename = f"{self.project_name}-{{num:02d}}-{{creation}}.log"
        return os.path.join(
            self.filepath,
            filename.format(creation=creation_time, num=self.file_count - 1),
        )

    def emit(self, record):
        """Log the provided record and roll over file if past rollover time."""
        if self.file_rollover_date != self._get_now():
            self.file_rollover_date = self._get_now()
            self.file_count += 1
            self.close()
            self.baseFilename = os.path.abspath(self._get_filepath())
        super().emit(record)

    def close(self):
        """Closes the current file and maybe zips and deletes it."""
        super().close()
        if self.zip:
            zip_file_mp(self.baseFilename, delete_original=True)
