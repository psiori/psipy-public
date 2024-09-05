# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import glob
import logging
import os
import time
from datetime import datetime, timedelta

import pytest

from psipy.core.io.logging import CALENDAR, RollingFileHandler

LOG = logging.getLogger(__name__)


class DummyRollingFileHandler(RollingFileHandler):
    now = 10

    def _get_now(self):
        return self.now


class TestLogging:
    @staticmethod
    def test_rollover(temp_dir):
        """Creates a new file every 1 hour.

        The end time is altered in the test to force a new file to be created.
        """
        # Set level to info
        prev_level = LOG.level
        LOG.setLevel(logging.INFO)

        now = datetime.now()
        hour_later = now + timedelta(hours=1)
        handler = RollingFileHandler(temp_dir, "LoggingTest", now, rollover="h")
        LOG.addHandler(handler)
        assert handler.file_count == 1
        LOG.info("File 1")
        handler.file_rollover_date = int(
            hour_later.strftime(CALENDAR[handler.rollover.lower()])
        )
        LOG.info("File 2")

        assert len(os.listdir(temp_dir)) == 2
        assert handler.file_count == 2
        for i, log in enumerate(sorted(glob.glob(temp_dir + "*.log"))):
            with open(log, "r") as f:
                assert f.readlines() == f"File {i+1}"

        # Remove the custom handler from the global logger and
        # reset logger level.
        LOG.handlers.pop()
        LOG.setLevel(prev_level)

    @staticmethod
    def test_more_than_one_entry_per_file(temp_dir):
        """Tests whether or not a file holds more than one log line.

        Test for Psipy PR #501 (https://github.com/psiori/psipy/pull/501).
        """
        # Set level to info
        prev_level = LOG.level
        LOG.setLevel(logging.INFO)

        now = datetime.now()
        handler = DummyRollingFileHandler(temp_dir, "LoggingTest", now, rollover="h")
        LOG.addHandler(handler)
        assert handler.file_count == 1
        LOG.info("File 1 1")
        LOG.info("File 1 2")
        handler.now = 11
        LOG.info("File 2 1")
        LOG.info("File 2 2")

        assert len(os.listdir(temp_dir)) == 2
        assert handler.file_count == 2
        for i, log in enumerate(sorted(glob.glob(temp_dir + "*.log"))):
            with open(log, "r") as f:
                for ii in range(1, 3):
                    assert f.readlines() == f"File {i+1} {ii}"

        # Remove the custom handler from the global logger and
        # reset logger level.
        LOG.handlers.pop()
        LOG.setLevel(prev_level)

    @staticmethod
    @pytest.mark.slow
    def test_zip(temp_dir):
        # Set level to info
        prev_level = LOG.level
        LOG.setLevel(logging.INFO)

        now = datetime.now()
        handler = DummyRollingFileHandler(
            temp_dir, "LoggingTest", now, rollover="h", zip=True
        )
        LOG.addHandler(handler)
        assert handler.file_count == 1
        LOG.info("File 1 1")
        LOG.info("File 1 2")
        handler.now = 11
        LOG.info("File 2 1")
        LOG.info("File 2 2")

        # Remove the custom handler from the global logger and
        # reset logger level.
        LOG.handlers.pop()
        LOG.setLevel(prev_level)
        handler.close()

        MAX_ITERS = 10
        for iter in range(1, MAX_ITERS + 1):
            try:
                assert handler.file_count == 2

                logs = sorted(glob.glob(os.path.join(temp_dir, "*.log")))
                zipfiles = sorted(glob.glob(os.path.join(temp_dir, "*.zip")))

                # delete_original=True
                assert len(os.listdir(temp_dir)) == 2
                assert len(logs) == 0
                assert len(zipfiles) == 2

                # delete_original=False
                # assert len(os.listdir(temp_dir)) == 4
                # assert len(logs) == len(zipfiles)

                for log, zf in zip(logs, zipfiles):
                    logname, _ = os.path.splitext(log)
                    zipname, _ = os.path.splitext(zf)
                    assert logname == zipname
            except Exception as err:
                if iter == MAX_ITERS:
                    raise err
                # Try again after 1 second.
                time.sleep(1)
