# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os
import time
from collections import OrderedDict
from datetime import datetime
from typing import Optional

import h5py
import numpy as np
import pytest

from psipy import __version__ as psipy_version
from psipy.rl.io.sart import ExpandableDataset, SARTLogger
from psipy.rl.io.sart import SARTReader, SARTWriter, nanlen
from psipy.rl.plant.tests.mocks import MockAction, MockPlant, MockState

data_array = np.array(
    [[101, 1.337, 22, 34.6, 67, 0.1, 1587.7, 97, 1.0111, 1.1, 34.3, 84.25, 1.9]]
)


@pytest.fixture
def test_data():
    plant = MockPlant(with_meta=True)
    action = MockAction(np.array([1, 2]))
    plant.notify_episode_starts()
    state = plant.check_initial_state(None)  # Supply a state for cost
    return dict(state=state.as_dict(), action=action.as_dict())


@pytest.fixture
def hdf5file(tmp_path):
    with h5py.File(tmp_path / "test-file.h5", "w") as f:
        yield f
    os.remove(tmp_path / "test-file.h5")


def nan_equals(a, b):
    """Compare two arrays elementwise, treating NaN == NaN."""
    return np.all(np.logical_or(a == b, np.logical_and(np.isnan(a), np.isnan(b))))


class TestExpandableDataset:
    @staticmethod
    def test_add_row(hdf5file):
        exp = ExpandableDataset(hdf5file, "test", (1,), 2)
        exp.add_row(np.array([10]))
        assert nan_equals(hdf5file["test"][:], np.array([10, np.nan])[:, None])

    @staticmethod
    def test_expand(hdf5file):
        exp = ExpandableDataset(hdf5file, "test", (1,), 2)
        for _ in range(3):
            exp.add_row(np.array([10]))
        exp.finalize()
        assert np.all(hdf5file["test"][:] == np.array([10, 10, 10])[:, None])


class TestWriter:
    @staticmethod
    def test_file_attrs(temp_dir):
        initial_time = datetime.now()
        writer = SARTWriter(temp_dir, "test", 1, initial_time)
        assert writer.file.attrs["task"] == "test"
        # As we don't know the exact creation time, see if it is formatted correctly
        assert len(writer.file.attrs["created"].split("-")) == 2
        # As initial time is give, check if the format is correct
        f = os.listdir(temp_dir)[0]
        assert f.split("-")[1] == initial_time.strftime("%y%m%d")
        assert f.split("-")[2] == initial_time.strftime("%H%M%S")
        assert writer.file.attrs["writer-version"] == writer.__version__()
        assert writer.file.attrs["psipyrl-version"] == psipy_version
        writer.notify_episode_stops()

    @staticmethod
    def test_write_episode(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        writer.append(test_data)
        writer.notify_episode_stops()

        assert len(os.listdir(temp_dir)) == 1
        f = h5py.File(os.path.join(temp_dir, os.listdir(temp_dir)[0]), "r")
        assert f["state"]["values"]["430TH01AT027_PV"].shape == (1,)
        # Make sure actions with a / are saved properly
        assert f["action"]["430PP41FIC006|PID_CV_MAN"].shape == (1,)
        assert f["action"]["430PP41FIC006|PID_CV_MAN"][:] == [1]
        assert f["meta"]["meta2"][:] == [123]
        assert f["meta"]["met|a3"][:] == np.array(["Start"])
        for ordering in ["state", "action", "meta"]:
            assert ordering in f.attrs.keys()

    @staticmethod
    def test_lost_key(temp_dir, test_data):
        # Remove a numeric action key and str meta key from the data
        from copy import deepcopy

        altered_test_data = deepcopy(test_data)
        del altered_test_data["action"]["430PP41FIC006/PID_CV_MAN"]
        del altered_test_data["state"]["meta"]["met/a3"]
        writer = SARTWriter(temp_dir, "Test", 1)
        writer.append(test_data)
        writer.append(altered_test_data)
        writer.append(test_data)
        writer.notify_episode_stops()

        f = h5py.File(os.path.join(temp_dir, os.listdir(temp_dir)[0]), "r")
        assert f["action"]["430PP41FIC006|PID_CV_MAN"].shape == (3,)

        # Testing array equality with nans is a pain
        desired = np.array([1.0, np.nan, 1.0])
        recorded = f["action"]["430PP41FIC006|PID_CV_MAN"][:]
        for i in range(len(desired)):
            try:
                assert desired[i] == recorded[i]
            except AssertionError:
                assert np.isnan(desired[i]) == np.isnan(recorded[i])

        assert f["meta"]["met|a3"].shape == (3,)
        assert np.array_equal(f["meta"]["met|a3"][:], np.array(["Start", "", "Start"]))

    @staticmethod
    def test_append_checks(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        writer.append(test_data)  # No problem.
        test_data["state"]["values"] = dict(test_data["state"]["values"])
        with pytest.raises(AssertionError):
            writer.append(test_data)
        test_data["state"]["values"] = OrderedDict(test_data["state"]["values"])
        writer.append(test_data)  # Problem fixed.
        test_data["action"] = dict(test_data["action"])
        writer.append(test_data)
        test_data["action"] = OrderedDict(test_data["action"])
        writer.append(test_data)
        test_data["badkey"] = (1, 2, 3)
        with pytest.raises(AssertionError):
            writer.append(test_data)
        writer.notify_episode_stops()

    @staticmethod
    def test_write_many_episodes(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        for _ in range(10):
            writer.append(test_data)
        writer.notify_episode_stops()

        assert len(os.listdir(temp_dir)) == 1
        f = h5py.File(os.path.join(temp_dir, os.listdir(temp_dir)[0]), "r")
        assert f["state"]["values"]["430TH01AT027_PV"].shape == (10,)
        assert f["action"]["430PP41FIC006|PID_CV_MAN"].shape == (10,)
        assert all(f["action"]["430PP41FIC006|PID_CV_MAN"][:] == 1)


class TestSARTLogger:
    @staticmethod
    @pytest.mark.slow
    def test_chunking(temp_dir, test_data):
        # Creates a new file every 1 hour. We append data for 1 seconds but
        # we change current end time inorder to have two files, a full and a
        # partial one.
        initial_time = datetime.now()
        logger = SARTLogger(temp_dir, "test", initial_time, rollover="h")
        logger.notify_episode_starts()
        total = 0  # for-loop run count
        t_end = time.time() + 1
        writer0: Optional[SARTWriter] = None
        try:
            while time.time() < t_end:
                logger.append(test_data)
                total += 1
                time.sleep(0.001)
                print(total)
                if total == 10:
                    writer0 = logger.writer
                    logger.file_rollover_date += 1
                assert logger.file_count == 1 + (total > 10)
        finally:  # always stop the logger.
            # wait for second writer
            logger.notify_episode_stops()
            # wait for first writer
            if writer0 is not None:
                writer0.join()
        assert logger.sample_count == total
        assert logger.episode_file_count == 2
        assert len(os.listdir(temp_dir)) == 2
        saved = 0
        for file in os.listdir(temp_dir):
            assert file.split("-")[1] == initial_time.strftime("%y%m%d")
            assert file.split("-")[2] == initial_time.strftime("%H%M%S")
            f = h5py.File(os.path.join(temp_dir, file), "r")
            saved += f["state"]["values"]["430TH01AT027_PV"].shape[0]  # example node
        assert saved == total

        # Make sure counters reset/increment properly.
        logger.notify_episode_starts()
        logger.notify_episode_stops()
        assert logger.episode_file_count == 1
        assert logger.episode_count == 2
        assert logger.sample_count == 0

    @staticmethod
    @pytest.mark.slow
    def test_no_rollover(temp_dir, test_data):
        # Creates one single writer without rollover
        logger = SARTLogger(temp_dir, "test")
        logger.notify_episode_starts()
        try:
            total = 0  # for-loop run count
            t_end = time.time() + 1
            while time.time() < t_end:
                logger.append(test_data)
                total += 1
                time.sleep(0.001)  # 1000hz
        finally:  # always stop the logger.
            logger.notify_episode_stops()  # close + join, blocking
        assert logger.sample_count == total
        assert len(os.listdir(temp_dir)) == 1
        saved = 0
        for file in os.listdir(temp_dir):
            f = h5py.File(os.path.join(temp_dir, file), "r")
            saved += f["state"]["values"]["430TH01AT027_PV"].shape[0]  # example node
        assert saved == total

    @staticmethod
    def test_single(temp_dir, test_data):
        # Creates one single writer without rollover
        logger = SARTLogger(temp_dir, "test", rollover="d", single=True)
        try:
            logger.notify_episode_starts()
            writer = logger.writer
            assert logger.episode_count == 1
            assert logger.episode_file_count == 1
            logger.append(test_data)
            logger.append(test_data)
            logger.notify_episode_stops()
            logger.notify_episode_starts()
            assert logger.episode_count == 1  # This does not count up in single mode!
            assert logger.episode_file_count == 1
            logger.append(test_data)
            logger.append(test_data)
            logger.append(test_data)
            logger.append(test_data)
            logger.notify_episode_stops()
            assert writer == logger.writer
        finally:
            logger.shutdown()


class TestReader:
    @staticmethod
    def test_load_episode(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        writer.append(test_data)
        writer.append(test_data)
        writer.notify_episode_stops()

        with SARTReader(os.path.join(temp_dir, os.listdir(temp_dir)[0])) as reader:
            experience = reader.load_full_episode()
            assert len(experience) == 4
            assert experience[0].shape == (2, len(MockState.channels()))
            # Necessary to use allclose due to
            # float imprecision (float32/float64 comparison)
            np.testing.assert_allclose(experience[0][0, None], data_array)
            assert all(experience[-1] == 1)
            assert all(experience[-2] == False)  # noqa E712

        with SARTReader(os.path.join(temp_dir, os.listdir(temp_dir)[0])) as reader:
            experience = reader.load_full_episode(
                state_channels=["430PP42FIC006/PID_PV", "430PP42DIC007/PID_PV"]
            )
            assert len(experience) == 4
            assert experience[0].shape == (2, 2)
            # Necessary to use allclose due to
            # float imprecision (float32/float64 comparison)
            np.testing.assert_allclose(experience[0][0, None], np.array([[101, 1.337]]))

    @staticmethod
    def test_load_meta(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1)
        writer.append(test_data)
        print(test_data)
        test_data["state"]["meta"][
            "met/a3"
        ] = "Thisisastringthatislongerthan10characters"
        writer.append(test_data)
        writer.notify_episode_stops()

        with SARTReader(os.path.join(temp_dir, os.listdir(temp_dir)[0])) as reader:
            meta = reader.load_meta()
            assert len(meta) == 3
            assert np.array_equal(meta["meta1"], np.array([1, 1]))
            assert np.array_equal(meta["meta2"], np.array([123, 123]))
            assert np.array_equal(
                meta["met/a3"], ["Start", "Thisisastringthatislongerthan10characters"]
            )

        with SARTReader(os.path.join(temp_dir, os.listdir(temp_dir)[0])) as reader:
            meta = reader.load_meta(meta_keys=("meta1",))
            assert len(meta) == 1
            assert np.array_equal(meta["meta1"], np.array([1, 1]))
            with pytest.raises(KeyError):
                _ = meta["meta2"]

    @staticmethod
    def test_read_during_write(temp_dir, test_data):
        writer = SARTWriter(temp_dir, "Test", 1, buffer_size=2)
        writer.append(test_data)
        time.sleep(0.025)
        # First read
        with SARTReader(os.path.join(temp_dir, os.listdir(temp_dir)[0])) as reader:
            experience = reader.load_full_episode()
            assert len(experience) == 4
            assert experience[0].shape == (1, len(MockState.channels()))
            # Necessary to use allclose due to
            # float imprecision (float32/float64 comparison)
            np.testing.assert_allclose(experience[0][0, None], data_array)
            assert all(experience[-1] == 1)
            assert all(experience[-2] == False)  # noqa E712
        writer.append(test_data)
        time.sleep(0.025)
        # Second read
        with SARTReader(os.path.join(temp_dir, os.listdir(temp_dir)[0])) as reader:
            experience = reader.load_full_episode()
            assert experience[0].shape == (2, len(MockState.channels()))
        writer.append(test_data)
        time.sleep(0.025)
        # Third read with dataset increase
        with SARTReader(os.path.join(temp_dir, os.listdir(temp_dir)[0])) as reader:
            experience = reader.load_full_episode()
            assert experience[0].shape == (3, len(MockState.channels()))
        writer.notify_episode_stops()


def test_nanlen():
    a = np.array([1, 2, 3, 4, 5, np.nan, np.nan])
    assert nanlen(a) == 5
    a = np.array([1, 2, 3, 4, 5, np.nan, 123])
    assert nanlen(a) == 5
    a = np.array([np.nan, 1, 2, 3, 4, 5, np.nan, 123])
    assert nanlen(a) == 0
    a = np.array([np.nan, np.nan, np.nan])
    assert nanlen(a) == 0
