# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""State/Action/Reward/Terminal log files.

Also provides a CLI to convert sart *.h5 files to csv.

.. autosummary::

    SARTWriter
    SARTLogger
    SARTReader
    ExpandableDataset
    sart_to_csv
    cli

"""

import logging
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from queue import Empty, Queue
from typing import Any, ClassVar, Dict, Iterable, Iterator
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from psipy import __version__ as psipy_version
from psipy.core.io.logging import CALENDAR
from psipy.core.threading_utils import StoppableThread
from psipy.core.utils import flatten, flatten_dict
from psipy.rl.core.cycle_manager import CM

__all__ = ["SARTLogger", "SARTWriter", "SARTReader"]


LOG = logging.getLogger(__name__)


def nanlen(seq: Sequence) -> int:
    """Returns the number of elements in sequence up to first `NaN` value.

    This method follows the naming of :meth:`numpy.nanmean` et al.

    Example::

        >>> nanlen(range(10))
        10
        >>> nanlen([1, 2, 3, float("nan")])
        3
        >>> nanlen([float("nan")])
        0

    Args:
        seq: Sequence to get the not-nan length for.
    """
    try:
        return np.where(np.isnan(seq))[0][0]
    except IndexError:  # no nan
        return len(seq)


class ExpandableDataset:
    """An HDF5 dataset that is automatically resized once it becomes full."""

    def __init__(
        self,
        hdf5_file: h5py.File,
        hdf_path: str,
        shape: Tuple,
        buffer_size: int,
        dtype: Union[np.dtype, str] = np.float32,
        is_string=False,
    ):
        self.rows = 0
        self.name = hdf_path
        self.buffersize = buffer_size
        self.current_buffer = buffer_size
        self.incoming_shape = shape
        self.dtype = dtype
        self.fill_val = np.nan
        if is_string:
            # HDF5 can not have a fillvalue when the dtype is string.
            # Therefore, it is set as None, and will default to "".
            self.dtype = h5py.string_dtype()
            self.fill_val = None
        
        self.dataset = hdf5_file.create_dataset(
            hdf_path,
            shape=(buffer_size, *shape),
            dtype=self.dtype,
            maxshape=(None, *shape),
            shuffle=True,
            fillvalue=self.fill_val,
        )

        # Now that the dataset is created, we can set the fill value for strings
        if self.fill_val is None:
            self.fill_val = ""

    def __len__(self):
        return self.rows

    def __repr__(self):
        return f"ExpandableDataset({self.name}:{self.__len__()}/{self.current_buffer})"

    def __del__(self):
        self.finalize()

    def _resize(self, rows: int) -> None:
        try:
            self.dataset.resize((rows, *self.incoming_shape))
        except ValueError as e:
            if not str(e).startswith("Not a dataset"):
                raise e

    def _maybe_expand(self) -> None:
        if self.rows == self.current_buffer:
            self.current_buffer += self.buffersize
            self._resize(self.current_buffer)
            LOG.debug("Expanded the writer buffer size")

    def add_row(self, data: Union[np.ndarray, str]) -> None:
        self._maybe_expand()
        if isinstance(data, str) and data == "default_fill":
            data = self.fill_val
        try:
            self.dataset[self.rows, ...] = data
            self.dataset.flush()
        except (OSError, TypeError) as e:
            LOG.warning(
                f"Attempted writing to {self.name} ({self.dtype}) with data: {data!r} "
                f"({type(data)})"
                f"\n\tError: {e}\n\tFilled with default fill value!"
            )
            self.dataset[self.rows, ...] = self.fill_val
        self.rows += 1

    def finalize(self) -> None:
        """Trim the unused rows from the dataset"""
        self._resize(self.rows)
        # Ensure that future writing will expand the buffer again
        self.current_buffer = self.rows

    @property
    def buffer_resizes(self) -> int:
        """Number of times the buffer resized"""
        return self.current_buffer // self.buffersize - 1

    @property
    def chunksize(self) -> Tuple[int, ...]:
        return self.dataset.chunks


class SARTWriter(StoppableThread):
    """HDF5 Writer for States, Actions, Costs, Terminals

    Holds a single hdf5 file on the top level containing groups for states,
    actions, and meta information.

    The writer expects its queue to be filled with an ordered dictionary containing
    the keys state, action, and meta (:meth:`append`). As information is put into
    the queue, it is extracted and saved into an hdf5 file within a separate
    thread (:meth:`run`), where each key in the queue'd dictionary becomes a group,
    except for the "leaf" keys, which become hdf5 datasets (:meth:`_create_groups`).
    Datasets start at a specified size and are expanded if their size limit is
    reached (:meth:`ExpandableDataset._maybe_expand`). Upon closing the writer,
    these datasets are resized to match their final length, dropping
    intermediate placeholder (nan) values (:meth:`notify_episode_stops` and
    :meth:`ExpandableDataset.finalize`).

    All written files written after SARTWriter version 2.0.0 have SWMR (single write,
    multiple read) capability, that is, they can be read while they are being written.
    Files written in SWMR mode cannot be opened with older versions of the HDF5
    library (basically any version older than the SWMR feature). This should not be an
    issue for us but it is good to keep in mind.

    The HDF5 file structure is defined as below:

    .. code-block:: bash

        File={
            state={
                values={
                    state_channel1=np.array where dim 0 = steps
                    ...
                }
                cost=np.array where dim 0 = steps
                terminals=np.array where dim 0 = steps
            }
            action={
                action_channel1=np.array where dim 0 = steps
                ...
            }
            meta={
                meta_info=np.array where dim 0 = steps;
                          dim 1 contains [timestamps, steps, sim_time]
            }
        }

    Usage note:

    - Versions < 2.0.0: If you keyboard interrupt a loop (SIGINT), as long as you don't
      terminate the process forcefully, the writer will finalize, close gracefully,
      and the collected data will be saved. There are no guarantees for non-corrupted
      data when SIGTERM'd.
    - Versions >= 2.0.0: Since the dataset is written after every :meth:`append`, the
      data up until the crash will NOT be corrupted, and can be read. Depending on the
      timing of the crash, data currently in the queue may also be written. What is not
      done during a crash is the 'finalization' of the dataset, i.e. the removal of the
      NaNs. This is dealt with in the :class:`SARTReader`.

    File will have the following naming scheme:
    ``filepath/{name}-{Optional[initial]}-{episode}-{filenum}-{creation_time}.h5``
    which follows the logging naming scheme.

    History and Lore: Ever wonder why it's called ``SART``? Back at inception, the
    writer would write "state, action, reward, terminal" values. Things eventually
    moved on to costs, but SACT doesn't sound as cool or rolls off the tongue as
    easily!

    Args:
        filepath: Path to save the file.
        name: File's unique project name.
        episode: Episode counter, for when running multiple consecutiev episodes
                 from the same loop.
        initial_time: Start time of outer loop.
        filenum: Additional counter to differentiate between multiple files
                 associated with the same episode.
        buffer_size: Size of internal datasets before increasing it by same amount.
    """

    _version: ClassVar[Tuple[int, int, int]] = (2, 0, 0)

    #: Datasets managed by the SARTWriter, each representing one column of data.
    expandable_datasets: Dict[str, ExpandableDataset]

    def __init__(
        self,
        filepath: str,
        name: str,
        episode: int,
        initial_time: Optional[datetime] = None,
        filenum: int = 1,
        buffer_size: int = 2000,
    ):
        super().__init__(name="sart-writer", daemon=True)
        os.makedirs(filepath, exist_ok=True)
        if initial_time is not None:
            creation_time = datetime.now().strftime("%y%m%d-%H")
            initial = initial_time.strftime("%y%m%d-%H%M%S")
            self.name = f"{name}-{initial}-{episode}-{filenum:02d}-{creation_time}.h5"
        else:
            creation_time = datetime.now().strftime("%y%m%d-%H%M%S")
            self.name = f"{name}-{creation_time}-{episode}-{filenum:02d}.h5"
        self.file = h5py.File(os.path.join(filepath, self.name), "a", libver="latest")

        self.file.attrs["task"] = name
        self.file.attrs["created"] = creation_time
        self.file.attrs["writer-version"] = self.__version__()
        self.file.attrs["psipyrl-version"] = psipy_version
        self.file.attrs["filenumber"] = filenum
        self.buffer = buffer_size
        LOG.debug("New SART HDF5 opened.")

        self.initialized = False
        self.inserts = 0
        self.sart_queue: Queue = Queue()
        self.expandable_datasets = OrderedDict()

        self.start()

    def __len__(self):
        return self.inserts

    def __str__(self):
        return (
            f"{self.name}-SART(len={self.__len__()}|queue=~{self.sart_queue.qsize()})"
        )

    @classmethod
    def __version__(self):
        return ".".join(map(str, self._version))

    def __del__(self):
        self.stop()

    def _create_groups(self, sart: Dict) -> None:
        """Creates expandable datasets for future data and saves their input order"""
        for path, value in flatten_dict(sart, sep="/", sanitizer="|").items():
            dtype = np.float64 if path.startswith("meta") else np.float32
            shape = np.asarray(value).shape
            is_string = True if isinstance(value, str) else False
            self.expandable_datasets[path] = ExpandableDataset(
                self.file,
                path,
                shape,
                self.buffer,
                dtype=dtype,
                is_string=is_string,
            )
        self.file.attrs["state"] = tuple(sart["state"]["values"].keys())
        self.file.attrs["action"] = tuple(sart["action"].keys())
        self.file.attrs["meta"] = tuple(sart["meta"].keys())

        # Enable single write, multiple read mode
        # Once SWMR is active, no more keys can be made
        self.file.swmr_mode = True

    def append(self, sart: Dict) -> None:
        """Insert dict into queue for later logging; keys=[state, action, meta]

        Values to be written must be singular, i.e. not in a list or array!
        """
        # TODO: Drop the following two lines by properly refactoring this class
        #       to expect `meta` as attribute of state. Watch out for the risk
        #       of mutating the passed-by-reference sart dictionary.
        sart = deepcopy(sart)
        sart["meta"] = sart["state"].pop("meta")
        assert set(sart.keys()) == {"action", "state", "meta"}
        assert isinstance(sart["action"], dict)
        assert isinstance(sart["state"]["values"], OrderedDict)
        # Not blocking forever in order to allow shutdown at some point if
        # something went bad. Queue actually has no size limit, just a failsafe.
        self.sart_queue.put(sart, timeout=1)

    def run(self) -> None:
        """Take items from the queue and save into the hdf5 file in its own thread."""
        while True:
            try:
                # Only block for one second in order to again check
                # whether the thread has been stopped.
                sart = self.sart_queue.get(timeout=0.5)
            except Empty:
                if self.stopped():
                    break
                continue

            if not self.initialized:  # Create the datasets with the proper shapes
                self._create_groups(sart)
                self.initialized = True

            with CM["sart-write"]:
                sart = flatten_dict(sart, sep="/", sanitizer="|")
                for path in self.expandable_datasets.keys():
                    value = sart.get(path, "default_fill")
                    self.expandable_datasets[path].add_row(value)

            self.inserts += 1
            self.sart_queue.task_done()

        # Once the loop above finishes, cleanup.
        for dataset in self.expandable_datasets.values():
            dataset.finalize()
        self.file.close()

    def notify_episode_stops(self) -> None:
        if not self.stopped():
            self.stop()
            self.join(2)
            LOG.debug("SART HDF5 closed at end of episode.")


class SARTLogger:
    """:class:`SARTLogger` handles the creation and rollover of :class:`SARTWriter`s.

    A single :class:`SARTLogger` instance is managed by the
    :class:`~psipy.rl.loop.Loop` to handle SART logfiles. SART logfiles are
    commonly stored on a per-episode basis, but in production might in addition
    be split depending on calendar date. Given the situation,
    :class:`SARTLogger` and  :class:`~psipy.rl.loop.Loop` can also be configured
    to only store a single SART file for many episodes. Note that this might be
    sub-ideal as those SART files are intended for the training of RL-based
    controllers.

    Args:
        filepath: Path to save the file.
        name: File's unique project name.
        initial_time: Start time of outer loop.
        rollover: At what point to rollover *.h5 files.
                  One of ``["d", "w", "m", "y"]``.
        single: Whether to store all episodes into a single chain of SART files,
                without introducing per-episode rollovers.
    """

    _version: ClassVar[Tuple[int, int, int]] = (2, 0, 1)

    #: Counter of the number of episodes seen. When running in ``single`` mode
    #: (see init args), this value will not be increasing!
    episode_count: int

    #: Number of files created for the current episode.
    episode_file_count: int

    #: Counter of the number of files created.
    file_count: int

    #: Number of samples seen over all episodes.
    sample_count: int

    #: :class:`SARTWriter` instance.
    writer: SARTWriter

    def __init__(
        self,
        filepath: str,
        name: str,
        initial_time: Optional[datetime] = None,
        rollover: Optional[str] = None,
        single: bool = False,
    ):
        super().__init__()
        self.filepath = filepath
        self.name = name
        self.initial_time = initial_time
        self.rollover = rollover
        self.single = single

        self.file_count = 0
        self.episode_count = 0
        self.episode_file_count = 0
        self.sample_count = 0
        #self.file_rollover_date = None

    def __del__(self):
        self.shutdown()

    def create_writer(self) -> SARTWriter:
        """Creates a properly named instance of :class:`SARTWriter`."""
        return SARTWriter(
            self.filepath,
            self.name,
            self.episode_count - 1,
            self.initial_time,
            filenum=self.episode_file_count - 1,
        )

    def get_now(self) -> int:
        """Gets :meth:`datetime.now`'s representation in :attr:`rollover`'s format.

        - In case :attr:`rollover` is ``None``, this method returns ``None`` and the
          appended data will be stored in a single file.
        - In other cases, this method returns an integer representation of the
          current date, e.g. 15 for ``self.rollover == 'h'`` at 3pm current hour.
        """
        if self.rollover is None:
            return -1
        return int(datetime.now().strftime(CALENDAR[self.rollover.lower()]))

    def notify_episode_starts(self) -> None:
        """Creates an episodes initial :class:`SARTWriter` and tracking values."""
        if self.single and hasattr(self, "writer"):
            # Do not create a new writer etc if `single` is true and a writer
            # was already created previously.
            return
        self.file_count += 1
        self.episode_count += 1
        self.episode_file_count = 1
        self.sample_count = 0
        self.writer = self.create_writer()
        self.file_rollover_date = self.get_now()

    def do_rollover(self) -> None:
        """Rolls sart logs over into a new file."""
        LOG.info(f"SART rolling over, stopping {self.writer.name}.")
        self.writer.stop()  # stops, non-blocking
        self.file_count += 1
        self.episode_file_count += 1
        self.writer = self.create_writer()
        LOG.info(f"SART rolled over, starting {self.writer.name}.")
        self.file_rollover_date = self.get_now()

    def append(self, sart: Dict[str, Any]) -> None:
        """Appends data to existing :class:`SARTWriter` and maybe creates a new one.

        Args:
            sart: SART data to write in h5 file
        """
        if self.file_rollover_date != self.get_now():
            self.do_rollover()
        self.writer.append(sart)
        self.sample_count += 1

    def notify_episode_stops(self) -> None:
        if not self.single:
            try:
                # stops and joins, blocking
                self.writer.notify_episode_stops()
            except AttributeError:
                pass

    def shutdown(self) -> None:
        try:
            # stops and joins, blocking
            self.writer.notify_episode_stops()
        except AttributeError:
            pass


class SARTReader:
    """HDF5 SART file reader.

    The reader has the ability to read a currently live (open) SART file. Usage
    in this case is as normal, i.e. there is nothing special to do.

    In the case that a file is read during writing, or crashed during writing, the
    reader will trim away the trailing NaNs/placeholder values that were not trimmed
    away due to the dataset not finalizing.
    """

    def __init__(self, filepath: str):
        try:
            self.file = h5py.File(filepath, "r", libver="latest", swmr=True)
        except OSError as e:
            if "No such file or directory" in e.args[0]:  # have more informative error
                raise OSError(f"Unable to open file (does it exist?): {filepath}")
            raise e
        LOG.debug(f"SART Reader opened `{filepath}` in SWMR mode.")
        self.paths = self._traverse_h5keys(self.file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _traverse_h5keys(self, h5_file) -> List[str]:
        """Recursively builds a list of paths of h5files's keys."""

        def traverse(h5_file, path: str) -> Iterator[str]:
            """Traverses down to the nodes of a top level key."""
            if not isinstance(h5_file[path], h5py.Dataset):
                for key in h5_file[path].keys():
                    for more in traverse(h5_file, f"{path}/{key}"):
                        yield more
            else:
                yield path

        return flatten(list(traverse(h5_file, key)) for key in h5_file.keys())

    def refresh(self):
        """Refreshes the open file."""
        for dataset in self.paths:
            self.file[dataset].id.refresh()

    def __len__(self) -> int:
        """Returns the number of samples after trimming of NaNs."""
        state_channels = self.file.attrs["state"][:]
        state_channels = [chan.replace("/", "|") for chan in state_channels]
        chan = state_channels[0]
        data = self.file[f"state/values/{chan.replace('/', '|')}"][:]
        return nanlen(data)

    def load_full_episode(
        self,
        state_channels: Optional[Iterable[str]] = None,
        action_channels: Optional[Iterable[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads full episode into (obs, actions, terminals, costs) tuple.

        Only loads specific channels if specified.

        NOTE: This breaks with states that involves mixed shapes, i.e. a state
              in which one channel is an image and the other is a vector.  These
              can not be coerced into one numpy array, and thus will fail.
        """

        self.refresh()
        if state_channels is None:
            state_channels = self.file.attrs["state"][:]
        state_channels = [chan.replace("/", "|") for chan in state_channels]
        if action_channels is None:
            action_channels = self.file.attrs["action"][:]
        action_channels = [chan.replace("/", "|") for chan in action_channels]

        n = len(self)

        state = [
            self.file[f"state/values/{chan.replace('/', '|')}"][:n]
            for chan in state_channels
        ]
        action = [
            self.file[f"action/{chan.replace('/', '|')}"][:n]
            for chan in action_channels
        ]
        cost = self.file["state/cost"][:n]
        terminal = self.file["state/terminal"][:n]
        return np.array(state).T, np.array(action).T, terminal, cost

    def load_meta(self, meta_keys: Optional[Iterable[str]] = None) -> Dict:
        """Loads meta information, only specific meta keys if specified."""
        if meta_keys is None:
            meta_keys = self.file.attrs["meta"][:]

        n = len(self)
        meta = {}
        for chan in meta_keys:
            try:
                meta[chan] = self.file[f"meta/{chan.replace('/', '|')}"][:n]
            except OSError:
                LOG.warning(f"Channel {chan} is not readable! Skipping...")

        return meta

    def close(self) -> None:
        try:
            self.file.close()
            LOG.debug("SART Reader successfully closed all files.")
        except (AttributeError, TypeError, ValueError):
            # ValueError might occur from logger writing to an already closed logfile.
            pass

    def __del__(self):
        self.close()


def sart_to_csv(filepath: str, targetdir: Optional[str] = None):
    """Convert SART *.h5 files to databrowser compatible csvs."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("sart_to_pdf reuqires the optional dependency pandas.")

    with SARTReader(filepath) as sart:
        obs_arr, actions, terminals, costs = sart.load_full_episode()
        obs_keys = sart.file.attrs["state"][:]
        obs = {key: values for key, values in zip(obs_keys, obs_arr.T)}
        meta = sart.load_meta()
        # action_keys = sart.file.attrs["action"][:]

    if "timestamp" not in obs and "timestamp" in meta:
        # timestamps = meta["timestamp"]
        # Following assert currently does not hold for thickener sart data.
        # assert all(b > a for a, b in zip(timestamps[:1], timestamps[1:]))
        obs["timestamp"] = np.arange(len(obs[obs_keys[0]]))

    df = pd.DataFrame(obs)

    if targetdir is None:
        targetdir = os.path.dirname(filepath)
    targetname, _ = os.path.splitext(os.path.basename(filepath))
    targetpath = os.path.join(targetdir, f"{targetname}.csv")

    os.makedirs(targetdir, exist_ok=True)
    df.to_csv(targetpath, sep=";", index=False)
    LOG.info(f"{filepath} -> {targetpath} done.")


def cli():
    """SART *.h5 to DataBrowser compatible *.csv conversion.

    Usage::

        python -m psipy.rl.io.sart --help

    """
    import sys
    from argparse import ArgumentParser

    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(logging.StreamHandler(sys.stdout))

    parser = ArgumentParser(
        description="SART *.h5 to DataBrowser compatible *.csv conversion."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="SART hdf5 files or directories to convert.",
    )
    parser.add_argument(
        "--targetdir", type=str, help="Target directory to save the csv files to."
    )
    args = parser.parse_args()

    for path in args.paths:
        if os.path.isdir(path):
            filenames = os.listdir(path)
            filepaths = [os.path.join(path, filename) for filename in filenames]
        else:
            filepaths = [path]
        for filepath in filepaths:
            sart_to_csv(filepath, args.targetdir)


if __name__ == "__main__":
    cli()
