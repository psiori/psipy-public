# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""MemoryZipFile provides convenient methods for adding objects to a zipfile.

    Based on python zipFile that can be used both with filepath or buffer.

    Supported data types for adding to the zip file:

    - List/tuple/dict of purely python str/float/int objects.
    - Dict of purely numpy arrays.
    - Single numpy array.
    - Single keras model.

"""

import inspect
import io
import logging
import os
from tempfile import TemporaryDirectory
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing.io import BinaryIO
from zipfile import ZIP_DEFLATED, ZipFile

import h5py
import numpy as np
import tensorflow as tf
import keras

# from psipy.core.io.saveable import TSaveable
from psipy.core.io.json import JSONEncodable, json_check, json_decode, json_encode
from psipy.core.io.zip import add_to_zip, path_join
from psipy.core.utils import flatten, index_or_none

# Optional dependencies.
try:
    import pandas as pd
except ImportError:
    pd = None


__all__ = ["MemoryZipFile", "json_encode", "json_decode"]


LOG = logging.getLogger(__name__)


class MemoryZipFile:
    """In-memory wrapper of :class:`ZipFile` with a lot of added convenience.

    Args:
        filepath_or_bytes: Optionally specifies a filepath where to save the
            :class:`MemoryZipFile` to disk. If not specified, it lives, as the
            name says, purely in memory. One can also pass raw bytes directly
            to instantiate from in-memory data only.

    """

    #: Class version, semver compatible.
    _version: ClassVar[Tuple[int, int, int]] = (1, 1, 3)

    #: Path to where the ZipFile is (to be) stored on disk.
    _filepath: Optional[str]

    #: The underlying bytes buffer.
    _buffer: io.BytesIO

    #: The underlying python ZipFile.
    _zipfile: ZipFile

    #: Whether the :attr:`_zipfile` is currently closed. Note that opening and
    #: closing is handled by this class automatically behind the scenes, so
    #: there should be no need to access this attribute directly from the
    #: outside.
    _closed: bool

    def __init__(self, filepath_or_bytes: Optional[Union[str, bytes]] = None):
        self._filepath = None
        self._buffer = io.BytesIO()

        if isinstance(filepath_or_bytes, str):
            self._filepath = filepath_or_bytes
            # Load existing mzp from disk.
            if os.path.exists(filepath_or_bytes):
                with open(filepath_or_bytes, "rb") as f:
                    self._buffer = io.BytesIO(f.read())

        # Create mzp from memory.
        if isinstance(filepath_or_bytes, bytes):
            self._buffer = io.BytesIO(filepath_or_bytes)

        self._closed = True  # closed initially
        self._maybe_open()
        self._cwd = ""

        self.namelist = self._zipfile.namelist
        self.close = self._zipfile.close

    def add(self, targetpath: str, data: Any):
        """Add file to zip at targetpath."""
        ext = self.infer_ext(data)
        if isinstance(data, tf.keras.Model):
            return self.add_keras(targetpath, data)
        if isinstance(data, np.ndarray):
            return self.add_numpy(targetpath, data)
        if ext == "json":
            # Clean python std only structure.
            return self.add_json(targetpath, data)
        if isinstance(data, (list, tuple, dict)):
            flat = flatten(data)
            if isinstance(data, dict):
                vals = data.values()
                if ext == "npz":
                    # Numpy only dict.
                    return self.add_numpy_dict(targetpath, data)
                if all((isinstance(v, np.ndarray) or json_check(v)) for v in vals):
                    # Python std + numpy mixed dict.
                    return self.add_mixed_dict(targetpath, data)
            if isinstance(data, (list, tuple)):
                if all(isinstance(v, np.ndarray) for v in data):
                    # Numpy only list.
                    return self.add_numpy_list(targetpath, data)
                if all((isinstance(v, np.ndarray) or json_check(v)) for v in flat):
                    # Python std + numpy mixed list.
                    return self.add_mixed_list(targetpath, data)
        if isinstance(data, io.BytesIO):
            return self.add_raw(targetpath, data.getvalue())
        if isinstance(data, (str, bytes)):
            return self.add_raw(targetpath, data)
        raise ValueError(
            f"Cannot determine how to add passed data. "
            f"Given data={data} ({type(data)})"
        )

    def get(self, filepath: str):
        _, ext = os.path.splitext(filepath)
        if ext == ".json":
            return self.get_json(filepath)
        if ext == ".h5":
            raise ValueError(
                "Cannot get .h5 contents. Use specialized get_* methods instead."
            )
        if ext in (".npy", ".npz"):
            return self.get_numpy(filepath)
        if ext == ".csv":
            return self.get_csv(filepath)
        if not ext:
            return self.get_mixed(filepath)
        raise ValueError("Unknown file extension.")

    def _maybe_open(self):
        if self._closed:
            self._zipfile = ZipFile(self._buffer, "a", ZIP_DEFLATED, False)
            self._closed = False

    def add_raw(self, targetpath: str, data: Union[str, bytes]):
        self._maybe_open()
        targetpath = path_join(self._cwd, targetpath)
        self._zipfile.writestr(targetpath, data)
        return self

    def add_json(self, targetpath: str, data: Dict[str, Any]):
        assert json_check(data)
        assert targetpath.endswith(".json")
        return self.add_raw(targetpath, json_encode(data))

    def get_json(self, filepath: str) -> JSONEncodable:
        self.check_exists(filepath)
        with self.open(filepath) as f:
            data = json_decode(f.read().decode("utf-8"))
        return data

    def add_meta(self, **kwargs):
        assert "__meta__.json" not in self.ls()
        return self.add_json("__meta__.json", kwargs)

    def get_meta(self, dirname: str = ""):
        if "__meta__.json" not in self.ls(dirname):
            return {}
        return self.get_json(path_join(dirname, "__meta__.json"))

    def add_numpy(self, targetpath: str, arr: np.ndarray):
        assert targetpath.endswith(".npy")
        data = io.BytesIO()
        np.save(data, arr)
        return self.add_raw(targetpath, data.getvalue())

    def get_numpy(self, filepath: str) -> np.ndarray:
        self.check_exists(filepath)
        data = np.load(self.get_bytesio(filepath))
        if isinstance(data, np.lib.npyio.NpzFile):
            data = {key: data[key] for key in data.files}
        return data

    def add_numpy_dict(self, targetpath: str, arrs: Dict[str, np.ndarray]):
        assert all(isinstance(v, np.ndarray) for v in flatten(arrs))
        assert targetpath.endswith(".npz")
        data = io.BytesIO()
        np.savez(data, **arrs)
        return self.add_raw(targetpath, data.getvalue())

    def add_mixed_dict(
        self, targetpath: str, data: Dict[str, Union[np.ndarray, str, float, int, None]]
    ):
        """Add dictionary of mixed types to the zipfile.

        Currently supports flat dictionaries made up of the following types:

        - ``int``/``float``/``str``/``None``
        - ``np.ndarray``

        """
        if os.path.splitext(targetpath)[1]:
            raise ValueError("Mixed dicts cannot have a specific file extension.")
        json_data = {k: v for k, v in data.items() if json_check(v)}
        numpy_data = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
        basename = os.path.basename(targetpath)
        wd = self.cwd()
        self.cd(targetpath)
        self.add_meta(dtype="mixed_dict")
        if json_data:
            self.add_json(f"{basename}.json", json_data)
        if numpy_data:
            self.add_numpy_dict(f"{basename}.npz", numpy_data)
        self.cd(wd)
        return self

    def get_mixed_dict(self, dirpath: str) -> Dict[str, Any]:
        """Get dictionary of mixed types from the zipfile.

        .. note::
            Currently only supports flat structures.
        """
        if os.path.splitext(dirpath)[1]:
            raise ValueError("Mixed dicts cannot have a specific file extension.")
        data: Dict[str, Any] = dict()
        wd = self.cwd()
        self.cd(dirpath)
        for filepath in self.ls(include_directories=True):
            if filepath == "__meta__.json":
                continue
            if len(filepath[len(dirpath) :].split("/")) > 2:
                raise NotImplementedError("Cannot parse nested mixed dicts yet.")
            data = {**data, **self.get(filepath)}
        self.cd(wd)
        return data

    def add_mixed_list(self, targetpath, mixed):
        wd = self.cwd()
        self.cd(targetpath)
        dtype = "mixed_tuple" if isinstance(mixed, tuple) else "mixed_list"
        self.add_meta(dtype=dtype, count=len(mixed))
        for i, item in enumerate(mixed):
            ext = self.infer_ext(item)
            itempath = str(i).zfill(len(str(len(mixed))))
            if ext is not None:
                itempath = f"{itempath}.{ext}"
            self.add(itempath, item)
        self.cd(wd)
        return self

    def get_mixed_list(self, dirpath: str):
        if os.path.splitext(dirpath)[1]:
            raise ValueError("Mixed lists cannot have a specific file extension.")
        wd = self.cwd()
        self.cd(dirpath)
        meta = self.get_meta()
        data = []
        for itempath in self.ls(include_directories=True):
            if itempath == "__meta__.json":
                continue
            data.append(self.get(itempath))
        self.cd(wd)
        if meta["dtype"] == "mixed_tuple":
            return tuple(data)
        return data

    def get_mixed(self, dirpath: str):
        meta = self.get_meta(dirpath)
        if "dtype" not in meta or meta["dtype"] == "mixed_dict":
            return self.get_mixed_dict(dirpath)
        if meta["dtype"] in ("mixed_tuple", "mixed_list"):
            return self.get_mixed_list(dirpath)
        if meta["dtype"] in ("numpy_list", "numpy_tuple"):
            return self.get_numpy_list(dirpath)

    def add_csv(
        self,
        targetpath: str,
        data: Union["pd.DataFrame", np.ndarray],
        columns: Optional[List] = None,
    ) -> "MemoryZipFile":
        """Add a numpy array or pandas dataframe as a csv to the zipfile.

        Args:
            targetpath: Targetpath within the zipfile to write the csv to.
            data: Data to convert to csv.
            columns: Optional column names to use for numpy arrays.
        """
        buf = io.StringIO()
        if pd is not None and isinstance(data, pd.DataFrame):
            if columns is None:
                columns = list(data.columns)
            data.to_csv(buf, index=False, header=True, columns=columns)
        else:
            if columns is None:
                columns = [str(i) for i in range(data.shape[1])]
            assert data.shape[1] == len(columns)
            buf.write(f"{','.join(columns)}\n")
            np.savetxt(buf, data, delimiter=",")
        return self.add_raw(targetpath, buf.getvalue())

    def get_csv(self, filepath: str, high_precision=False) -> "pd.DataFrame":
        """Get contents from csv file.

        Setting ``high_precision`` to True is significantly slower than using
        the default which looses some precision in the reading process.
        """
        float_precision = "round_trip" if high_precision is True else None
        return pd.read_csv(self.get_bytesio(filepath), float_precision=float_precision)

    def add_numpy_list(
        self, targetpath: str, arrs: Union[Tuple[np.ndarray, ...], List[np.ndarray]]
    ):
        assert all(isinstance(v, np.ndarray) for v in flatten(arrs))
        wd = self.cwd()
        self.cd(targetpath)
        dtype = "numpy_list" if isinstance(arrs, list) else "numpy_tuple"
        self.add_meta(dtype=dtype, count=len(arrs))
        npzpath = f"{targetpath}.npz"
        keys = [str(i) for i in range(len(arrs))]
        self.add_numpy_dict(npzpath, dict(zip(keys, arrs)))
        self.cd(wd)
        return self

    def get_numpy_list(self, dirpath: str):
        wd = self.cwd()
        self.cd(dirpath)
        meta = self.get_meta()
        npz = self.get_numpy(f"{os.path.basename(dirpath)}.npz")
        data = [npz[key] for key in sorted(npz.keys(), key=int)]
        self.cd(wd)
        if meta["dtype"] == "numpy_tuple":
            return tuple(data)
        return data

    def get_hdf5(
        self, filepath: str, custom_objects: Optional[List[Callable]] = None
    ) -> h5py.File:
        self.check_exists(filepath)
        return h5py.File(self.get_bytesio(filepath), "r")

    def add_keras(self, targetpath: str, model: tf.keras.Model):
        """Add a :mod:`tensorflow.keras` model.

        .. todo::
            Maybe unify this method with :meth:`add_tf`, adding a new ``format``
            keyword argument similar to the underlying ``save_model`` method?

        Args:
            targetpath: Location to add the model inside the zipfile.
            model: Model instance to store.
        """
        self._maybe_open()
        with TemporaryDirectory() as tmpdir:
            tmppath = path_join(tmpdir, targetpath)
            keras.saving.save_model(model, tmppath)
            self.add_files(targetpath, tmppath)
        return self

 
    def get_keras(
        self, filepath: str, custom_objects: Optional[Union[List, Dict]] = None, support_legacy: bool = True
    ) -> tf.keras.Model:
        """Get a keras model from the zipfile at filepath.

        Args:
            filepath: Relative to the zipfile's *current working directory*.
            custom_objects: May contain a list of objects (for example custom
                layers) used by the loaded model but not native to Keras.
        """

        if support_legacy and filepath.endswith(".keras"):
            legacy_filepath = filepath[:-6] + ".h5"

            if self.exists(legacy_filepath):
                filepath = legacy_filepath

        self.check_exists(filepath)  # will throw an error if the file does not exist

        object_map: Optional[Dict[str, Callable]] = None
        if isinstance(custom_objects, dict):
            object_map = custom_objects
        elif isinstance(custom_objects, list):
            object_map = {inspect.unwrap(co).__name__: co for co in custom_objects}

        with TemporaryDirectory() as tmpdir:
            files = self.ls("/", abs=True, recursive=True, include_directories=False)

            filepath = path_join(self._cwd, filepath)
            if filepath[0] == "/":  # No 'root' slash in zipfiles.
                filepath = filepath[1:]

            self._zipfile.extractall(path=tmpdir, members=[filepath])
            abspath = path_join(tmpdir, filepath)

            model = keras.saving.load_model(abspath, custom_objects=object_map, compile=False)

        # KERAS 3 presently (2025-08-22) does NOT preserve the models input and # output layer types; it wraps tensors in a list, if they were not 
        # already.
        # this seems to be "by intetntion" or at least not likely to change
        # quickly: https://github.com/keras-team/keras/issues/19999
        # so we need to unwrap the tensors here.
        # so annoying that the new distribution violates basic contracts 
        # like model == model.save().load() :(

        return model

    def add_files(self, targetpath: str, *sourcepaths: str):
        """Add files from the filesystem.

        Args:
            targetpath: Location to add the files inside the zipfile.
            sourcepaths: Paths to files or directories to add.
        """
        self._maybe_open()
        targetpath = path_join(self._cwd, targetpath)
        for sourcepath in sourcepaths:
            add_to_zip(self._zipfile, sourcepath, targetpath)
        return self

    def add_tf(self, targetpath: str, model: tf.keras.Model):
        """Add :mod:`tensorflow.keras` model in SavedModel format.

        Args:
            targetpath: Location to add the model inside the zipfile.
            model: Model instance to store.
        """
        self._maybe_open()
        if os.path.splitext(targetpath)[1] != "":
            raise ValueError("SavedModel expects directory.")
        with TemporaryDirectory() as tmpdir:
            model.save(tmpdir, save_format="tf")
            self.add_files(targetpath, tmpdir)
        return self

    def get_tf(
        self,
        path: str,
        custom_objects: Optional[Union[List, Dict]] = None,
    ) -> tf.keras.Model:
        """Get :mod:`tensorflow.keras` model.

        Args:
            path: SavedModel path relative to the zipfile's *current working directory*.
            custom_objects: May contain a list of objects (for example custom
                layers) used by the loaded model but not native to Keras.
        """
        if os.path.splitext(path)[1] != "":
            raise ValueError("SavedModel expects directory.")
        path = path if path[-1] == "/" else f"{path}/"
        self.check_exists(path)
        object_map: Optional[Dict[str, Callable]] = None
        if isinstance(custom_objects, dict):
            object_map = custom_objects
        elif isinstance(custom_objects, list):
            object_map = {inspect.unwrap(co).__name__: co for co in custom_objects}
        with TemporaryDirectory() as tmpdir:
            files = self.ls(path, abs=True, recursive=True, include_directories=False)
            self._zipfile.extractall(path=tmpdir, members=files)
            abspath = path_join(tmpdir, self._cwd, path)
            return tf.keras.models.load_model(abspath, custom_objects=object_map)

    def get_bytesio(self, filepath: str) -> BinaryIO:
        self.check_exists(filepath)
        return io.BytesIO(self.read(filepath))

    def save(self, filepath: Optional[str] = None):
        """Close zipfile and write zip to disk."""
        # Use old filepath or use and store new filepath.
        if filepath is None:
            filepath = self._filepath
        self._filepath = filepath

        if filepath is None:
            raise ValueError("No filepath provided.")

        dirpath = os.path.dirname(filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open(filepath, "wb") as f:
            f.write(self.getvalue())

        return self

    def cd(self, *dirnames: str):
        """Change working directory inside zip."""
        if any(os.path.splitext(dirname)[1] for dirname in dirnames):
            raise ValueError("Directories may not have an extension.")
        cwd = "/"
        if dirnames[0][0] != "/":
            cwd = self._cwd
        cwd = path_join(cwd, *dirnames)
        cwd = os.path.normpath(cwd)
        cwd = path_join(cwd, "").lstrip("./")
        self._cwd = cwd
        return self

    def cwd(self, leading_slash=True) -> str:
        """Get current working directory inside zip."""
        if leading_slash and (not self._cwd or self._cwd[0] != "/"):
            return f"/{self._cwd}"
        return self._cwd

    def getvalue(self) -> bytes:
        """Close zipfile and return bytes."""
        self._zipfile.close()
        self._closed = True
        return self._buffer.getvalue()

    def read(self, filepath: str) -> bytes:
        """Read file contained in zip.

        Args:
            filepath: File's location inside the zip.
        """
        filepath = path_join(self._cwd, filepath)
        return self._zipfile.read(filepath)

    def open(self, filepath: str) -> BinaryIO:
        """Open file contained in zip.

        Args:
            filepath: File's location iside the zip.
        """
        filepath = path_join(self._cwd, filepath)
        # No 'root' slash in zipfiles.
        if filepath[0] == "/":
            filepath = filepath[1:]
        return self._zipfile.open(filepath)

    def ls(
        self,
        *dirpaths: str,
        abs: bool = False,
        recursive: bool = False,
        include_directories: bool = False,
    ) -> List[str]:
        """List files in current or specific directories.

        Args:
            *dirpaths: Directories to list containing files for.
            abs: Whether to return absolute paths.
            recursive: Whether to also list files in subdirectories.
        """
        if not dirpaths:
            dirpaths = ("",)
        dirpath = path_join(*dirpaths)
        if os.path.splitext(dirpath)[1]:
            raise ValueError("Directories may not have an extension.")
        if dirpath.startswith("/"):
            cwd = dirpath[1:]
            qualified = path_join(cwd, "")
        else:
            cwd = self.cwd(leading_slash=False)
            qualified = path_join(cwd, dirpath, "")
        # Get paths which start with current path.
        files = [v for v in self.namelist() if v.startswith(qualified)]
        # Strip current path from all of them.
        files = [v[len(qualified) :] for v in files]
        if not recursive:
            # Cut everything off at first slash
            files = list(set([v[: index_or_none(v, "/", 1)] for v in files]))
        if include_directories:
            # Extract all intermediate directories.
            directories = list(set([v[: v.rindex("/") + 1] for v in files if "/" in v]))
            files += [
                directory[: ix + 1]
                for directory in directories
                for ix in range(len(directory))
                if directory[ix] == "/"
            ]
        else:
            # Drop directories.
            files = [v for v in files if not v.endswith("/")]
        if abs:
            # Maybe re-add the current directory's path to get absolute paths.
            files = [path_join(qualified, file) for file in files]
        return sorted(list(set(files)))

    def abspath(self, path: str, leading_slash=True) -> str:
        cwd = self.cwd(leading_slash=leading_slash)
        return path_join(cwd, path)

    def exists(self, path: str) -> bool:
        """Check whether given path exists in zipfile either as file or directory.

        Args:
            path: Path to check for existence.
        """
        path = path_join(self._cwd, path)
        if path[0] == "/":  # No 'root' slash in zipfiles.
            path = path[1:]
        return path in self.ls("/", recursive=True, include_directories=True)

    def check_exists(self, path: str) -> None:
        if not self.exists(path):
            ls = self.ls("/", recursive=True, include_directories=True)
            raise FileNotFoundError(
                f"`{path}` does not exist in current zip directory {self._cwd}. "
                "Note that directories need to be specified using a trailing slash. "
                f"Contents: {ls}"
            )

    def infer_ext(self, data):
        if isinstance(data, tf.keras.Model):
            return "h5"
        if isinstance(data, np.ndarray):
            return "npy"
        flat = flatten(data)
        if all(v is None or isinstance(v, (str, float, int)) for v in flat):
            # Clean python std only structure.
            return "json"
        if isinstance(data, dict):
            if all(isinstance(v, np.ndarray) for v in flat):
                # Numpy only dict.
                return "npz"
        return None

    # WIP. The following lays the foundation for a possible way to load an unknown
    # class from a given list of custom objects. Useful when the exact type of the
    # saved class is unknown during development, but can be narrowed down by the
    # user through a list of passed class types during runtime.
    # def get_custom(
    #     self, dirpath: str, custom_objects: Optional[Union[List, Dict]] = None
    # ) -> TSaveable:
    #     object_map: Dict[str, TSaveable] = dict()
    #     if isinstance(custom_objects, dict):
    #         object_map = custom_objects
    #     elif isinstance(custom_objects, list):
    #         object_map = {inspect.unwrap(co).__name__: co for co in custom_objects}
    #     meta = self.get_meta(dirpath)
    #     if meta.class_name not in object_map:
    #         raise ValueError(
    #             f"Could not load {dirpath} from zip, as no matching custom object "
    #             f"was provided. Was expecting {meta.class_module}.{meta.class_name}"
    #         )
    #     cwd = self.cwd()
    #     self.cd(dirpath)
    #     obj = object_map[meta.class_name].load(self)
    #     self.cd(cwd)
    #     return obj
