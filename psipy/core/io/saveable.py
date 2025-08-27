# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Saveable, a flexible baseclass for saveable python objects.

Core functionality for saving and loading classes. Focus on flexibility: All
different types of models should be saveable in a unified way, no matter
which library was employed when creating them.

Use this class as a mixin or base class. Make sure to call :meth:`update_config` in the :meth:`__init__` method of the subclass to pass parameters that should be stored AFTER initialization (e.g. after super().__init__() has been called). Complex member types (e.g. classes) should be handled by overwriting :meth:`_save` and :meth:`_load` to handle the serialization and deserialization of the complex data types.

Most end-user functionality is provided through
:mod:`~psipy.core.io.memory_zip_file`. See that documentation for further details on what data can be stored inside a :class:`Saveable`.

Also refer to :mod:`psipy.core.io.saveable_sklearn`.

Example:

    >>> class Model1(Saveable):
    ...     def __init__(self, arg1=1, arg2=2):
    ...         super().__init__()
    ...         self.update_config(arg1=arg1, arg2=arg2)  # call update_config 
    ...         # to pass parameters that should be stored AFTER initialization 
    ...         # (e.g. after super().__init__() has been called)
    ...
    ...     def _save(self, zipfile):
    ...         return zipfile.add("config.json", self.get_config())
    ...
    ...     @classmethod
    ...     def _load(cls, zipfile):
    ...         config = zipfile.get("config.json")
    ...         return cls.from_config(config)
    >>>
    >>> class Model2(Saveable):
    ...     def __init__(self, value=None):
    ...         super().__init__()
    ...         self.update_config(value=value)
    ...         self._submodel = Model1()
    ...         self.value = value
    ...
    ...     @property
    ...     def value(self):
    ...         return self._value
    ...
    ...     @value.setter
    ...     def value(self, value):
    ...         self.update_config(value=value)
    ...         self._value = value
    ...
    ...     def _save(self, zipfile):
    ...         zipfile.add("config.json", self.get_config())
    ...         zipfile = self._submodel.save(zipfile)
    ...         return zipfile
    ...
    ...     @classmethod
    ...     def _load(cls, zipfile):
    ...         instance = cls.from_config(zipfile.get("config.json"))
    ...         instance._submodel = Model1.load(zipfile)
    ...         return instance
    >>>
    >>> import tempfile
    >>> filepath = os.path.join(tempfile.mkdtemp(), "model.zip")
    >>> _ = Model2().save(filepath)
    >>> model = Model2.load(filepath)


.. autosummary::

    Saveable

"""

import inspect
import os
from abc import ABCMeta
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from psipy.__version__ import VERSION as psipy_version
from psipy.core.io.memory_zip_file import MemoryZipFile
from psipy.core.utils import git_commit_hash

__all__ = ["Saveable", "TSaveable"]


TSaveable = TypeVar("TSaveable", bound="Saveable")


class Saveable(metaclass=ABCMeta):
    """Saveable abstract base class.

    Usage: Inherit from this class, add built-in datatypes for storage by calling update_config() after super().__init__() has been called and whenever the value of a member variable changes.
    Overwrite the :meth:`_save` and :meth:`_load` methods to handle the serialization and deserialization of complex member types. What you don't register neither using update_config() nor in the _save() method will not be saved.

    Example:
        >>> class Model(Saveable):
        ...     def __init__(self, arg1=1, arg2=2):
        ...         super().__init__()
        ...         self.update_config(arg1=arg1, arg2=arg2)

    Inside :meth:`save` the primary method one uses is likely
    :meth:`MemoryZipFile.add <psipy.core.io.memory_zip_file.MemoryZipFile.add>`.
    That method guesses the fileformat by inspecting both the passed object
    (second argument) as well as the filepaths's (first argument) extension.
    The supported formats are the following:

    - ``tf.keras.Model``: ``.keras``
    - ``Dict[str, Union[None, int, float, str]]``: ``.json``
    - ``Dict[str, np.ndarray]``: ``.npz``
    - ``np.ndarray``: ``.npy``
    - ``Union[str, bytes, int float]``: ``.txt``

    For further information on what data can be stored inside a
    :class:`Saveable` take a look at the :mod:`~psipy.core.io.memory_zip_file`
    documentation.

    """

    #: Specifies the version of the Saveable ABC, but is also overloaded
    #: by implementations which specify their own version independent of
    #: the Saveable ABC's version.
    _version: ClassVar[Tuple[int, int, int]] = (1, 3, 0)

    def __init__(self: TSaveable, **kwargs):
        super().__init__() # this is a problem for a mixin, because it steels the parameters from "neighboring" mixins and superclasses. This means, Saveable can only be used in a base class, not e.g. in a class B subclassing A, as A might not receive its params. Furthermore, the sequence of the mixins and superclasses now is important. If mixins should be used, the init methods should all be cooperative (passing *args (if allowed in the package) and **kwargs to super().__init__()). Therefore we deprecated this in favor of update_config.

        if kwargs:
            import warnings
            warnings.warn(
                "Passing parameters to Saveable.__init__ is DEPRECATED. "
                "Use update_config() instead. If this was not intended, please be aware that Saveable.__init__ grasps all parameters passed to it and does not pass them to neither the superclass nor other mixins. This may cause initialization parameters to not reach their desired recipients and lead to unexpected behavior if not used carefully.",
                DeprecationWarning,
                stacklevel=2
            )
        self._config = kwargs

    @property
    def version(self: TSaveable) -> Tuple[int, int, int]:
        """Gets version number of the implementing class."""
        return self._version

    @property
    def saveable_version(self: TSaveable) -> Tuple[int, int, int]:
        """Gets version number of the Saveable baseclass."""
        return Saveable._version

    @classmethod
    def get_meta(cls: Type[TSaveable]) -> Dict[Any, str]:
        """Gets metadata, stored automatically for every saveable.

        Note that the git_hash contained in the meta data might be all zeros
        (when not saving from within a valid git repository) or not psipy's git
        hash (when saving from some other repository, using psipy).
        """
        return dict(
            class_name=cls.__name__,
            class_module=cls.__module__,
            psipy_version=".".join(map(str, psipy_version)),
            saveable_version=".".join(map(str, Saveable._version)),
            version=".".join(map(str, cls._version)),
            git_hash=git_commit_hash(fallback=True),
        )

    def get_config(self: TSaveable) -> Dict[str, Any]:
        """Returns the config needed for recreating the object self.

        Simply given the returned dictionary it is possible to fully recreate
        the current instance of the class using :meth:`from_config`. The
        recreated instance is not necessarily fitted/trained.
        """
        return self._config

    def update_config(self: TSaveable, **kwargs) -> None:
        """Adds a value to the config. Call this in the __init__ method of the subclass to pass parameters that should be stored AFTER initialization (e.g. after super().__init__() has been called). You can also call this method whenever the value of a member variable changes. All values in the config are stored and will be passed to the constructor of the class when loading from a zipfile. Call with only explicit parameters, do not pass-in kwargs from other mixins or superclasses."""
        self._config.update(kwargs)

    def update_config_from_dict(self: TSaveable, config: Dict[str, Any]) -> None:
        """Updates the config from a dictionary."""
        self._config.update(config)

    @classmethod
    def from_config(cls: Type[TSaveable], config: Dict[str, Any]) -> TSaveable:
        """Initializes a new instance of cls respecting config."""
        return cls(**config)

    def save(
        self: TSaveable,
        zipfile: Union[str, MemoryZipFile],
        name: Optional[str] = None,
    ) -> MemoryZipFile:
        """Saves the Saveable instance.

        If the zipfile argument is a string, a new zip is created and the
        saved object stored within a directory of the same name as the zipfile
        itself, in order to support easier unpacking. The complementary
        :meth:`load` method handles loading from strings accordingly.

        .. todo::
            Enhance handling of adding to and overwriting existing Saveable
            zip files.

        Args:
            zipfile: Filepath or MemoryZipFile to create or add to.
            name: Name to store files into within the zipfile. Helpful to easily
                  allow for storing multiple Saveables into a single zip.
        """
        name = type(self).__name__ if name is None else name

        filepath = None
        if isinstance(zipfile, str):
            filepath = zipfile
            # Ensure target is a file ending on .zip
            if filepath.endswith("/"):
                filepath = os.path.join(filepath, name)
            if not filepath.endswith(".zip"):
                filepath = f"{filepath}.zip"
            zipfile = MemoryZipFile(filepath)
            # All files within the zip should be wrapped in a common folder.
            basepath, _ = os.path.splitext(os.path.basename(filepath))
            zipfile.cd(basepath)

        zipfile.cd(name)
        zipfile.add("meta.json", self.get_meta())
        zipfile = self._save(zipfile)
        if filepath is not None:
            zipfile.save(filepath)
        zipfile.cd("..")
        return zipfile

    @classmethod
    def load(
        cls: Type[TSaveable],
        zipfile: Union[str, bytes, MemoryZipFile],
        name: Optional[str] = None,
        *,
        custom_objects: Optional[Union[List, Dict]] = None,
        validate_version=True,
    ) -> TSaveable:
        """Instantiates ``cls`` from a given zipfile.

        This method is primarily called in two cases: On the top level from a
        library user who wants to load an instance from disk and then
        recursively within the loaded class' ``load`` method for sub-instances.
        On the top level this method is called using a filepath (:class:`str`),
        on the lower level using a :class:`MemoryZipFile` instance.

        There is a third special case where one would directly want to pass a
        top-level :class:`MemoryZipFile` instance, for example when loading the
        :class:`MemoryZipFile` bytes from a database.

        If zipfile is a string, the zip is opened and expected to contain a
        folder of the same name as the zipfile's filename. This is ensured by
        this method and handled the same by :meth:`save`.

        Args:
            zipfile: Can be both a filepath or a MemoryZipFile in order to
                support calling this method again from within a :meth:`_load`
                implementation.
            name: Name of the Saveable to load. By default, the calling class'
                name (``cls.__name__``) is used. This argument specifies the
                subdirectory of the zipfile's current working directory to
                load the model from.
            custom_objects: Keyword argument currently without behaviour on its
                own, but passed further down to submodel's protected
                :meth:`_load` methods when they expect it. Relevant when a
                submodel requires ``custom_objects`` to properly load keras
                models. `See the tensorflow docs for details
                <https://tensorflow.org/api_docs/python/tf/keras/models/load_model>`_.
            validate_version: Whether to validate the Saveable's major versions.
                May be disabled in order to make upgrading to new major psipy
                versions less of a hassle as long as one is confident that his
                or her own stored Saveable is still compatible.
        """
        if name is None:
            name = cls.__name__
        if isinstance(zipfile, (str, bytes)):
            zipfile = MemoryZipFile(zipfile)
            # Collect only directories and no zip file special files
            top_level = [
                d for d in zipfile.ls(include_directories=True) if d.endswith("/")
            ]
            if len(top_level) == 0:
                raise ValueError(
                    "No top-level directory found, files contained in zip: "
                    f"{zipfile.ls(include_directories=True, recursive=True)}"
                )
            if len(top_level) > 1:
                raise ValueError(
                    "ZIP ambiguous, multiple top-level directories: "
                    f"{zipfile.ls(include_directories=True)}"
                )
            zipfile.cd(top_level[0])

        zipfile.cd(name)

        if validate_version:
            cls.validate_version(zipfile)

        if "custom_objects" in inspect.getfullargspec(cls._load).args:
            # Implementing class may expect `custom_objects` as an optional
            # keyword argument to :meth:`_load`, but this is not a requirement
            # for simplicity and compatibility purposes.
            instance = cls._load(zipfile, custom_objects=custom_objects)  # type: ignore
        else:
            instance = cls._load(zipfile)

        zipfile.cd("..")
        return instance

    @classmethod
    def validate_version(cls: Type[TSaveable], zipfile: MemoryZipFile) -> None:
        """Validates major versions stored in zip against current class."""
        zip_meta = {
            key: version.split(".")[0]
            for key, version in zipfile.get("meta.json").items()
            if key.endswith("version")
        }
        meta = {
            key: version.split(".")[0]
            for key, version in cls.get_meta().items()
            if key.endswith("version")
        }
        if zip_meta != meta:
            raise ValueError(
                f"Version incompatibility detected when loading zip.\n"
                f"Version in zip: {str(zip_meta)}\n"
                f"Current versions: {str(meta)}"
            )

    def _save(self: TSaveable, zipfile: MemoryZipFile) -> MemoryZipFile:
        """Saves :meth:`get_config` results as json, to be overloaded by subclasses."""
        return zipfile.add("config.json", self.get_config())

    @classmethod
    def _load(cls: Type[TSaveable], zipfile: MemoryZipFile) -> TSaveable:
        """Loads ``config.json`` contents, to be overloaded by subclasses."""
        return cls.from_config(zipfile.get("config.json"))
