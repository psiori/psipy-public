# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Configuration utilities for python packages and programs.

.. autosummary::

    config_to_dict
    ConfigSection

"""

from collections import defaultdict
from configparser import ConfigParser
from typing import Dict, Optional, Union, get_type_hints

from psipy.core.utils import guess_primitive

__all__ = ["config_to_dict", "ConfigSection"]


ParseableTypes = Optional[Union[str, int, float, bool]]


def config_to_dict(config: ConfigParser) -> Dict[str, Dict[str, ParseableTypes]]:
    """Convert :class:`~configparser.ConfigParser` object to :class:`dict`.

    Args:
        config: :class:`~configparser.ConfigParser` instance to convert.
    """
    dct: Dict[str, Dict[str, ParseableTypes]] = defaultdict(dict)
    for section in config.sections():
        for option in config.options(section):
            dct[section][option] = guess_primitive(config.get(section, option))
    return dict(dct)


class ConfigSection:
    """Base class for individual configuration sections.

    Allows for specifying data types and default values in pythonic fashion.
    Datatypes will be ensured on initialization.

    Example usage::

        >>> class LogConfigSection(ConfigSection):
        ...     backup: int = 365
        ...     folder: str = "logs"
        ...     formatter: str = "basic"
        ...     level_file: str = "DEBUG"
        ...     level: str = "DEBUG"
        ...     log_raw_json: int = 0
        ...     sart_rollover: str = "m"

    """

    def __init__(self, **kwargs: Union[str, int]):
        types = get_type_hints(self.__class__)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, types[key](value))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        items = self.__dict__.items()
        kwargs = ", ".join([f"{key}={repr(value)}" for key, value in items])
        return f"{self.__class__.__name__}({kwargs})"
