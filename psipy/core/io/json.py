# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Extended JSON encoding and decoding capabilities.

Extends the native json encoder and decoder to support additional datatypes:

- Tuples: JSON has no concept of tuples, therefore all python tuples are
  converted to python lists when json encoding tuples and decoding them again.
  This encoder class stores python tuples as dictionaries containing both the
  tuple's values as well as a ``__tuple__`` type marker.

- Numpy scalars: Encodes numpy scalars as dictionary with an additional field
  for the dtype. Numpy scalars do not inherit from basic python numeric types,
  e.g. :class:`np.int64 <numpy.dtype>` is not of type :class:`int`. This encoder
  class stores numpy scalars as dictionaries containing the string
  representation of the scalar as well as a ``__numpy__`` type marker.

Example:

    >>> data = {'dict': ({'of': ('t', 'u')}, ('p', np.int64(1)), ['e', 's'])}
    >>> json_check(data)
    True
    >>> packed = json_encode(data)
    >>> json_decode(packed)
    {'dict': ({'of': ('t', 'u')}, ('p', 1), ['e', 's'])}


.. autosummary::

    json_check
    json_decode
    json_encode
    JSONEncodable
    NativelyJSONEncodable

"""

import json
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np

from psipy.core.utils import flatten

__all__ = [
    "json_check",
    "json_decode",
    "json_encode",
    "JSONEncodable",
    "NativelyJSONEncodable",
]


# Currently this only annotates according to the legal top-level types, as
# anything else would require a recursive definition, as described in
# https://github.com/python/mypy/issues/731
NativelyJSONEncodable = Union[None, str, int, float, bool, List[Any], Dict[str, Any]]
JSONEncodable = Union[NativelyJSONEncodable, np.generic, Tuple[Any, ...]]


def json_check(elem: Any) -> bool:
    """Check whether the passed element could be serialized to json.

    Examples:

        >>> json_check({'a': np.int32(32)})
        True
        >>> json_check({"this", "is", "a", "set"})
        False
        >>> json_check([np.zeros(10)])
        False
        >>> json_check(None)
        True
        >>> json_check("string")
        True
        >>> json_check(b"bytest")
        False

    """
    if not isinstance(elem, Iterable):
        elem = [elem]
    if not isinstance(elem, (dict, list, str, tuple)):
        return False
    json_types = (str, int, float, np.generic)
    return all(val is None or isinstance(val, json_types) for val in flatten(elem))


class _JSONEncoder(json.JSONEncoder):
    """Encodes additional object to be json compatible.

    .. note::
        Private API. Use :meth:`json_encode` and :meth:`json_decode` instead.
        See their documentation for additional information.

    To be used in combination with :meth:`_json_decode_hook`. The type markers
    will be used to correctly decode the objects. In the future this class can
    be extended to support further types in order to store other objects in
    JSON format.

    Example:

        >>> data = {'dict': (np.int64(1),)}
        >>> json.dumps(data, cls=_JSONEncoder)
        '{"dict": {"__tuple__": true, "items": [{"__numpy__": "int64", "value": "1"}]}}'

    """

    def encode(self, obj: JSONEncodable) -> str:
        """Encodes obj to a json string. Called once on the top level object."""

        def json_pack(obj: JSONEncodable) -> NativelyJSONEncodable:
            """Recursive tuple packing and numpy scalar handling."""
            if isinstance(obj, tuple):
                return {"__tuple__": True, "items": [json_pack(val) for val in obj]}
            if isinstance(obj, list):
                return [json_pack(val) for val in obj]
            if isinstance(obj, dict):
                return {key: json_pack(val) for key, val in obj.items()}
            if isinstance(obj, np.generic):  # numpy scalar handling
                return {"__numpy__": obj.__class__.__name__, "value": str(obj)}
            return obj

        return super(_JSONEncoder, self).encode(json_pack(obj))


def _json_decode_hook(obj: Dict[str, NativelyJSONEncodable]) -> JSONEncodable:
    """Unpacks objects annotated by :class:`_JSONEncoder`.

    .. note::
        Private API. Use :meth:`json_encode` and :meth:`json_decode` instead.
        See their documentation for additional information.

    To be used in combination with :class:`_JSONEncoder` as object_hook argument
    to :meth:`json.loads`. Should be exetended in accordance with
    :class:`_JSONEncoder`.

    Example:

        >>> data = {'dict': ({'of': ('t', 'u')}, ('p', np.int64(1)), ['e', 's'])}
        >>> packed = json.dumps(data, cls=_JSONEncoder)
        >>> json.loads(packed, object_hook=_json_decode_hook)
        {'dict': ({'of': ('t', 'u')}, ('p', 1), ['e', 's'])}

    """
    if "__tuple__" in obj and obj["__tuple__"]:
        items = cast(Iterable, obj["items"])
        return tuple(items)
    if "__numpy__" in obj:
        dtype = cast(str, obj["__numpy__"])
        return getattr(np, dtype)(obj["value"])
    return obj


def json_encode(obj: JSONEncodable) -> str:
    """Encodes the passed object into json, handling natively unsupported datatypes.

    .. note::
        This does not double check whether the given object is json encodable!
        If you want to make sure of that first, use :meth:`json_check`.

    Example:

        >>> data = {'dict': (np.int64(1),)}
        >>> json_encode(data)
        '{"dict": {"__tuple__": true, "items": [{"__numpy__": "int64", "value": "1"}]}}'

    """
    return json.dumps(obj, cls=_JSONEncoder)


def json_decode(string: str) -> JSONEncodable:
    """Decodes string encoded by :meth:`json_encode` into a python object.

    Antagonist to :meth:`json_encode`, should always be used together.

    Example:

        >>> data = {'dict': ({'of': ('t', 'u')}, ('p', np.int64(1)), ['e', 's'])}
        >>> packed = json_encode(data)
        >>> json_decode(packed)
        {'dict': ({'of': ('t', 'u')}, ('p', 1), ['e', 's'])}

    """
    return json.loads(string, object_hook=_json_decode_hook)
