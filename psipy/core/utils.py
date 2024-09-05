# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""General utility functions.

.. autosummary::

    add_bool_arg
    argument_names
    busy_sleep
    call_picky
    consecutive_runs
    count_unique_values_per_row
    deep_set
    dict_merge
    flatten
    flatten_dict
    git_commit_hash
    groupbys
    guess_primitive
    index_or_none
    lazy_get
    pick
    rolling_window
    split_path_name_ext
    unflatten_dict

"""

import argparse
import collections
import os.path
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from inspect import getfullargspec, isclass, signature
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

__all__ = [
    "add_bool_arg",
    "argument_names",
    "busy_sleep",
    "call_picky",
    "consecutive_runs",
    "deep_set",
    "dict_merge",
    "flatten_dict",
    "flatten",
    "git_commit_hash",
    "groupbys",
    "guess_primitive",
    "index_or_none",
    "lazy_get",
    "pick",
    "split_path_name_ext",
    "unflatten_dict",
]


#: Unbound type variable used in utility methods to correctly typehint return
#: values based on argument types.
T = TypeVar("T")


def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool = False,
    help: Optional[str] = None,
) -> argparse._MutuallyExclusiveGroup:
    """Adds a boolean flag to the given :class:`~ArgumentParser`.

    Parses "natural language" strings as boolean (true, yes, false, no, ...) and
    also accepts flags without values (``--flag`` as true, ``--no-flag`` as
    false).

    Example::

        >>> parser = argparse.ArgumentParser()
        >>> gr = add_bool_arg(parser, "go", default=False)
        >>> parser.parse_args(["--go"]).go
        True
        >>> parser.parse_args(["--no-go"]).go
        False

    """

    def str2bool(val):
        """Parse string as boolean."""
        val = val.lower()
        if val in ("true", "t", "yes", "y", "1"):
            return True
        if val in ("false", "f", "no", "n", "0"):
            return False
        raise ValueError(f"Expected type convertible to bool, received `{val}`.")

    name = name.lstrip("-")
    gr = parser.add_mutually_exclusive_group()
    gr.add_argument(
        f"--{name}", nargs="?", default=default, const=True, type=str2bool, help=help
    )
    gr.add_argument(f"--no-{name}", dest=name, action="store_false")
    return gr


def argument_names(func: Callable) -> List[str]:
    """Get the names of the arguments to ``func``. Includes ``self`` if present.

    Example::

        >>> def abcsum(a, b, c=2):
        ...     return a + b + c
        >>> argument_names(abcsum)
        ['a', 'b', 'c']

    """
    # Somehow, for tensorflow 2.3 (+?) classes (e.g. tf.keras.layers.Dense),
    # the code below does return only `self`, `args` and `kwargs`. Therefore
    # we need to access the class' `__init__` method directly.
    if isclass(func):
        func = getattr(func, "__init__")  # noqa
    # `getfullargspec` doesn't work properly for wrapped methods,
    # `signature` on the other hand misses out on 'self' argument
    names = set(signature(func).parameters).union(getfullargspec(func).args)
    return sorted(list(names))


def busy_sleep(duration: float):
    """Busy sleep for ``duration`` seconds.

    Problem this resolves which occurred on Azure devops macos workers::

            def test_with(timer):
                with timer:
                    time.sleep(0.01)
        >       assert_almost_equal(timer.time, 0.01, decimal=2)
        E       AssertionError:
        E       Arrays are not almost equal to 2 decimals
        E        ACTUAL: 0.06307673454284668
        E        DESIRED: 0.01

    Args:
        duration: Time to sleep in fractional seconds.
    """
    start = time.perf_counter()
    while time.perf_counter() < (start + duration):
        continue


def call_picky(func: Callable[..., T], *args, **kwargs) -> T:
    """Calls ``func`` with ``args`` and ``kwargs`` that fit its spec.

    Example:

        >>> def abcsum(a, b, c):
        ...     return a + b + c
        >>> dct = {"b": 2, "c": 3, "d": 4}
        >>> call_picky(abcsum, 1, **dct)
        6

    """
    return func(*args, **pick(kwargs, *argument_names(func)))


def consecutive_runs(lst: List[int]) -> List[List[int]]:
    """Returns runs of consecutive incrementing integers, without sorting.

    Inspired by an `official example
    <https://docs.python.org/2.6/library/itertools.html#examples>`_ from the
    python 2.6 documentation.

    Example::

        >>> data = [1, 4, 5, 6, 10, 15, 16, 17, 18, 22, 25, 26, 27, 28]
        >>> consecutive_runs(data)
        [[1], [4, 5, 6], [10], [15, 16, 17, 18], [22], [25, 26, 27, 28]]
        >>> data = [1, 9, 6, 5, 4, 10, 15, 16, 17]
        >>> consecutive_runs(data)
        [[1], [9], [6], [5], [4], [10], [15, 16, 17]]

    """
    groups = groupby(enumerate(lst), lambda x: x[0] - x[1])
    return [[v[1] for v in g] for k, g in groups]


def deep_set(dct: Dict, path: Sequence[str], value: Any, inplace: bool = False) -> Dict:
    """Deeply sets ``value`` at ``path`` in nested dictionary ``dct``.

    Args:
        dct: Dictionary to modify.
        path: Individual parts of the path into ``dct`` to set the value at.
        value: New value, overwrites any existing value at ``path``.
        inplace: Whether to modify the given dict ``dct`` in place or create a
            copy. Defaults to creating a copy in order to prevent unexpected
            side-effects.

    Returns:
        Modified dictionary, either as a reference to the received ``dct`` or
        as a modified deepcopy when ``inplace == True``.

    Example:

        >>> deep_set(dict(), ["a", "b", "c"], 1)
        {'a': {'b': {'c': 1}}}
        >>> deep_set({"a": {"b": {"c": 2, "d": 3}}}, ["a", "b", "c"], 1)
        {'a': {'b': {'c': 1, 'd': 3}}}

    """
    if not inplace:
        dct = deepcopy(dct)
    subdct = dct
    for key in path[:-1]:
        subdct = subdct.setdefault(key, {})
    subdct[path[-1]] = value
    return dct


def dict_merge(dct: Dict, other: Dict, inplace: bool = False) -> Dict:
    """Merges dictionaries recursively.

    Inspired by :meth:`dict.update`, instead of updating only top-level
    key/value pairs, :meth:`dict_merge` recurses down to an arbitrary depth.
    Note that while nested dictionaries are merged recursively, values with
    matching keys in ``dct`` and ``other`` but of other type will be
    overwritten. See the example for a better understanding of this.

    Example:

        >>> dct = {"a": {"b": 1}, "c": 2, "d": 3}
        >>> other = {"a": {"e": 4}, "c": {"f": 5}}
        >>> dict_merge(dct, other)
        {'a': {'b': 1, 'e': 4}, 'c': {'f': 5}, 'd': 3}

    Args:
        dct: Target dictionary.
        other: Source dictionary.
        inplace: Whether to modify the given dictionary ``dct`` in place or
            to create a copy. Defaults to creating a copy in order to prevent
            unexpected side-effects.

    Returns:
        Modified dictionary, either as a reference to the received ``dct`` or
        as a modified deepcopy when ``inplace == True``.

    """
    if not inplace:
        dct = deepcopy(dct)
    for k in other.keys():
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(other[k], collections.Mapping)
        ):
            # Can use inplace=True here as `dct` already will be a deepcopy of
            # the # original data if the user requested inplace=False initially.
            dict_merge(dct[k], other[k], inplace=True)
        else:
            # Create new keys in `dct` or overwrite non-dict values.
            dct[k] = other[k]
    return dct


def flatten(data: Any) -> List[Any]:
    """Flattens data structures to a flat list of values.

    .. warning::
        Also truncates dictionaries, only keeping their values in the order
        provided when iterating over them.

    Example:

        >>> flatten([[1], (2, 5), {"a": 3}, (((4,),),)])
        [1, 2, 5, 3, 4]
        >>> sorted(flatten({"a": 3, "b": 1, "c": 2}))
        [1, 2, 3]

    """
    if not isinstance(data, collections.abc.Iterable):
        return [data]
    if isinstance(data, dict):
        data = data.values()
    flat: List[Any] = []
    for item in data:
        if isinstance(item, (dict, list, tuple)):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def flatten_dict(
    obj: Dict[str, Any], sep: str = ".", sanitizer: Optional[str] = None
) -> Dict[str, Any]:
    """Recursively builds a flat dictionary of ``{deep.paths: values}`` mappings.

    Uses ``sep`` as separator between dictionary key levels and optionally
    replaces pre-existing uses of ``sep`` in the dictionary keys using the
    ``sanitizer`` string. Can be undone by using :meth:`unflatten_dict` with
    the same arguments.

    Example:

        >>> flatten_dict({"a": {"b": {"c": (1, 2)}, "d": {"e": [3, 4]}}})
        {'a.b.c': (1, 2), 'a.d.e': [3, 4]}
        >>> flatten_dict({"a": {"b": {"c": (1, 2)}, "d": {"e": [3, 4]}}}, sep="/")
        {'a/b/c': (1, 2), 'a/d/e': [3, 4]}
        >>> flatten_dict({"a": {"b": {"c.d": (1, 2)}}}, sanitizer="|")
        {'a.b.c|d': (1, 2)}
        >>> flatten_dict({"a": {"b/d": {"c": (1, 2)}}}, sep="/", sanitizer="|")
        {'a/b|d/c': (1, 2)}

    """
    result = dict()
    for outer_k, outer_v in obj.items():
        if sanitizer is not None:
            outer_k = outer_k.replace(sep, sanitizer)
        if isinstance(outer_v, dict):
            for inner_k, inner_v in flatten_dict(outer_v, sep, sanitizer).items():
                result[f"{outer_k}{sep}{inner_k}"] = inner_v
        else:
            result[f"{outer_k}"] = outer_v
    return result


def git_commit_hash(short: bool = False, fallback: bool = False) -> str:
    """Get the current commit hash of the repo at this point in time.

    Args:
        short: Only return the first 7 characters of the hash.
        fallback: Whether to fallback to all zeros if not in a valid git repo.
    """
    command = ["git", "rev-parse", "HEAD"]
    if short:
        command.insert(2, "--short")
    try:
        return subprocess.check_output(command).decode("ascii").strip()
    except Exception as err:
        if fallback:
            if short:
                return "0" * 7
            return "0" * 40
        raise err


def groupbys(
    items: Sequence[T],
    *keyfuncs: Callable[[T], Optional[str]],
) -> List[List[T]]:
    """Group by many keys, producing groups with exactly keyfunc-many items.

    For each unique key (over all keyfuncs), one group is created. Item
    positioning in the resulting groups match their respective keyfunc by index.

    Note that if a keyfunc returns the same key for multiple items, only the
    last item of that key for that keyfunc will remain in the group!

    Groups for which not all keyfuncs return a key are not returned.

    Example::

        >>> lst = ["aa", "ab", "ac", "ba", "bb", "bc", "ca", "cb", "cc"]
        >>> func1 = lambda v: v[0]
        >>> func2 = lambda v: v[1]
        >>> groupbys(lst, func1, func2)
        [['ac', 'ca'], ['bc', 'cb'], ['cc', 'cc']]

    Args:
        items: Sequence of items to group, commonly for example filepaths.
        *keyfuncs: Functions to use for grouping. Each function represents one
                   item within a group. So items matching the first keyfunc are
                   assigned as first element of their respective group.
    """
    dct: Dict[str, List[Union[T, None]]] = defaultdict(lambda: [None] * len(keyfuncs))
    for ix, keyfunc in enumerate(keyfuncs):
        for item in items:
            key = keyfunc(item)
            if key is not None:
                dct[key][ix] = item
    # Need the 'type: ignore' as mypy does currently not correctly pickup on
    # the 'if not any' filtering.
    return [
        vals  # type: ignore
        for vals in dct.values()
        if not any(v is None for v in vals)
    ]


def guess_primitive(val: str) -> Union[None, bool, float, int, str]:
    """Try coercing given string to some other primitive.

    Example::

        >>> guess_primitive("true")
        True
        >>> guess_primitive("False")
        False
        >>> guess_primitive("None")
        >>> guess_primitive("none")
        'none'
        >>> guess_primitive("")
        >>> guess_primitive("1.2")
        1.2
        >>> guess_primitive("1,2")
        '1,2'
        >>> guess_primitive("12")
        12

    Args:
        val: String value to coerce to other primitive.
    """
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val == "None" or val == "":
        return None
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        pass

    return val


def index_or_none(val: str, pattern: str, offset: int = 0) -> Union[int, None]:
    """Return the leftmost index of ``pattern`` in ``val`` if found, otherwise None.

    Example::

        >>> haystack = "some/sub/path"
        >>> index_or_none(haystack, "/")
        4
        >>> haystack[:index_or_none(haystack, "/", 1)]
        'some/'
        >>> haystack = "directory"
        >>> haystack[:index_or_none(haystack, "/", 1)]
        'directory'

    Args:
        val: Haystack.
        pattern: Needle.
        offset: Additional offset to add to the resulting index.
    """
    try:
        return val.index(pattern) + offset
    except ValueError:
        return None


def lazy_get(obj: Dict[str, T], *keys: str, default: Optional[T] = None) -> T:
    """Gets the first key found in obj, optionally falling back to default.

    Args:
        obj: Dictionary to test for keys.
        keys: Keys ordered by priority, when the former are not found,
              the latter are used.
        default: Keyword argument containing the default value to fallback to
                 when none of ``keys`` is found in ``obj``.

    Example::

        >>> obj = dict(c=3, a=1, b=2)
        >>> lazy_get(obj, "e", "b")
        2
        >>> lazy_get(obj, "e", default=4)
        4

    """
    for key in keys:
        try:
            return obj[key]
        except KeyError:
            pass
    if default:
        return default
    if keys:
        raise KeyError(keys[-1])  # for loop above produced no result
    raise KeyError()  # `keys` argument was empty


def unflatten_dict(
    obj: Dict[str, Any], sep: str = ".", sanitizer: Optional[str] = None
) -> Dict[str, Any]:
    """Unflattens a dictionary of ``{deep.paths: values}`` mappings.

    Splits keys by ``sep`` and optionally replaces ``sanitizer`` characters by
    ``sep`` after unflattening. Can be used to undo :meth:`flatten_dict` by
    using the same arguments.

    Example:

        >>> unflatten_dict({'a.b.c': (1, 2), 'a.d.e': [3, 4]})
        {'a': {'b': {'c': (1, 2)}, 'd': {'e': [3, 4]}}}
        >>> unflatten_dict({'a/b/c': (1, 2), 'a/d/e': [3, 4]}, sep="/")
        {'a': {'b': {'c': (1, 2)}, 'd': {'e': [3, 4]}}}
        >>> unflatten_dict({'a.b.c|d': (1, 2)}, sanitizer="|")
        {'a': {'b': {'c.d': (1, 2)}}}
        >>> unflatten_dict({'a/b|d/c': (1, 2)}, sep="/", sanitizer="|")
        {'a': {'b/d': {'c': (1, 2)}}}
        >>> unflatten_dict(flatten_dict({"a": {"b": {"c": (1, 2)}}}))
        {'a': {'b': {'c': (1, 2)}}}

    """
    result: Dict[str, Any] = dict()
    for key, value in obj.items():
        path = key.split(sep)
        if sanitizer is not None:
            path = [k.replace(sanitizer, sep) for k in key.split(sep)]
        deep_set(result, path, value, inplace=True)
    return result


def pick(dct: Dict[str, T], *keys: Union[List[str], str]) -> Dict[str, T]:
    """Extracts a subset of the dictionary ``dct`` based on ``keys``.

    Example:

        >>> dct = {"a": 1, "b": 2, "c": 3}
        >>> pick(dct, "b", "c")
        {'b': 2, 'c': 3}

    """
    flat_keys: List[str] = flatten(keys)
    return {key: dct[key] for key in flat_keys if key in dct}


def split_path_name_ext(filepath: str) -> Tuple[str, str, str]:
    """Splits a given filepath into its directory path, filename and file extension.

    In contrast to manual usage of :meth:`os.path.splitext`,
    :meth:`split_path_name_ext` extracts the following "dual" extensions as
    actual extension instead of having the first part belong to the filename
    (replace ``*`` with an arbitrary value like for example ``tar``):

    - ``.*.gz``
    - ``.*.bz2``

    Example:

        >>> split_path_name_ext("path/to/file.tar.gz")
        ('path/to', 'file', '.tar.gz')
        >>> split_path_name_ext("path/to/file.123.bz2")
        ('path/to', 'file', '.123.bz2')
        >>> split_path_name_ext("path/to/file.tar.other")
        ('path/to', 'file.tar', '.other')
        >>> split_path_name_ext("filename.tar")
        ('', 'filename', '.tar')
        >>> split_path_name_ext("dirpath/filename")
        ('dirpath', 'filename', '')
        >>> split_path_name_ext("filename")
        ('', 'filename', '')
        >>> split_path_name_ext(".hidden.ext")
        ('', '.hidden', '.ext')
        >>> split_path_name_ext(".hidden")
        ('', '.hidden', '')

    """
    dirpath = os.path.dirname(filepath)
    name, ext = os.path.splitext(os.path.basename(filepath))
    if ext in [".gz", ".bz2"]:
        name, ext2 = os.path.splitext(name)
        ext = ext2 + ext
    return dirpath, name, ext
