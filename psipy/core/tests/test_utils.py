# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import argparse
import json
from collections import OrderedDict

import numpy as np
import pytest

from psipy.core.utils import add_bool_arg, call_picky, deep_set, dict_merge
from psipy.core.utils import flatten, flatten_dict, lazy_get, pick, unflatten_dict


def test_deep_set():
    obj = dict(a=0, b=1, c=dict(d=3), deep=False)
    obj2 = deep_set(obj, ["c", "e", "f"], 5)
    assert "e" not in obj["c"]
    assert obj2["c"]["e"]["f"] == 5
    obj2 = deep_set(obj, ["c", "e", "f"], 5, inplace=True)
    assert "e" in obj["c"]
    assert obj2 is obj


def test_flatten():
    assert flatten([1, 2, 4]) == [1, 2, 4]
    assert flatten((1, 2, 4)) == [1, 2, 4]
    assert flatten([1, 2, (1, 2, 4)]) == [1, 2, 1, 2, 4]
    assert flatten([1, 2, ([1, 2, 4], 2, 4)]) == [1, 2, 1, 2, 4, 2, 4]
    assert sorted(flatten({"a": (1, 2, 3), "b": 5})) == [1, 2, 3, 5]
    assert flatten([OrderedDict((("a", (1, 2, 3)), ("b", 5))), 4]) == [1, 2, 3, 5, 4]


def test_flatten_unflatten_dict():
    # Tests the combination of unflatten_dict(flatten_dict()) only, as the
    # individual methods are already sufficiently tested in their docstring.

    def deep_equals(a, b):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    data = {"a": {"b": {"c": (1, 2)}, "d": {"e": [3, 4]}}}
    assert deep_equals(unflatten_dict(flatten_dict(data)), data)
    data = {"a": {"b": {"c": (1, 2)}, "d": {"e": [3, 4]}}}
    kw = dict(sep="/")
    assert deep_equals(unflatten_dict(flatten_dict(data, **kw), **kw), data)
    data = {"a": {"b": {"c.d": (1, 2)}}}
    kw = dict(sanitizer="|")
    assert deep_equals(unflatten_dict(flatten_dict(data, **kw), **kw), data)
    data = {"a": {"b/d": {"c": (1, 2)}}}
    kw = dict(sep="/", sanitizer="|")
    assert deep_equals(unflatten_dict(flatten_dict(data, **kw), **kw), data)


def test_pick():
    assert pick(dict(a=1, b=2, c=3), "a", "c") == dict(a=1, c=3)


def test_call_picky():
    def _zeros(shape, dtype=float):
        return np.zeros(shape, dtype=dtype)

    a = call_picky(_zeros, 3, dtype=np.int8, foo=42)
    b = np.zeros(shape=3, dtype=np.int8)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == b.dtype


def test_dictmerge_inplace():
    dct1 = dict(a=1, b=2, c="b")
    dct2 = dict(a=1, d=[1, 2, 3], c=9)

    dct_merged = dict_merge(dct1, dct2, inplace=True)
    assert dct_merged is dct1

    assert sorted(list(dct1.keys())) == ["a", "b", "c", "d"]
    assert dct_merged["a"] == 1
    assert dct_merged["b"] == 2
    assert dct_merged["c"] == 9
    assert dct1["c"] == 9
    assert dct_merged["d"] == [1, 2, 3]


def test_dictmerge_not_inplace():
    dct1 = dict(a=1, b=2, c="b")
    dct2 = dict(a=1, d=[1, 2, 3], c=9)
    dct_merged = dict_merge(dct1, dct2, inplace=False)

    assert sorted(list(dct_merged.keys())) == ["a", "b", "c", "d"]
    assert dct_merged["a"] == 1
    assert dct_merged["b"] == 2
    assert dct_merged["c"] == 9
    assert dct1["c"] == "b"
    assert dct_merged["d"] == [1, 2, 3]


def test_dictmerge_nested():
    dct1 = {
        "patient": {
            "name": None,
            "id": None,
            "birthdate": None,
            "gender": None,
            "institut": None,
            "doctor": "Dr. House",
        },
        "identifier": "abc",
    }

    dct2 = dict(
        patient=dict(name="Max", id=9, birthdate="07.09.1956", new_field=[1, 2, 3]),
        new_root_elem=dict(a=1, b=2),
    )
    dict_merge(dct1, dct2, inplace=True)

    assert "patient" in dct1.keys()
    assert dct1["patient"]["name"] == "Max"
    assert dct1["patient"]["id"] == 9
    assert dct1["patient"]["gender"] is None
    assert dct1["patient"]["new_field"] == [1, 2, 3]
    assert dct1["identifier"] == "abc"
    assert dct1["new_root_elem"]["a"] == 1


@pytest.mark.parametrize(
    "arg,val",
    [
        ("--go", True),
        ("--no-go", False),
        ("--go=true", True),
        ("--go=t", True),
        ("--go=yes", True),
        ("--go=y", True),
        ("--go=1", True),
        ("--go=false", False),
        ("--go=f", False),
        ("--go=no", False),
        ("--go=n", False),
        ("--go=0", False),
    ],
)
def test_add_bool_arg(arg, val):
    parser = argparse.ArgumentParser()
    add_bool_arg(parser, "go", default=False)
    assert parser.parse_args([arg]).go is val


def test_add_bool_arg_defaults():
    parser = argparse.ArgumentParser()
    add_bool_arg(parser, "go", default=False)
    assert parser.parse_args([]).go is False

    parser = argparse.ArgumentParser()
    add_bool_arg(parser, "go", default=True)
    assert parser.parse_args([]).go is True


def test_lazy_get():
    obj = dict(b=2, d=4, a=1, c=3)
    assert lazy_get(obj, "a", "b") == 1
    assert lazy_get(obj, "b") == 2
    assert lazy_get(obj, "b", "a") == 2
    assert lazy_get(obj, "e", "a") == 1
    assert lazy_get(obj, "e", default=42) == 42
    assert lazy_get(obj, "e", "a", default=42) == 1
    assert lazy_get(obj, "a", "e", default=42) == 1
    with pytest.raises(KeyError):
        lazy_get(obj, "e")
    with pytest.raises(KeyError):
        lazy_get(obj)
