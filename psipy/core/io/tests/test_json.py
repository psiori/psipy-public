# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import json

import numpy as np

from psipy.core.io.json import _json_decode_hook, _JSONEncoder, json_check


class TestJsonCheck:
    @staticmethod
    def test_ndarray():
        assert not json_check({"a": np.array([1, 2, 3])})


class TestZipFileJSONEncoder:
    @staticmethod
    def test_tuple_pack_unpack():
        """Make sure json en-/decoding lets tuples be tuples."""
        data = {"dict": ({"of": ("t", "u")}, ("p", "l"), ["e", "s"])}
        packed = json.dumps(data, cls=_JSONEncoder)
        print(packed)
        unpacked = json.loads(packed, object_hook=_json_decode_hook)
        assert data == unpacked

    @staticmethod
    def test_numpy_pack_unpack():
        data = {"int": np.int64(42), "float": np.float16(3.14159)}
        packed = json.dumps(data, cls=_JSONEncoder)
        print(packed)
        unpacked = json.loads(packed, object_hook=_json_decode_hook)
        assert unpacked == data
        assert isinstance(unpacked["int"], np.int64)
        assert isinstance(unpacked["float"], np.float16)
