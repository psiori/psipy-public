# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import io

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from psipy.core.io.memory_zip_file import MemoryZipFile


def test_add_json():
    zipfile = MemoryZipFile()
    zipfile.add("dict.json", {"a": 1, "b": [1, 2, 3]})
    assert "dict.json" in zipfile.namelist()

    json = zipfile.get("dict.json")
    assert json == {"a": 1, "b": [1, 2, 3]}


def test_add_dict_tuple():
    zipfile = MemoryZipFile()
    zipfile.add("dict.json", {"tuple": (1, 2, 3)})
    assert "dict.json" in zipfile.namelist()

    json = zipfile.get_json("dict.json")
    assert json == {"tuple": (1, 2, 3)}


def test_add_tuple():
    zipfile = MemoryZipFile()
    zipfile.add("tuple.json", (1, 2, 3, "str", None, 1.0))
    assert "tuple.json" in zipfile.namelist()

    json = zipfile.get("tuple.json")
    assert json == (1, 2, 3, "str", None, 1.0)


def test_add_list():
    zipfile = MemoryZipFile()
    zipfile.add("list.json", [1, 2, 3, "str", None, 1.0])
    assert "list.json" in zipfile.namelist()

    json = zipfile.get("list.json")
    assert json == [1, 2, 3, "str", None, 1.0]


def test_add_bad_dict():
    zipfile = MemoryZipFile()
    with pytest.raises(AssertionError):
        zipfile.add_json("bad.json", {"bad": pd.DataFrame([1, 2, 3])})


@pytest.mark.usefixtures("tensorflow")
def test_add_keras():
    inp = tf.keras.Input((1, 2))
    out = tf.keras.layers.Dense(1)(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)

    zipfile = MemoryZipFile()
    zipfile.add("model.keras", model)
    assert "model.keras" in zipfile.namelist()

    loaded_model = zipfile.get_keras("model.keras")
    assert loaded_model.input.shape == model.input.shape
    assert loaded_model.output.shape == model.output.shape

    data = np.random.random(2).reshape((1, 1, 2))
    assert np.array_equal(loaded_model(data).numpy(), model(data).numpy())


    # assert loaded_model.get_config() == model.get_config() 
    # TODO: reactivate, asap. unfortunately, save + load presently changes the input shapes build arguments of layers from tuple () before saving to array [] and fails the comparison. Dimensions stay the same :( thus, we cannot compare the configs directly for the time being, until the behavior in keras is fixed.


def test_get_unknown():
    zipfile = MemoryZipFile()
    with pytest.raises(ValueError):
        zipfile.get("model.random_extension")


def test_add_raw_str():
    zipfile = MemoryZipFile()
    zipfile.add_raw("raw.txt", "some text")
    assert "raw.txt" in zipfile.namelist()
    assert zipfile.read("raw.txt").decode("utf-8") == "some text"


def test_add_raw_bytes():
    zipfile = MemoryZipFile()
    zipfile.add_raw("raw2.txt", b"some text 2")
    assert "raw2.txt" in zipfile.namelist()
    assert zipfile.read("raw2.txt").decode("utf-8") == "some text 2"


def test_add_bytes_buffer():
    zipfile = MemoryZipFile()
    bio = io.BytesIO(b"bytes buffer: \x00\x01")
    zipfile.add("bytes.txt", bio)
    assert "bytes.txt" in zipfile.namelist()
    assert bio.getvalue() == zipfile.get_bytesio("bytes.txt").getvalue()


def test_add_numpy():
    arr = np.random.random((3, 4, 5))

    zipfile = MemoryZipFile()
    zipfile.add("arr.npy", arr)
    assert "arr.npy" in zipfile.namelist()

    loaded_arr = zipfile.get("arr.npy")
    assert np.all(arr == loaded_arr)


def test_add_numpy_dict():
    arrs = {"a": np.random.random((3, 4, 5)), "b": np.random.random((6, 7))}

    zipfile = MemoryZipFile()
    zipfile.add("arrs.npz", arrs)
    assert "arrs.npz" in zipfile.namelist()

    loaded_arrs = zipfile.get("arrs.npz")
    assert np.all(arrs["a"] == loaded_arrs["a"])
    assert np.all(arrs["b"] == loaded_arrs["b"])


def test_add_mixed_dict():
    data = {
        "a": np.random.random((3, 4, 5)),
        "b": np.random.random((6, 7)),
        "string": "some text",
        "float": 0.1,
        "int": 1,
        "nonetype": None,
    }

    zipfile = MemoryZipFile()
    zipfile.add("data", data)
    assert "data/data.json" in zipfile.namelist()
    assert "data/data.npz" in zipfile.namelist()

    data2 = zipfile.get("data")
    assert data.keys() == data2.keys()
    for key in data.keys():
        assert np.all(np.asarray(data[key] == data2[key]))


def test_add_mixed_list():
    data = [
        1,
        "str",
        None,
        ["inner", "list"],
        {"inner": "dict"},
        np.random.random((3, 4, 5)),
        {"npz": np.random.random((2, 4))},
        {"mixed": np.random.random((2, 4)), "dict": (1, 2, 3)},
    ]

    zipfile = MemoryZipFile()
    zipfile.add("data", data)
    assert "data/0.json" in zipfile.namelist()
    assert "data/1.json" in zipfile.namelist()
    assert "data/2.json" in zipfile.namelist()
    assert "data/3.json" in zipfile.namelist()
    assert "data/4.json" in zipfile.namelist()
    assert "data/5.npy" in zipfile.namelist()
    assert "data/6.npz" in zipfile.namelist()
    assert "data/7/__meta__.json" in zipfile.namelist()

    data2 = zipfile.get("data")
    assert len(data) == len(data2)
    for i, val in enumerate(data[:5]):
        assert val == data2[i]
    assert np.all(data[5] == data2[5])
    assert np.all(data[6]["npz"] == data2[6]["npz"])
    assert np.all(data[7]["mixed"] == data2[7]["mixed"])
    assert data[7]["dict"] == data2[7]["dict"]


def test_cd():
    zipfile = MemoryZipFile()
    zipfile.cd("somedir")
    assert zipfile.cwd() == "/somedir/"
    zipfile.cd("anotherdir")
    assert zipfile.cwd() == "/somedir/anotherdir/"
    zipfile.cd("..")
    assert zipfile.cwd() == "/somedir/"
    zipfile.cd("..")
    assert zipfile.cwd() == "/"


def test_add_in_cwd():
    zipfile = MemoryZipFile()
    zipfile.cd("adir")
    zipfile.add("dict.json", {"a": 1, "b": [1, 2, 3]})
    assert "adir/dict.json" in zipfile.namelist()


def test_ls():
    zipfile = MemoryZipFile()
    zipfile.add("first.json", {"a": 7})
    zipfile.cd("adir")
    zipfile.add("second.json", {"a": 1, "b": [1, 2, 3]})
    assert zipfile.ls() == ["second.json"]
    assert zipfile.ls(abs=True) == ["adir/second.json"]
    zipfile.cd("..")
    assert zipfile.ls(include_directories=False) == ["first.json"]
    assert zipfile.ls(include_directories=True) == ["adir/", "first.json"]
    assert zipfile.ls(include_directories=True, abs=True) == ["adir/", "first.json"]
    expected = ["adir/", "adir/second.json", "first.json"]
    assert zipfile.ls(include_directories=True, recursive=True) == expected
    expected = ["adir/", "adir/second.json", "first.json"]
    assert zipfile.ls(include_directories=True, abs=True, recursive=True) == expected


def test_csv_array():
    df1 = pd.DataFrame(np.random.random((10, 3)))
    zipfile = MemoryZipFile()
    zipfile.add_csv("data.csv", df1)
    df2 = zipfile.get_csv("data.csv", high_precision=True)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df2)
    # columns always come out as str
    df1.columns = [str(v) for v in df1.columns]
    pd.testing.assert_frame_equal(df1, df2)

    arr = np.random.random((10, 3))
    zipfile = MemoryZipFile()
    zipfile.add_csv("data.csv", arr)
    df = zipfile.get_csv("data.csv")
    assert np.allclose(arr.tolist(), df.values.tolist())
    df = zipfile.get_csv("data.csv", high_precision=True)
    assert np.array_equal(arr.tolist(), df.values.tolist())

    arr = np.random.randint(0, 1000, (10, 3))
    zipfile = MemoryZipFile()
    zipfile.add_csv("data.csv", arr)
    df = zipfile.get_csv("data.csv")
    assert np.array_equal(arr, df.values)

    arr = np.random.randint(0, 1000, (10, 3))
    zipfile = MemoryZipFile()
    zipfile.add_csv("data.csv", arr, columns=["a", "b", "c"])
    df = zipfile.get_csv("data.csv")
    assert np.array_equal(arr, df.values)
    assert list(df.columns) == ["a", "b", "c"]

    arr = np.random.randint(0, 1000, (10, 3))
    zipfile = MemoryZipFile()
    with pytest.raises(AssertionError):
        zipfile.add_csv("data.csv", arr, columns=["a", "b", "c", "d"])

    df1 = pd.DataFrame(np.random.random((10, 3)))
    zipfile = MemoryZipFile()
    zipfile.add_csv("data.csv", df1, columns=[1, 2])

    df1 = pd.DataFrame(np.random.random((10, 3)), columns=["a", "b", "c"])
    zipfile = MemoryZipFile()
    zipfile.add_csv("data.csv", df1)
    df2 = zipfile.get_csv("data.csv", high_precision=True)
    assert df1.equals(df2)


def test_tf_savedmodel():
    # we have deprecated add_tf, as it saved a keras model only using tf 2.x.
    # this version of memory_zip_file cannot load pb + variables anymore, 
    # as the present tf >=2.18 and keras 3 do not support it. If you have an 
    # old model with pb + variables, import tf_keras with tf 2.x and use that 
    # to load the model you can then save it using keras 3 with 
    # keras.saving.save_model(), or by adding that model to the zipfile with
    # add_keras().
    inp = tf.keras.Input((1, 2))
    out = tf.keras.layers.Dense(1)(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)
    inputs = np.random.random(2).reshape((1, 1, 2))
    result = model(inputs).numpy()

    zipfile = MemoryZipFile()
    zipfile.cd("abc")
    zipfile.add_tf("SavedModel123.keras", model)
    assert "SavedModel123.keras" in zipfile.ls(include_directories=True)
    assert "abc/SavedModel123.keras" in zipfile.ls(include_directories=True, abs=True)
    loaded_model = zipfile.get_tf("SavedModel123.keras")

    assert loaded_model.input.shape == model.input.shape
    assert loaded_model.output.shape == model.output.shape

    result2 = loaded_model(inputs).numpy()
    assert np.array_equal(result, result2)


def test_tf_savedmodel_from_disk():

    # inputs and results from "save time".
    inputs = np.array([[0.3, 0.4]])
    result = np.array([[-0.40714467]], dtype=np.float32)

    # code that hase been used to save the model.
    # if a new fixture needs to be created, rerun this code,
    # and update the result above accordingly.
    #inp = tf.keras.Input((2,))
    #hid = tf.keras.layers.Dense(10)(inp)
    #out = tf.keras.layers.Dense(1)(hid)
    #model = tf.keras.Model(inputs=inp, outputs=out)
    #zipfile = MemoryZipFile()
    #zipfile.cd("abc")
    #zipfile.add_keras("SavedModel123.keras", model)
    #zipfile.save("SavedModelMemoryZipFile.zip")
    #print(model(inputs).numpy())
    #breakpoint()

    path = "psipy/core/io/tests/assets/SavedModelMemoryZipFile.zip"
    zip = MemoryZipFile(path)
    loaded_model = zip.get_tf("abc/SavedModel123.keras")
    result2 = loaded_model(inputs).numpy()
    assert np.array_equal(result, result2)
