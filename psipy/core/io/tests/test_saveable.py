# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from psipy.core.io import MemoryZipFile, Saveable
from psipy.core.utils import git_commit_hash


class SaveableMock(Saveable):
    def __init__(self, arg1, arg2=2):
        Saveable.__init__(self, arg1=arg1, arg2=arg2)
        self.arg1 = arg1
        self.arg2 = arg2

    def _save(self, zipfile):
        return zipfile.add("config.json", self.get_config())

    @classmethod
    def _load(cls, zipfile):
        return cls.from_config(zipfile.get("config.json"))


class NoopLayerMock(tf.keras.layers.Layer):
    def __init__(self, arg, **kwargs):
        self.arg = arg
        super(NoopLayerMock, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def get_config(self):
        return dict(arg=self.arg)


class TestSaveable:
    @staticmethod
    def test_get_config():
        saveable = SaveableMock(arg1=2, arg2=3)
        assert saveable.get_config() == dict(arg1=2, arg2=3)

    @staticmethod
    def test_from_config():
        saveable = SaveableMock(arg1=[1, 2, 3])
        saveable2 = SaveableMock.from_config(saveable.get_config())
        assert saveable.get_config() == saveable2.get_config()
        assert saveable2.arg1 == [1, 2, 3]

    @staticmethod
    def test_save_load():
        dirpath = tempfile.mkdtemp()
        filepath = os.path.join(dirpath, "test.zip")

        saveable = SaveableMock(arg1=[1, 2, 3])
        mzp = saveable.save(filepath)

        # Test adding data and saving to same path.
        mzp.add_json("added.json", {"a": 1}).save()

        saveable2 = SaveableMock.load(filepath)
        assert saveable.get_config() == saveable2.get_config()

        zipfile = MemoryZipFile(filepath)
        zipfile.cd("test", SaveableMock.__name__)
        assert "meta.json" in zipfile.ls()
        meta = zipfile.get("meta.json")
        assert "class_name" in meta
        assert "class_module" in meta
        assert "psipy_version" in meta
        assert "saveable_version" in meta
        assert meta["saveable_version"] == ".".join(map(str, Saveable._version))
        assert "version" in meta
        assert "git_hash" in meta
        assert meta["git_hash"] == git_commit_hash(fallback=False)

        # Make sure the json added after the fact was actually written.
        print(zipfile.namelist())
        json = zipfile.get_json("/test/added.json")
        assert json == {"a": 1}

    @staticmethod
    def test_save_load_memzipfile():
        # Prepare memory zip file
        memzip = MemoryZipFile()
        basepath = "test"
        memzip.cd(basepath)

        saveable = SaveableMock(arg1=[1, 2, 3])
        saveable.save(memzip)

        # use memory to initialize saveable directly
        saveable = SaveableMock.load(memzip.getvalue())
        assert saveable.get_config() == saveable.get_config()

        # use memory to initialize Saveable from MemoryZipFile
        memzip3 = MemoryZipFile(memzip.getvalue())
        assert "test/" in memzip3.ls(include_directories=True)
        memzip3.cd("test")
        memsaveable = SaveableMock.load(memzip3)
        assert saveable.get_config() == memsaveable.get_config()

        memzip3.cd(SaveableMock.__name__)
        assert "meta.json" in memzip3.ls()
        meta = memzip3.get("meta.json")
        assert "version" in meta
        assert "saveable_version" in meta
        assert "psipy_version" in meta

    @staticmethod
    def test_save_to_directory():
        filepath = os.path.join(tempfile.mkdtemp(), "test/")

        saveable = SaveableMock(arg1=[1, 2, 3])
        saveable.save(filepath)

        name = f"{SaveableMock.__name__}.zip"
        assert name in os.listdir(filepath)

        filepath = os.path.join(filepath, name)
        saveable2 = SaveableMock.load(filepath)
        assert saveable.get_config() == saveable2.get_config()

    @staticmethod
    def test_save_without_ext():
        dirpath = tempfile.mkdtemp()
        filepath = os.path.join(dirpath, "test")

        saveable = SaveableMock(arg1=[1, 2, 3])
        saveable.save(filepath)

        assert "test.zip" in os.listdir(dirpath)

        filepath = os.path.join(dirpath, "test.zip")
        saveable2 = SaveableMock.load(filepath)
        assert saveable.get_config() == saveable2.get_config()

    @staticmethod
    def test_submodel_save_load():
        class MultiSaveableMock(Saveable):
            def __init__(self):
                Saveable.__init__(self)
                self._sub1 = SaveableMock(arg1=42)

            def _save(self, zipfile):
                zipfile.add("config.json", self.get_config())
                self._sub1.save(zipfile)
                return zipfile

            @classmethod
            def _load(cls, zipfile):
                obj = cls.from_config(zipfile.get("config.json"))
                obj._sub1 = SaveableMock.load(zipfile)
                return obj

        obj = MultiSaveableMock()
        filepath = os.path.join(tempfile.mkdtemp(), "test.zip")
        obj.save(filepath)
        obj2 = MultiSaveableMock.load(filepath)
        assert obj._sub1.get_config() == obj2._sub1.get_config()

    @staticmethod
    def test_multi_submodel_save_load():
        class MultiSaveableMock(Saveable):
            def __init__(self):
                Saveable.__init__(self)
                self._sub1 = SaveableMock(arg1=42)
                self._sub2 = SaveableMock(arg1={"dict": "arg"})

            def _save(self, zipfile):
                zipfile.add("config.json", self.get_config())
                self._sub1.save(zipfile, "sub1")
                self._sub2.save(zipfile, "sub2")
                return zipfile

            @classmethod
            def _load(cls, zipfile):
                obj = cls.from_config(zipfile.get("config.json"))
                obj._sub1 = SaveableMock.load(zipfile, "sub1")
                obj._sub2 = SaveableMock.load(zipfile, "sub2")
                return obj

        obj = MultiSaveableMock()
        filepath = os.path.join(tempfile.mkdtemp(), "test.zip")
        obj.save(filepath)
        obj2 = MultiSaveableMock.load(filepath)
        assert obj._sub1.get_config() == obj2._sub1.get_config()
        assert obj._sub2.get_config() == obj2._sub2.get_config()

    @staticmethod
    def test_hierachical_submodel_save_load():
        class HierarchicalSaveableMock2(Saveable):
            def __init__(self):
                Saveable.__init__(self)
                self._sub = SaveableMock(arg1=42)

            def _save(self, zipfile):
                zipfile.add("config.json", self.get_config())
                self._sub.save(zipfile)
                return zipfile

            @classmethod
            def _load(cls, zipfile):
                obj = cls.from_config(zipfile.get("config.json"))
                obj._sub = SaveableMock.load(zipfile)
                return obj

        class HierarchicalSaveableMock(Saveable):
            def __init__(self):
                Saveable.__init__(self)
                self._sub = HierarchicalSaveableMock2()

            def _save(self, zipfile):
                zipfile.add("config.json", self.get_config())
                self._sub.save(zipfile)
                return zipfile

            @classmethod
            def _load(cls, zipfile):
                obj = cls.from_config(zipfile.get("config.json"))
                obj._sub = HierarchicalSaveableMock2.load(zipfile)
                return obj

        obj = HierarchicalSaveableMock()
        filepath = os.path.join(tempfile.mkdtemp(), "test.zip")
        obj.save(filepath)
        obj2 = HierarchicalSaveableMock.load(filepath)
        assert obj._sub._sub.get_config() == obj2._sub._sub.get_config()
        assert obj._sub._sub.get_config() == obj2._sub._sub.get_config()

    @staticmethod
    @pytest.mark.usefixtures("tensorflow")
    def test_save_load_keras():
        class SaveableKerasMock(Saveable):
            def __init__(self):
                Saveable.__init__(self)
                inp = tf.keras.Input((1, 2))
                h = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1), output_shape=(1, 2))(inp)
                h = NoopLayerMock(arg=123)(h)
                out = tf.keras.layers.Dense(1)(h)
                self.model = tf.keras.Model(inputs=inp, outputs=out)

            def _save(self, zipfile):
                return zipfile.add("model.keras", self.model)

            @classmethod
            def _load(cls, zipfile):
                instance = cls()
                instance.model = zipfile.get_keras(
                    "model.keras",
                    custom_objects=[NoopLayerMock],
                )
                return instance

        dirpath = tempfile.mkdtemp()
        filepath = os.path.join(dirpath, "keras.zip")

        saveable = SaveableKerasMock()
        saveable.save(filepath)

        saveable2 = SaveableKerasMock.load(filepath)
        for l1, l2 in zip(saveable.model.layers, saveable2.model.layers):
            # Do not diff embedded encoded lambda function string.
            conf1 = l1.get_config()
            conf1.pop("function", None)
            conf2 = l2.get_config()
            conf2.pop("function", None)
            assert conf1 == conf2

    @staticmethod
    def test_save_load_mixed_dict():
        class SaveableMixedMock(SaveableMock):
            def _save(self, zipfile):
                return zipfile.add("config", self.get_config())

            @classmethod
            def _load(cls, zipfile):
                return cls.from_config(zipfile.get("config"))

        filepath = os.path.join(tempfile.mkdtemp(), "mixed.zip")
        arg1 = [1, 2, 3]
        arg2 = np.random.random((6, 7))

        a = SaveableMixedMock(arg1=arg1, arg2=arg2)
        assert a.get_config()["arg1"] == arg1
        assert np.all(a.get_config()["arg2"] == arg2)
        a.save(filepath)  # saves get_config() which is a mixed dict
        assert os.path.exists(filepath)

        b = SaveableMixedMock.load(filepath)
        assert a.get_config()["arg1"] == b.get_config()["arg1"]
        assert np.all(a.get_config()["arg2"] == b.get_config()["arg2"])

    @staticmethod
    def test_save_load_custom_objects():
        """Test for custom object. This is still draft version, therefore, just testing
        if load passes custom_objects to load method
        """

        class ScalerMock(Saveable):
            def __init__(self):
                Saveable.__init__(self)
                self._max = 10
                self._min = 0

        class CustomScalerMock(ScalerMock):
            def __init__(self):
                Saveable.__init__(self)
                self._max = 5
                self._min = 2

        class SaveableCustomMock(Saveable):
            def __init__(self, arg1, arg2=2):
                Saveable.__init__(self, arg1=arg1, arg2=arg2)
                self.arg1 = arg1
                self.arg2 = arg2
                self.custom_scaler = None

            def _save(self, zipfile):
                self.custom_scaler.save(zipfile)
                return zipfile.add("config.json", self.get_config())

            @classmethod
            def _load(cls, zipfile, custom_objects):
                assert custom_objects == [CustomScalerMock]

        dirpath = tempfile.mkdtemp()
        filepath = os.path.join(dirpath, "custom_object_model.zip")

        saveable = SaveableCustomMock(42)
        saveable.custom_scaler = CustomScalerMock()
        saveable.save(filepath)

        saveable2 = SaveableCustomMock.load(  # noqa
            filepath, custom_objects=[CustomScalerMock]
        )
