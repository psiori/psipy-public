# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Saveable mixin for sklearn.

See :mod:`psipy.core.io.saveable`.

.. autosummary::

    SaveableSklearnMixin

"""

from typing import Any, Dict

from sklearn.base import BaseEstimator

from psipy.core.io.memory_zip_file import MemoryZipFile
from psipy.core.io.saveable import Saveable

__all__ = ["SaveableSklearnMixin"]


class SaveableSklearnMixin(BaseEstimator, Saveable):
    """Saveable mixin for Sklearn  :class:`~sklearn.base.BaseEstimator` models.

    Inherit from the Sklearn class and this mixin. The resulting class
    should be tested, as this mixin does not yet account for sub-models and
    more advanced model attributes than core python and numpy types.

    Example usage on Sklearn's :class:`~sklearn.preprocessing.StandardScaler`::

        class SaveableStandardScaler(StandardScaler, Saveable):
            ...
        import tempfile
        filepath = os.path.join(tempfile.mkdtemp(), "scaler.zip")
        SaveableStandardScaler().fit(np.random.uniform(size=(100, 2)).save(filepath)
        scaler = SaveableStandardScaler.load(filepath)

    """

    def get_config(self) -> Dict[str, Any]:
        """Returns config needed for recreating the object self.

        Simply given the returned dictionary it is possible to fully recreate
        the current instance of the class using :meth:`from_config`. The
        recreated instance is not necessarily fitted/trained.
        """
        return self.get_params()

    def _save(self, zipfile: MemoryZipFile) -> MemoryZipFile:
        zipfile.add("config.json", self.get_config())
        zipfile.add_mixed_dict("state", self.__getstate__())
        return zipfile

    @classmethod
    def _load(cls, zipfile: MemoryZipFile):
        obj = cls.from_config(zipfile.get("config.json"))
        obj.__setstate__(zipfile.get_mixed_dict("state"))
        return obj
