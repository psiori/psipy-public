# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Global pytest fixtures.

Read more on the official `pytest documentation <https://docs.pytest.org/en/latest/
fixture.html#using-fixtures-from-classes-modules-or-projects>`_.

.. autosummary::

    psipy.conftest.temp_dir
    psipy.conftest.tensorflow

Usage with test classes, especially useful for fixtures where you do not care
about the return value but more about scope management:

.. code-block:: python
    :linenos:

    @pytest.mark.usefixtures("tensorflow")
    class TestMyClass:
        # All methods in this class make use of the "tfsession" fixture.

        @staticmethod
        def test_mymethod():
            # This method runs inside a fresh tensorflow graph and session.
            assert len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) == 0

.. code-block:: python
    :linenos:

    @pytest.mark.usefixtures("tensorflow")
    def test_mymethod():
        # This method runs inside a fresh tensorflow graph and session.
        assert len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) == 0

"""

import shutil

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def temp_dir(tmpdir):
    """Create a temporary directory that is automatically deleted on test end."""
    path = str(tmpdir.mkdir("test-tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def tensorflow():
    """Clears the tensorflow session after the fact.

    It is currently unclear whether using this is actually required
    with tensorflow 2 -- need to investigate how tensorflow 2 graph
    garbage collection / global graph / scoping works.
    """
    np.random.seed(10)
    tf.random.set_seed(10)
    yield None
    tf.keras.backend.clear_session()


@pytest.fixture
def tfgraph():
    """Provides a test specific tensorflow graph.

    Deprecated in favor of :meth:`~psipy.conftest.tensorflow`.
    """
    with tf.Graph().as_default() as graph:
        yield graph


@pytest.fixture
def tfsession(tfgraph):
    """Provides a test specific tensorflow session.

    Deprecated in favor of :meth:`~psipy.conftest.tensorflow`.
    """
    with tf.compat.v1.Session(graph=tfgraph) as session:
        yield session
