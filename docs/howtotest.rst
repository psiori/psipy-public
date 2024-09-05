How to test
=================

Running tests
--------------------------------------

During development, it is recommended to install :mod:`psipy` into a python virtual environment including all necessary as well as optional pip requirements:

.. code-block:: bash

    $ python3 -m venv .venv
    $ pip install -e ".[dev,automl,gym]"

When running the ``psipy`` tests, one has multiple options which can be combined creatively.

Running all tests:: bash

.. code-block:: bash

    $ pytest

Running just a specific submodules's tests:: bash

.. code-block:: bash

    $ pytest -k psipy/rl

Excluding a specific submodule's tests:: bash

.. code-block:: bash

    $ pytest -k "not psipy/dataroom"

Excluding tests marked as slow by the developer:: bash

.. code-block:: bash

    $ pytest -m "not slow"


Writing tests
--------------------------------------

Grouping tests
````````````````````````````````

When a lot of tests are colocated in a single ``test_*.py`` file, it might make sense to either split them into multiple files or alternatively group them into ``Test*`` classes:

.. code-block:: python
    :linenos:

    class MyClass:
        def method(self):
            self.attr = 1

    class TestMyClass:
        @staticmethod
        def test_method():
            a = MyClass().method()
            assert a.attr == 1



Global pytest fixtures
````````````````````````````````

See :mod:`psipy.conftest` for more details...

.. autosummary::

    psipy.conftest.tensorflow


pytest markers
````````````````````````````````

Markers can be used to group test methods and select them when running ``pytest``.
Methods are marked using a decorator:

.. code-block:: python

    @pytest.mark.MARKER

For example:

.. code-block:: python
    :linenos:

    @pytest.mark.slow
    def test_mymethod():
        import time
        time.sleep(10)

Current markers used throughout :mod:`psipy`:

- ``slow``: Marks tests which need more than ~2 seconds to execute.
