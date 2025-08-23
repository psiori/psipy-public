.. _installation-label:

Installation
============

Make sure you are using Python 3.12. Any version below is not supported, versions
above are kept in mind but not actively tested.

.. code:: bash

    python3 --version

Clone the repository:

.. code:: bash

    git clone git@github.com:psiori/psipy.git

Install the package into your environment:

.. code:: bash

    uv pip install -e "./psipy[dev]"

By using ``-e``, the install is "editable". This results in any changes you
apply to the psipy code directly being reflected in the installed package.
