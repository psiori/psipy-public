# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Utility functions for running python code in jupyter notebooks.

.. autosummary::

    is_notebook

"""

__all__ = ["is_notebook"]


def is_notebook() -> bool:
    """Check whether code is executed in a jupyter notebook."""
    try:
        ipython = get_ipython()  # type: ignore
    except NameError:
        return False
    return str(type(ipython)) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"
