#!/usr/bin/env python3

# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os
from typing import Dict, Tuple

from setuptools import find_packages, setup


VERSION = None
path = os.path.abspath(os.path.dirname(__file__))
contents = {}  # type: Dict[str, Tuple[int, int, int]]
if not VERSION:
    with open(os.path.join(path, "psipy", "__version__.py")) as f:
        exec(f.read(), contents)
    VERSION = contents["__version__"]


# Parse requirements files.
with open("requirements.txt") as f:
    install_requires = [m for m in f.read().splitlines() if m and not m.startswith("#")]
with open("requirements-dev.txt") as f:
    dev_requires = [m for m in f.read().splitlines() if m and not m.startswith("#")]


setup(
    name="psipy",
    version=VERSION,
    description="Public part of PSIORI's internal ML-library 'psipy'",
    author="PSIORI Developers",
    author_email="info@psiori.com",
    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=["setuptools>=41.0.0"],
    tests_require=dev_requires,
    extras_require={
        "dev": dev_requires,
        "gym": ["gymnasium[classic-control]~=0.29.1"],
        "win": ["windows-curses~=2.1.0"],
    },
    entry_points=dict(console_scripts=["tui = psipy.rl.io.terminal_interface:main"]),
)
