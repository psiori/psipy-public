# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os.path
from zipfile import ZipFile

import pytest

from psipy.core.io.zip import add_to_zip, zip_file, zip_file_mp


def test_zip_file(temp_dir: str):
    filepath = os.path.join(temp_dir, "file.log")
    zippath = os.path.join(temp_dir, "file.zip")
    with open(filepath, "w") as fobj:
        fobj.write("{'some': 'log'}\n")
    zip_file(filepath)
    with ZipFile(zippath) as zf:
        with zf.open("file.log", "r") as zfobj:
            # Need to strip newlines for windows.
            assert zfobj.read().decode().strip() == "{'some': 'log'}"


def test_add_to_zip(temp_dir: str):
    contents = os.path.join(temp_dir, "contents")
    os.makedirs(contents)
    with open(os.path.join(contents, "file.log"), "w") as fobj:
        fobj.write("{'some': 'log'}\n")
    with open(os.path.join(contents, "file2.log"), "w") as fobj:
        fobj.write("{'some': 'log2'}\n")
    contents2 = os.path.join(contents, "subdir")
    os.makedirs(contents2)
    with open(os.path.join(contents2, "file.log"), "w") as fobj:
        fobj.write("{'some': 'log'}\n")
    with open(os.path.join(contents2, "file2.log"), "w") as fobj:
        fobj.write("{'some': 'log2'}\n")

    zippath = os.path.join(temp_dir, "file.zip")
    with ZipFile(zippath, "w") as zf:
        add_to_zip(zf, contents)
    expected = ["file.log", "file2.log", "subdir/file.log", "subdir/file2.log"]
    with ZipFile(zippath, "r") as zf:
        assert sorted(zf.namelist()) == expected


@pytest.mark.slow
def test_zip_file_mp(temp_dir: str):
    filepath = os.path.join(temp_dir, "file.log")
    zippath = os.path.join(temp_dir, "file.zip")
    with open(filepath, "w") as fobj:
        fobj.write("{'some': 'log'}\n")
    p = zip_file_mp(filepath)
    p.join()
    with ZipFile(zippath) as zf:
        with zf.open("file.log") as zfobj:
            # Need to strip newlines for windows.
            assert zfobj.read().decode().strip() == "{'some': 'log'}"
