# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os

from psipy.core.io import azure
from psipy.core.io.azure import _mock_block_blob_service as mock_service
from psipy.core.io.azure import download_blob, upload_blob
from psipy.core.io.azure import upload_blob_from_file, upload_dir

"""Test azure i/o functionality.

Currently untested:

    - Upload retrying
    - Overwrite/no-overwrite uploads.

"""

azure._called_from_test = True


def test_upload_blob_from_file(temp_dir):
    tmpfile = os.path.join(temp_dir, "file1")
    with open(tmpfile, "w") as f:
        f.write("file1")
    mock_service.reset()
    upload_blob_from_file("container", "blob_name", tmpfile)
    calls = [(c, b) for c, b, d in mock_service.calls["upload_blob"]]
    assert calls == [("container", "blob_name")]
    upload_blob_from_file("container", "blob/a", tmpfile)
    calls = [(c, b) for c, b, d in mock_service.calls["upload_blob"]]
    assert calls == [("container", "blob_name"), ("container", "blob/a")]


def test_upload_blob():
    mock_service.reset()
    upload_blob("container", "blob_name", b"123")
    assert mock_service.calls["upload_blob"] == [("container", "blob_name", b"123")]


def test_upload_dir(temp_dir):
    with open(os.path.join(temp_dir, "file1"), "w") as f:
        f.write("file1")
    with open(os.path.join(temp_dir, ".file2"), "w") as f:
        f.write(".file2")
    os.makedirs(os.path.join(temp_dir, "a"))
    with open(os.path.join(temp_dir, "a", "file3"), "w") as f:
        f.write("file3")
    with open(os.path.join(temp_dir, "a", ".file4"), "w") as f:
        f.write(".file4")

    mock_service.reset()
    upload_dir("container", "blob", temp_dir, ignore_hidden=False)
    assert len(mock_service.calls["get_blob_properties"]) == 4
    assert len(mock_service.calls["upload_blob"]) == 4
    expected = sorted(
        [
            ("container", "blob/file1"),
            ("container", "blob/.file2"),
            ("container", "blob/a/file3"),
            ("container", "blob/a/.file4"),
        ]
    )
    calls = sorted([(c, b) for c, b, d in mock_service.calls["upload_blob"]])
    assert expected == calls

    mock_service.reset()
    upload_dir("container", "blob", temp_dir, ignore_hidden=True)
    assert len(mock_service.calls["get_blob_properties"]) == 2
    assert len(mock_service.calls["upload_blob"]) == 2
    expected = sorted([("container", "blob/file1"), ("container", "blob/a/file3")])
    calls = sorted([(c, b) for c, b, d in mock_service.calls["upload_blob"]])
    assert expected == calls


def test_download_blob(temp_dir):
    os.makedirs(os.path.join(temp_dir, "sub"))
    path = os.path.join(temp_dir, "sub", "somefile.txt")
    download_blob("container", "blobname", path)
    assert os.path.isfile(path)
    with open(path, "r") as fobj:
        assert fobj.read() == "container-blobname-content"
