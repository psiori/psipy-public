# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Provides functionality to zip files.

Strongly inspired by the cpython sources.
https://github.com/python/cpython/blob/624cc10/Lib/zipfile.py#L2377

.. autosummary::

    add_to_zip
    zip_file
    zip_file_mp
    zip_files

"""

import multiprocessing as mp
import os
from zipfile import ZIP_DEFLATED, ZipFile


def path_join(*parts: str) -> str:
    """Like :meth:`os.path.join` but always using forward slashes.

    Note that this method treats backward slashes as directory separators,
    independent of the host system (unix vs windows). This is useful for
    unifying file paths in zip files which should be portable to other file
    systems, but might produce unexpected behavior as **both forward (``/``) and
    backward (``\\``) slashes result in forward slashes in this method's
    output**.

    Python is generally very happy with using forward slashes for accessing the
    filesystem also on windows. Not using forward slashes (and favoring
    :meth:`os.path.join`) on windows therefore only becomes relevant when
    paths are to be used by other programs.

    Example::

        >>> # Don't worry about the double escaping below, its doctest specific.
        >>> path_join("\\\\1/2/3\\\\12", "456")
        '1/2/3/12/456'
        >>> path_join("abc")
        'abc'
        >>> path_join("abc", "")
        'abc/'
        >>> path_join("")
        ''
        >>> path_join("123", "/456")
        '/456'

    Args:
        *parts: Individual parts to join to a single path.
    """
    parts = [subpart for part in parts for subpart in part.split("\\")]
    return "/".join(os.path.join(*parts).split(os.sep))


def add_to_zip(zf: ZipFile, sourcepath: str, zippath: str = "") -> None:
    """Write a file or directory to the ZipFile object.

    Args:
        zf: The zip to add the file to.
        sourcepath: The path to the file or directory to add to the zip.
        zippath: Path inside the zip to put the file.
    """
    if os.path.isfile(sourcepath):
        with open(sourcepath, "rb") as fp:
            # Use forwardslash-only path for pointing into the zipfile.
            zf.writestr(path_join(zippath), fp.read())
    elif os.path.isdir(sourcepath):
        for name in sorted(os.listdir(sourcepath)):
            add_to_zip(zf, os.path.join(sourcepath, name), os.path.join(zippath, name))
    else:
        raise ValueError(f"sourcepath {sourcepath} neither file nor directory.")


def zip_files(
    targetpath: str,
    *sourcepaths: str,
    delete_originals: bool = False,
) -> None:
    """Zip many files into a zip of a given name.

    Args:
        targetpath: Path to put the new zipfile.
        sourcepaths: Paths to files to add to the zip. No directories allowed!
        delete_originals: Whether to delete the original files after zipping.
    """
    with ZipFile(targetpath, "w", ZIP_DEFLATED) as zf:
        for sourcepath in sourcepaths:
            zippath = os.path.basename(sourcepath)
            if not zippath:
                zippath = os.path.basename(os.path.dirname(sourcepath))
            if zippath in ("", os.curdir, os.pardir):
                zippath = ""
            add_to_zip(zf, sourcepath, zippath)
    if delete_originals:
        for sourcepath in sourcepaths:
            os.remove(sourcepath)


def zip_file(sourcepath: str, delete_original: bool = False) -> str:
    """Zip a single file.

    Uses :meth:`zip_files`.

    Args:
        sourcepath: File to zip.
        delete_original: Whether to delete the original file after it has been zipped.
    """
    sourcebase, _ext = os.path.splitext(sourcepath)
    targetpath = f"{sourcebase}.zip"
    zip_files(targetpath, sourcepath, delete_originals=delete_original)
    return targetpath


def zip_file_mp(
    sourcepath: str,
    delete_original: bool = False,
) -> mp.process.BaseProcess:
    """Zip a single file in the background.

    Args:
        sourcepath: File to zip.
        delete_original: Whether to delete the original file after it has been zipped.
    """
    ctx: mp.context.BaseContext = mp.get_context("spawn")
    process = ctx.Process(
        target=zip_file,
        kwargs=dict(
            sourcepath=sourcepath,
            delete_original=delete_original,
        ),
    )
    process.start()
    return process
