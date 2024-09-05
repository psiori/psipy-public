# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""I/O methods to upload files to azure.

Access to azure storage requires the ``AZURE_STORAGE_ACCOUNT`` and
``AZURE_STORAGE_ACCESS_KEY`` environment variables to be set.

Also provides a slim command line interface, for details see :meth:`cli` or run::

    python -m psipy.core.io.azure --help


.. autosummary::
    cli
    get_blob_client
    get_blob_service
    to_connection_string
    upload_blob
    upload_blob_from_file
    upload_dir

"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Union

from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient, generate_blob_sas
from tqdm import tqdm

from psipy.core.utils import add_bool_arg

__all__ = [
    "get_blob_service",
    "download_blob",
    "upload_blob",
    "upload_blob_from_file",
    "upload_dir",
]


LOG = logging.getLogger(__name__)


## Start of test definitions.
# During test time, :attr:`_called_from_test` will be ``True``, resulting in
# the methods implemented far below to use the ``_Mock*`` classes instead of
# the actual ``azure.*`` classes.

#: Whether the azure io methods were called from tests / during test time.
_called_from_test = False


class _MockBlob:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.content = content

    def readall(self):
        return self.content


class _MockContainerClient:
    def __init__(self, container: str, calls: Dict[str, Any]):
        self.container = container
        self.calls = calls


class _MockBlobClient:
    def __init__(self, container: str, blob: str, calls: Dict[str, Any]):
        self.container = container
        self.blob = blob
        self.url = f"http://{blob}"
        self.calls = calls
        self.exists = False

    def download_blob(self) -> _MockBlob:
        return _MockBlob(self.blob, f"{self.container}-{self.blob}-content".encode())

    def upload_blob(self, data: bytes, **kwargs) -> None:
        print("upload_blob", self.container, self.blob)
        self.calls["upload_blob"].append((self.container, self.blob, data))

    def get_blob_properties(self):
        self.calls["get_blob_properties"].append((self.container, self.blob))
        if self.exists:
            return {}
        raise AzureError("Blob does not exist.")


class _MockBlobServiceClient:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = defaultdict(list)

    def get_blob_client(self, container: str, blob: str) -> _MockBlobClient:
        return _MockBlobClient(container, blob, self.calls)

    def get_container_client(self, container: str) -> _MockContainerClient:
        return _MockContainerClient(container, self.calls)


_mock_block_blob_service = _MockBlobServiceClient()

## End of test definitions.


def to_connection_string(name: str, key: str) -> str:
    return (
        f"DefaultEndpointsProtocol=https;AccountName={name};"
        f"AccountKey={key};EndpointSuffix=core.windows.net"
    )


def get_blob_service() -> Union[_MockBlobServiceClient, BlobServiceClient]:
    """Get an azure blob service instance."""
    if _called_from_test:
        return _mock_block_blob_service
    azure_storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
    azure_storage_access_key = os.getenv("AZURE_STORAGE_ACCESS_KEY")
    # https://docs.microsoft.com/en-us/rest/api/storageservices/previous-azure-storage-service-versions
    # https://github.com/Azure/azure-sdk-for-python/blob/azure-storage-blob_12.3.2/sdk/
    #   storage/azure-storage-blob/azure/storage/blob/_serialize.py
    azure_storage_api_version = os.getenv("AZURE_STORAGE_API_VERSION", "2019-07-07")

    if not azure_storage_account or not azure_storage_access_key:
        raise ValueError(
            "Environment variables 'AZURE_STORAGE_ACCOUNT' and/or "
            "'AZURE_STORAGE_ACCESS_KEY' are not properly set."
        )
    conn = to_connection_string(azure_storage_account, azure_storage_access_key)
    return BlobServiceClient.from_connection_string(
        conn, api_version=azure_storage_api_version
    )


def get_blob_client(container: str, blobpath: str):
    service = get_blob_service()
    return service.get_blob_client(container, blobpath)


def get_container_client(container: str):
    service = get_blob_service()
    return service.get_container_client(container)


def get_blob_sas_url(container: str, blobpath: str) -> str:
    azure_storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
    azure_storage_access_key = os.getenv("AZURE_STORAGE_ACCESS_KEY")
    url = get_blob_client(container, blobpath).url
    expiry = datetime.utcnow() + timedelta(days=365)
    sas = generate_blob_sas(
        azure_storage_account,
        container,
        blobpath,
        account_key=azure_storage_access_key,
        start=datetime.utcnow() - timedelta(days=1),
        expiry=expiry,
        permission="r",
    )
    return f"{url}?{sas}"


def list_blobs(container: str, prefix: str = "", suffix: str = "") -> List[str]:
    client = get_container_client(container)
    return [
        blob_info.name
        for blob_info in client.list_blobs(name_starts_with=prefix)
        if blob_info.name.endswith(suffix)
    ]


def update_progress_bar(bar: tqdm, current: int, total: int):
    """Updates a given tqdm progressbar using current and total value.

    Helper function for having using tqdm progress bars in azure progress
    callbacks.

    Args:
        bar: :class:`tqdm` instance to update.
        current: number of bytes done uploading.
        total: Total number of bytes to upload.
    """
    if bar.total != total:
        bar.reset(total)
    bar.update(current)


def download_blob(
    container: str, blobpath: str, filepath: Optional[str] = None, tmpfile: bool = False
) -> str:
    """Download a blob to a given local targetpath.

    Args:
        container: Container within ``AZURE_STORAGE_ACCOUNT`` to download from.
        blobpath: Path within ``container`` to download.
        targetpath: Local filesystem path to download file to.
    """
    # To retrieve all blobs under a given path into the container, the following
    # can be used: `service.list_blobs(container, prefix=blobpath)`
    if tmpfile and filepath is not None:
        raise ValueError("'tmpfile' may not be True when filepath provided.")
    if filepath is None:
        if tmpfile:
            name, ext = os.path.splitext(os.path.basename(blobpath))
            if ext in [".gz", ".bz2"]:
                name, ext2 = os.path.splitext(name)
                ext = f"{ext2}{ext}"
            filepath = NamedTemporaryFile(prefix=name, suffix=ext, delete=False).name
        else:
            filepath = os.path.basename(blobpath.replace("/", os.sep))
    client = get_blob_client(container, blobpath)
    with open(filepath, "wb") as fobj:
        fobj.write(client.download_blob().readall())
    return filepath


def read_blob(container: str, blobpath: str) -> bytes:
    """Download from azure blob to memory.

    Args:
        container_name: Azure blob container.
        blob_name: Azure blob path/name.
    """
    blob = get_blob_client(container, blobpath)
    return blob.download_blob().readall()


def blob_exists(container: str, blobpath: str) -> bool:
    """Check if a blob exists.

    There is no native ``exists`` check in the azure blob storage sdk
    apparently: https://github.com/Azure/azure-sdk-for-python/issues/9507
    """
    client = get_blob_client(container, blobpath)
    try:
        client.get_blob_properties()
    except AzureError:
        return False
    return True


def upload_blob(
    container: str,
    blobpath: str,
    data: bytes,
    timeout: int = 60 * 5,
    overwrite: bool = False,
):
    """Upload bytes to blob storage.

    Args:
        container: Target container within ``AZURE_STORAGE_ACCOUNT`` to upload to.
        blobpath: Target path within ``container`` to upload to.
        data: Bytes to upload.
        overwrite: Whether to overwrite existing blobs.
    """
    client = get_blob_client(container, blobpath)
    client.upload_blob(
        data,
        length=len(data),
        connection_timeout=timeout,
        overwrite=overwrite,
    )


def upload_blob_from_file(
    container: str,
    blobpath: str,
    filepath: str,
    retries: int = 0,
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    """Upload file to blob storage.

    Args:
        container: Target container within ``AZURE_STORAGE_ACCOUNT`` to upload to.
        blobpath: Target path within ``container`` to upload to.
        filepath: Local filesystem path to file to upload.
        retries: Number of times to retry failed uploads.
        overwrite: Whether to overwrite existing blobs.
    """
    blobpath = blobpath.strip("/")
    LOG.info(f"Uploading {filepath}...")

    if not overwrite and blob_exists(container, blobpath):
        LOG.info(f"Blob {blobpath} exists, skipping upload.")
        return None

    client = get_blob_client(container, blobpath)
    for retry in range(retries + 1):
        try:
            with open(filepath, "rb") as data:
                client.upload_blob(data, overwrite=overwrite)
        except AzureError as err:
            if retry == retries:
                raise err
            LOG.warning(f"Ran into {type(err).__name__}, retrying...")
        else:
            return client.url
    return None


def upload_dir(
    container: str,
    blobpath: str,
    dirpath: str,
    ignore_hidden: bool = True,
    retries: int = 0,
    overwrite: bool = False,
    ignore_errors: bool = False,
) -> None:
    """Recursively uploads directory to blob storage.

    Method will upload all subfiles in a directory, also hidden files.

    Args:
        container: Target container within ``AZURE_STORAGE_ACCOUNT`` to upload to.
        blobpath: Target path within ``container`` to upload to.
        dirpath: Local filesystem path to directory to upload.
        ignore_hidden: Whether to ignore local hidden files (start with ``.``).
        retries: Number of times to retry failed uploads.
        overwrite: Whether to overwrite existing blobs.
        ignore_errors: Whether to ignore errors raised when uploading individual files.
    """
    LOG.info(f"Uploading {dirpath}..")
    blobpath = blobpath.strip("/")
    for path, _, files in os.walk(dirpath):
        for name in files:
            filepath = os.path.join(path, name)
            if name.startswith(".") and ignore_hidden:
                LOG.info(f"Hidden file {filepath} skipped.")
                continue
            targetblob = blobpath
            targetpath = filepath[len(dirpath) :]  # strip source dirpath
            targetpath = targetpath.replace(os.sep, "/").strip("/")
            if targetpath:
                targetblob = f"{blobpath}/{targetpath}"
            try:
                upload_blob_from_file(
                    container,
                    targetblob,
                    filepath,
                    retries=retries,
                    overwrite=overwrite,
                )
            except AzureError as err:
                if not ignore_errors:
                    raise err
                LOG.warning(f"Could not upload {filepath} due to {type(err).__name__}.")


def cli():
    """PSIPY Azure CLI. Currently single purpose: Upload files to a blob storage.

    Usage::

        python -m psipy.core.io.azure --help

    """
    import os.path
    import sys
    from argparse import ArgumentParser

    from dotenv import load_dotenv

    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(logging.StreamHandler(sys.stdout))

    parser = ArgumentParser(
        description=(
            "PSIPY Azure CLI. Currently single purpose. \n\n"
            "Upload files to an azure blob storage, making use of the "
            "'AZURE_STORAGE_ACCOUNT' and 'AZURE_STORAGE_ACCESS_KEY' "
            "environment variables. All path components from source files "
            "will be retained!"
        )
    )
    parser.add_argument("target", type=str, help="Target container name and path.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Paths to upload. Can be both individual files and directories.",
    )
    parser.add_argument(
        "--retries", type=int, help="Number of retries if an upload fails."
    )
    add_bool_arg(
        parser,
        "full-path",
        default=False,
        help="Whether to retain the full local path at the target location.",
    )
    add_bool_arg(
        parser,
        "ignore-hidden",
        default=True,
        help="Whether to ignore hidden files (starting with a period.)",
    )
    add_bool_arg(
        parser, "overwrite", default=False, help="Whether to overwrite existing blobs."
    )
    add_bool_arg(
        parser,
        "ignore-errors",
        default=False,
        help="Whether to ignore errors when uploading files.",
    )
    args = parser.parse_args()

    if not (
        (os.getenv("AZURE_STORAGE_ACCOUNT") or os.getenv("AZURE_STORAGE_ACCESS_KEY"))
        and os.path.exists(".env")
    ):
        load_dotenv(dotenv_path=".env")
    else:
        raise ValueError("`AZURE_STORAGE_ACCOUNT` `AZURE_STORAGE_ACCESS_KEY` not set.")

    # Split target into containername and blobpath.
    container, *blobpathparts = args.target.split("/")
    blobpath = "/".join(blobpathparts)

    for path in args.paths:
        targetpath = blobpath
        if os.path.isdir(path):
            if os.sep in path and args.full_path:
                # If the full-path should be used for uploading, append the full
                # directory path to the targetpath. The paths further down into
                # that directory will be attached by the upload dir method.
                targetpath = f"{targetpath}/{path.replace(os.sep, '/')}"
            upload_dir(
                container,
                targetpath,
                path,
                ignore_hidden=args.ignore_hidden,
                retries=args.retries,
                overwrite=args.overwrite,
                progress_bar=True,
                ignore_errors=args.ignore_errors,
            )
        else:
            if os.sep in path and args.full_path:
                # Full-path should be used in the target container and we
                # actually received an actual filepath, not just a filename,
                # create a new targetpath of both.
                targetpath = f"{blobpath}/{path.replace(os.sep, '/')}"
            else:
                # Full path should not be used, upload flat to target blobpath.
                targetpath = f"{blobpath}/{os.path.basename(path)}"
            upload_blob_from_file(
                container,
                targetpath,
                path,
                retries=args.retries,
                overwrite=args.overwrite,
                progress_bar=True,
            )
    LOG.info("Upload done.")


if __name__ == "__main__":
    cli()
