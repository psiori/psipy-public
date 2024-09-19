# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Upload images to Azure blob store ready for labelling by QM.

Usage:

    ./tools/img2qm.py \
        input_images/*.ppm \
        qm-200925-truckunload-trolley-headblock \
        main_segmentation.csv \
        --jpg \
        --prefix main \
        --account autocranedevdata \
        --access_key SECRET_ACCESS_KEY
"""


import os
from glob import glob
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from psipy.core.io.azure import get_blob_sas_url, upload_blob, upload_blob_from_file


def main(
    pattern: str,
    container: str,
    csv: str,
    prefix: Optional[str] = None,
    account: Optional[str] = None,
    access_key: Optional[str] = None,
    jpg: bool = False,
    quality: int = 100,
    overwrite: bool = False,
):
    """Upload images to azure, producing a csv containing individual SAS urls.

    Can compress images on the fly when uploading as jpg.

    Args:
        pattern: Source image file glob pattern. E.g. ``input_images/*.ppm``
        container: Target azure blob container.
        csv: Target csv file to write the SAS urls to.
        prefix: Prefix to use within the container ('sub directory').
        account: Storage account name, can also be provided through env.
        access_key: Storage account secret access key, can also be provided
                    through env.
        jpg: Whether to convert files to jpg.
        quality: When converting to jpg, quality to use.
        overwrite: Whether to overwrite existing blobs.
    """
    load_dotenv()
    if account is not None:
        os.environ["AZURE_STORAGE_ACCOUNT"] = account
    if access_key is not None:
        os.environ["AZURE_STORAGE_ACCESS_KEY"] = access_key
    with open(csv, "w") as fp:
        for filepath in tqdm(glob(pattern, recursive=True)):
            name, ext = os.path.splitext(os.path.basename(filepath))
            blobname = f"{name}.jpg"
            if prefix:
                blobname = f"{prefix}/{blobname}"
            if jpg:
                img = Image.open(filepath)
                with BytesIO() as imbuf:
                    img.save(imbuf, format="JPEG", quality=quality)
                    upload_blob(
                        container,
                        blobname,
                        imbuf.getvalue(),
                        overwrite=overwrite,
                    )
            else:
                upload_blob_from_file(
                    container,
                    blobname,
                    filepath,
                    overwrite=overwrite,
                )
            url = get_blob_sas_url(container, blobname)
            fp.write(f"{url}{os.linesep}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
