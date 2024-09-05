ARG BASE_IMAGE=psipy.azurecr.io/psipy:latest-base
FROM ${BASE_IMAGE}

COPY requirements.txt /wd/psipy/
COPY requirements-dev.txt /wd/psipy/
RUN pip3 install --no-cache-dir -r /wd/psipy/requirements.txt \
                                -r /wd/psipy/requirements-dev.txt

COPY . /wd/psipy
RUN pip3 install --no-cache-dir /wd/psipy[dev,gym,automl] \
 && rm -rf /wd/psipy
