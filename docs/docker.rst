.. _docker-label:

Docker
=================

psipy includes docker images usable as base images specifically for development.
While they theoretically also could be used for deployment, they currently are
rather bloated in order to support a wide range of python development needs.

Use pre-built images
--------------------

The quickest way to get started is by using the following commands. This will
login to the azure container registry, pull the latest psipy cpu image and
start a jupyter notebook server on ``localhost:9000``::

    az acr login -n psipy
    docker run -it -p 9000:8888 -v `pwd`:/wd psipy.azurecr.io/psipy:latest

The ``psipy:latest`` image is updated automatically from the ``develop`` branch.
The same holds for tagged images (``psipy:1.0.0``) and *base* images
(``psipy:latest-base``). *base* images contain everything psipy builds upon,
besides the python requirements and psipy itself. The following images are
provided from the ``psipy.azurecr.io`` registry:

- ``psipy:latest-base``
- ``psipy:latest-gpu-base``
- ``psipy:latest``
- ``psipy:latest-gpu``
- ``psipy:1.0.0-base``
- ``psipy:1.0.0-gpu-base``
- ``psipy:1.0.0``
- [...]

cnvrg
`````

The preferred way to run machine learning tasks in PSIORI is to make use of
`cnvrg <https://app.psiori.cnvrg.io/PSIORI>`_. Currently, the latest psipy cpu
and gpu images are available there and, as the images on the ACR described
above, always up to date with the current ``develop`` state of psipy.


Build your own
--------------

If you want to build a docker image from your local repository's state, take a
look at the ``make docker/build`` command. By pulling images from the
psipy ACR first, the docker build cache is pre-populated and the builds sped
up. Passing ``GPU=1`` as in the following example builds the gpu images,
omitting the ``GPU=1`` would build the non-gpu images::

    az acr login -n psipy
    make docker/pull GPU=1
    make docker/build GPU=1
