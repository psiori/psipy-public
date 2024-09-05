Distribution
=================

When distributing python code, sending around a zip of python sources might not
always be the best choice. Instead, having a single shared object similar to
when sharing a C++ library might be great.

psipy provides this functionality through the ``./tools/pypack`` script.
``pypack`` uses ``nuitka`` to compile the library into a single ``psipy.so``
(or ``psipy.pyd`` on windows) file, which can be used the same way as having
the ``./psipy`` package directory available locally.

These distributable files can be build manually (see below), but are also
automatically provided for the latest ``develop`` state of the project through
the CI/CD pipeline. The artifacts can be found by viewing any of the builds
on `azure pipelines <https://dev.azure.com/psiori/psipy/_build?definitionId=6&_a=summary&repositoryFilter=6&branchFilter=405>`_.


Usage
--------------

.. code-block:: bash
  :linenos:

  Convenience wrapper for nuitka to create distributable compiled libraries.
  Developed to package psipy, can be applied to other python packages.

  ./pypack module [--package] [--output] [--exclude] [--no-tests] [--cleanup]
                  [--verbose] [--debug] [--help]

  Example Usage:
  --------------
  ./tools/pypack psipy -p psipy.ts.search --no-tests --exclude psipy.ts.model

  Arguments:
  ----------
  module
    Path to module. The final part of the path (basename) will also be the
    created libraries name.

  -p | --package
    One may provide this command multiple times to define subpackages of
    "module" to include. Given those subpackages, their imports will be recursed
    into and included as well. If no package is provided explicitly, the whole
    "module" package will be included. Default: Same as "module" positional arg.

  -o | --output
    Output directory, absolute or relative to directory from which pypack is
    called from.
    Default: dist

  --no-tests
    Whether to exclude tests, specifically subpackages named '*.tests'.
    Default: False

  --cleanup
    Whether to cleanup build files when done. Default: False

  --debug
    Default: False

  -h | --help
    This information.
