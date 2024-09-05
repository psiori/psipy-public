How to doc
=================


Sphinx logic in creating documentation
--------------------------------------


What is the source for documentation generation?
````````````````````````````````````````````````

- Every folder which contains an ``__init__.py`` in the root will be handled as a "package" by sphinx, which means that
  the ``*.py`` files inside are listed as "modules" and can be seen and clicked on in the resulting documentation.

- Subfolders that also contain an ``__init__.py`` are handled as "subpackages".

- Every ``__init__.py`` provides an opportunity to write some general text about this (sub)package.


Creating docstrings
```````````````````

Sphinx searches for text like

  .. code-block:: python
     :linenos:

     """docstring"""

  These are printed in the same location where they are found: At (sub)package level, beginning of submodule or under
  the path/ name of a class or function

- Some properties inside a docstring for a class like ``Example``, ``Args``, ``Attributes`` and others are collected an
  printed according to the sphinx logic. At the end of this document is a code example, where you can see how sphinx
  deals with those properties.

- A docstring can be modified by the same means as this document can be. For more information see the references below.

.. note:: Document classes and their constructor on the class level, not inside
   the init function, as it won't be displayed in the resulting docs.


Docstring content
-----------------


(Sub)package level
``````````````````

- Every package, subpackage and module needs some introductory docstring inside its ``__init__.py`` to tell about the
  functionality and usage of this module.

  .. note:: A reader of the psipy documentation might still be evaluating, whether they actually want to use a package
     and is probably unfamiliar with the reasoning behind it, especially concerning application, math and overall logic.
     Upper level docstrings should therefore be written both informative and from a (new) user's perspective.

- The docstring inside the package level ``__init__.py`` is more than any other docstring of the same package supposed to
  get the readers attention and to want them to use this module.

- Unlike other docstrings deeper inside the modules (see below) the introductory docstring should be used to give a
  more or less broad explanation of what the package does:

  + The explanation should concentrate on the overall logic of the package and can contain some short(!) overview of the
    underlying principles.

  + In doubt, explanation can to be chosen over shortness as long as the whole text is still of a reasonable length,
    a clear structure and well readable.


Module level
````````````

- Under every class or function there is supposed to be a docstring, which allows to fully understand what is going on
  in this very method:

  + The information inside the docstrings should be as concise as possible and can assume at least a basic understanding
    of the logic and namings of the machine learning field. Conciseness should be chosen over explanation, as long as
    the logic itself still can be derived from the docstrings.

  + It is not allowed to put information about one class inside the docstring of another class and
    reference to it. Since code can change we should avoid having to deal with changing references.


General structure and prerequisites
-----------------------------------

- It is a definite prerequisite, that the grammar and spelling of the docstrings are correct.

- The docstrings have to be well readable.

- To prevent duplicate docstrings the information to be read by a developer or data scientist to
  fully understand a method is the following:

  1. Package docstring

     2. Subpackage docstring (if existant)

        3. Module docstring

           4. Class docstring

              5. Function docstring

  .. note:: A function docstring can be very short, as long as the information given in the levels above is providing the
     necessary background.

- Packages which have to appear at the landing page (the one with the grid like appearance with links and short
  descriptions) should be placed there by their code owners and described with a short text about the package

Referencing
```````````

- Referencing inside document to different pages: :ref:`docker-label` and :ref:`installation-label`.
- Inside the same module classes and functions can be referenced as shown in :py:mod:`docs.example_google` below.
- Even third party packages can be referenced: :class:`np.int64 <numpy.dtype>`.
  If you reference to third party packages don't forget to add the package in
  `conf.py <https://github.com/psiori/psipy/blob/develop/docs/conf.py>`_:

  .. code-block:: python

   intersphinx_mapping = {
       ...
       "numpy": ("https://docs.scipy.org/doc/numpy/", None),
   }


Naming
``````

- Every package, module, class or function has to be named in a descriptive way, for example: ``psipy.core.utils.py``.
  This way it is obvious at first glance, that this module belongs to the core of psipy and probably contains some
  utility classes and functions.
- Be sure that every part of a module path is meaningful. Don't use words like "internal"  "model" or words from
  products like "dataroom" or "databrowser"


Other
`````

- The copyright note must be placed on top of the modules, but outside the docstrings for classes or function:

  .. code-block:: python

     # Copyright (C) PSIORI GmbH, Germany
     # Proprietary and confidential, all rights reserved.

     """ Docstring of module ... """


Sources
-------

| A quick reference of reStructuredText
| http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html

| The sphinx autodoc module, which pulls in documentation from module docstrings
| https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

| Napoleon, the sphinx extension which reads google code style
| https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

| A step through guide for using sphinx with Napoleon
| https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
|


Example docstrings
------------------

To explore the possibilities of reStructuredText and the conventions of Google Code Style see the below example. The
code itself can be found at `docs/example_google.py`

----------------------------------------------------

.. automodule:: docs.example_google
