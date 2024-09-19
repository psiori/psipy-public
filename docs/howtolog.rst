How to log
=======================

Convention for writing log messages:

.. code-block:: python
    :linenos:

    LOG.debug("message without")
    LOG.info("notice..", extra=dict(some="json"))
    LOG.warning("watch out!", extra={"more": 1.0}))
    LOG.error("an error occured", extra={"alist": [1, 2, 3]})

- Message (first argument) is required!
- The ``extra`` dictionary is optional.
- By default only the message will be logged.

Projects using psipy can configure their loggers to extend this functionality:

- Logfiles of json strings containing not only the message but also the ``extra`` and further metadata.
- By default still only messages are logged to stdout.
- One can enable printing the full json onto stdout, similar to what appears in the logfile.

Writing json logfiles has the advantage of easily searching logfiles or importing them into database for more extensive analysis.

An example for configuring the logger in such a way can be found in the `AndritzRL project <https://github.com/psiori/AndritzRL/blob/8c582ea/andrl/config.py#L186-L224>`_.

Currently there is no specification for what kind of information is to be contained in the ``extra`` fields. Individual projects might want to establish such standards in order to make log files easily searchable.


`jq <https://stedolan.github.io/jq/>`_
--------------

    jq is like sed for JSON data - you can use it to slice and filter and map and transform structured data with the same ease that sed, awk, grep and friends let you play with text.

The best way to get started with jq is probably any cheat sheet from google. The following just gives some examples specific to parsing json logs as created by projects following the guide above.

- Pretty print: ``jq . LOGFILE.log``
- ERRORs:  ``jq '. | select(.levelname == "ERROR")' LOGFILE.log``
- Specific timespan: ``jq '. | select((.asctime > "2020-05-11 13:59") and (.asctime < "2020-05-11 14"))' LOGFILE.log``


Custom fields
^^^^^^^^^^^^^^^^^^^^^

Consistent usage of custom fields through the ``extra`` dictionary (see above) allows one to quickly select parts of the logfile relevant to the given project.
For instance, when using :mod:`psipy.rl` it is recommended to enhance all log entries by the ``cycle`` number. The <AndritzRL package `https://github.com/psiori/AndritzRL>`_ for example does do so.

This way, we can easily select a specific region not based on world time, but based on interaction cycles:

.. code-block:: bash

    jq '. | select((.cycle < 250) and (.cycle < 300))' LOGFILE.log

Similarly, one can only retrieve log entries which have a given ``extra`` entry, specifically interesting if not all messages have that entry:


.. code-block:: bash

    jq '. | select(has("cluster"))' LOGFILE.log
