.. _pandas.Series.head:

:orphan:

pandas.Series.head
******************

Return the first `n` rows.

This function returns the first `n` rows for the object based
on position. It is useful for quickly testing if your object
has the right type of data in it.

:param n:
    int, default 5
        Number of rows to select.

:return: obj_head : same type as caller
    The first `n` rows of the caller object.

Examples
--------
.. literalinclude:: ../../../examples/series/series_head.py
   :language: python
   :lines: 27-
   :caption: Getting the first n rows.
   :name: ex_series_head

.. command-output:: python ./series/series_head.py
   :cwd: ../../../examples

.. seealso::

    :ref:`DataFrame.tail <pandas.DataFrame.tail>`

