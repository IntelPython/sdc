.. _pandas.Series.isin:

:orphan:

pandas.Series.isin
******************

Check whether `values` are contained in Series.

Return a boolean Series showing whether each element in the Series
matches an element in the passed sequence of `values` exactly.

:param values:
    set or list-like
        The sequence of values to test. Passing in a single string will
        raise a ``TypeError``. Instead, turn a single string into a
        list of one element.

        .. versionadded:: 0.18.1

        Support for values as a set.

:return: Series
    Series of booleans indicating if each element is in values.

:raises:
    TypeError
        - If `values` is a string

Examples
--------
.. literalinclude:: ../../../examples/series/series_isin.py
   :language: python
   :lines: 27-
   :caption: Check whether values are contained in Series.
   :name: ex_series_isin

.. command-output:: python ./series/series_isin.py
   :cwd: ../../../examples

.. seealso::

    :ref:`DataFrame.isin <pandas.DataFrame.isin>`
        Equivalent method on DataFrame.

