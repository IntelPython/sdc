.. _pandas.Series.nsmallest:

:orphan:

pandas.Series.nsmallest
***********************

Return the smallest `n` elements.

:param n:
    int, default 5
        Return this many ascending sorted values.

:param keep:
    {'first', 'last', 'all'}, default 'first'
        When there are duplicate values that cannot all fit in a
        Series of `n` elements:

:param - ``first``:
    return the first `n` occurrences in order
        of appearance.

:param - ``last``:
    return the last `n` occurrences in reverse
        order of appearance.

:param - ``all``:
    keep all occurrences. This can result in a Series of
        size larger than `n`.

:return: Series
    The `n` smallest values in the Series, sorted in increasing order.

Limitations
-----------
- Parameter 'keep' except 'first' is currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_nsmallest.py
   :language: python
   :lines: 27-
   :caption: Return the smallest n elements.
   :name: ex_series_nsmallest

.. command-output:: python ./series/series_nsmallest.py
   :cwd: ../../../examples

.. seealso::

    :ref:`Series.nlargest <pandas.Series.nlargest>`
        Get the n largest elements.

    :ref:`Series.sort_values <pandas.Series.sort_values>`
        Sort Series by values.

    :ref:`Series.head <pandas.Series.head>`
        Return the first n rows.

