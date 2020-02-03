.. _pandas.Series.nlargest:

:orphan:

pandas.Series.nlargest
**********************

Return the largest `n` elements.

:param n:
    int, default 5
        Return this many descending sorted values.

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
    The `n` largest values in the Series, sorted in decreasing order.

Limitations
-----------
- Parameter 'keep' except 'first' is currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_nlargest.py
   :language: python
   :lines: 27-
   :caption: Return the largest n elements.
   :name: ex_series_nlargest

.. command-output:: python ./series/series_nlargest.py
   :cwd: ../../../examples

.. seealso::

    :ref:`Series.nsmallest <pandas.Series.nsmallest>`
        Get the n smallest elements.

    :ref:`Series.sort_values <pandas.Series.sort_values>`
        Sort Series by values.

    :ref:`Series.head <pandas.Series.head>`
        Return the first n rows.

