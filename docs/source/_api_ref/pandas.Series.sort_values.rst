.. _pandas.Series.sort_values:

:orphan:

pandas.Series.sort_values
*************************

Sort by the values.

Sort a Series in ascending or descending order by some
criterion.

:param axis:
    {0 or 'index'}, default 0
        Axis to direct sorting. The value 'index' is accepted for
        compatibility with DataFrame.sort_values.

:param ascending:
    bool, default True
        If True, sort values in ascending order, otherwise descending.

:param inplace:
    bool, default False
        If True, perform operation in-place.

:param kind:
    {'quicksort', 'mergesort' or 'heapsort'}, default 'quicksort'
        Choice of sorting algorithm. See also :func:`numpy.sort` for more
        information. 'mergesort' is the only stable  algorithm.

:param na_position:
    {'first' or 'last'}, default 'last'
        Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
        the end.

:return: Series
    Series ordered by values.

Examples
--------
.. literalinclude:: ../../../examples/series/series_sort_values.py
   :language: python
   :lines: 27-
   :caption: Sort by the values.
   :name: ex_series_sort_values

.. command-output:: python ./series/series_sort_values.py
   :cwd: ../../../examples

.. note::
    Parameters axis, kind, na_position are currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.sort_index <pandas.Series.sort_index>`
        Sort by the Series indices.

    :ref:`DataFrame.sort_values <pandas.DataFrame.sort_values>`
        Sort DataFrame by the values along either axis.

    :ref:`DataFrame.sort_index <pandas.DataFrame.sort_index>`
        Sort DataFrame by indices.

