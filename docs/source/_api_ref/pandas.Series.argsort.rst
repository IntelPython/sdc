.. _pandas.Series.argsort:

:orphan:

pandas.Series.argsort
*********************

Override ndarray.argsort. Argsorts the value, omitting NA/null values,
and places the result in the same locations as the non-NA values.

:param axis:
    int
        Has no effect but is accepted for compatibility with numpy.

:param kind:
    {'mergesort', 'quicksort', 'heapsort'}, default 'quicksort'
        Choice of sorting algorithm. See np.sort for more
        information. 'mergesort' is the only stable algorithm

:param order:
    None
        Has no effect but is accepted for compatibility with numpy.

:return: Series
    Positions of values within the sort order with -1 indicating
    nan values.

Examples
--------
.. literalinclude:: ../../../examples/series/series_argsort.py
   :language: python
   :lines: 27-
   :caption: Override ndarray.argsort.
   :name: ex_series_argsort

.. command-output:: python ./series/series_argsort.py
   :cwd: ../../../examples

.. note::
    Parameters axis, kind, order are currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    `numpy.absolute
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.argsort.html#numpy.ndarray.argsort>`_
        Return indices of the minimum values along the given axis.

