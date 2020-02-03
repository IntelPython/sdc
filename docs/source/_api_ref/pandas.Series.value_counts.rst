.. _pandas.Series.value_counts:

:orphan:

pandas.Series.value_counts
**************************

Return a Series containing counts of unique values.

The resulting object will be in descending order so that the
first element is the most frequently-occurring element.
Excludes NA values by default.

:param normalize:
    boolean, default False
        If True then the object returned will contain the relative
        frequencies of the unique values.

:param sort:
    boolean, default True
        Sort by frequencies.

:param ascending:
    boolean, default False
        Sort in ascending order.

:param bins:
    integer, optional
        Rather than count values, group them into half-open bins,
        a convenience for ``pd.cut``, only works with numeric data.

:param dropna:
    boolean, default True
        Don't include counts of NaN.

:return: Series

Examples
--------
.. literalinclude:: ../../../examples/series/series_value_counts.py
   :language: python
   :lines: 27-
   :caption: Getting the number of values excluding NaNs
   :name: ex_series_value_counts

.. command-output:: python ./series/series_value_counts.py
   :cwd: ../../../examples

.. note::
    Parameter bins and dropna for Strings are currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.count <pandas.Series.count>`

