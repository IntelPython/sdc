.. _pandas.Series.fillna:

:orphan:

pandas.Series.fillna
********************

Fill NA/NaN values using the specified method.

:param value:
    scalar, dict, Series, or DataFrame
        Value to use to fill holes (e.g. 0), alternately a
        dict/Series/DataFrame of values specifying which value to use for
        each index (for a Series) or column (for a DataFrame).  Values not
        in the dict/Series/DataFrame will not be filled. This value cannot
        be a list.

:param method:
    {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed Series
        pad / ffill: propagate last valid observation forward to next valid
        backfill / bfill: use next valid observation to fill gap.

:param axis:
    {0 or 'index'}
        Axis along which to fill missing values.

:param inplace:
    bool, default False
        If True, fill in-place. Note: this will modify any
        other views on this object (e.g., a no-copy slice for a column in a
        DataFrame).

:param limit:
    int, default None
        If method is specified, this is the maximum number of consecutive
        NaN values to forward/backward fill. In other words, if there is
        a gap with more than this number of consecutive NaNs, it will only
        be partially filled. If method is not specified, this is the
        maximum number of entries along the entire axis where NaNs will be
        filled. Must be greater than 0 if not None.

:param downcast:
    dict, default is None
        A dict of item->dtype of what to downcast if possible,
        or the string 'infer' which will try to downcast to an appropriate
        equal type (e.g. float64 to int64 if possible).

:return: Series
    Object with missing values filled.

Limitations
-----------
- Parameters method, limit, downcast are currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_fillna.py
   :language: python
   :lines: 27-
   :caption: Fill NA/NaN values using the specified method.
   :name: ex_series_fillna

.. command-output:: python ./series/series_fillna.py
   :cwd: ../../../examples

.. seealso::

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html#pandas.Series.interpolate>`_
        Fill NaN values using interpolation.

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.reindex.html#pandas.Series.reindex>`_
        Conform object to new index.

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.asfreq.html#pandas.Series.asfreq>`_
        Convert TimeSeries to specified frequency.

