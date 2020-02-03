.. _pandas.DataFrame.describe:

:orphan:

pandas.DataFrame.describe
*************************

Generate descriptive statistics that summarize the central tendency,
dispersion and shape of a dataset's distribution, excluding
``NaN`` values.

Analyzes both numeric and object series, as well
as ``DataFrame`` column sets of mixed data types. The output
will vary depending on what is provided. Refer to the notes
below for more detail.

:param percentiles:
    list-like of numbers, optional
        The percentiles to include in the output. All should
        fall between 0 and 1. The default is
        ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.

:param include:
    'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored
        for ``Series``. Here are the options:

:param - 'all':
    All columns of the input will be included in the output.

:param - A list-like of dtypes:
    Limits the results to the
        provided data types.
        To limit the result to numeric types submit
        ``numpy.number``. To limit it instead to object columns submit
        the ``numpy.object`` data type. Strings
        can also be used in the style of
        ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
        select pandas categorical columns, use ``'category'``

:param - None (default):
    The result will include all numeric columns.

:param exclude:
    list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored
        for ``Series``. Here are the options:

:param - A list-like of dtypes:
    Excludes the provided data types
        from the result. To exclude numeric types submit
        ``numpy.number``. To exclude object columns submit the data
        type ``numpy.object``. Strings can also be used in the style of
        ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
        exclude pandas categorical columns, use ``'category'``

:param - None (default):
    The result will exclude nothing.

:return: Series or DataFrame
    Summary statistics of the Series or Dataframe provided.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

