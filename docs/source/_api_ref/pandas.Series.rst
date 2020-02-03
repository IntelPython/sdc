.. _pandas.Series:

:orphan:

pandas.Series
*************

One-dimensional ndarray with axis labels (including time series).

Labels need not be unique but must be a hashable type. The object
supports both integer- and label-based indexing and provides a host of
methods for performing operations involving the index. Statistical
methods from ndarray have been overridden to automatically exclude
missing data (currently represented as NaN).

Operations between Series (+, -, /, \*, \*\*) align values based on their
associated index values-- they need not be the same length. The result
index will be the sorted union of the two indexes.

:param data:
    array-like, Iterable, dict, or scalar value
        Contains data stored in Series.

        .. versionchanged :: 0.23.0

        and later.

:param index:
    array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If both a dict and index
        sequence are used, the index will override the keys found in the
        dict.

:param dtype:
    str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.

:param copy:
    bool, default False
        Copy input data.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

