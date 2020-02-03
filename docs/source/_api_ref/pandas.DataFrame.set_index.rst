.. _pandas.DataFrame.set_index:

:orphan:

pandas.DataFrame.set_index
**************************

Set the DataFrame index using existing columns.

Set the DataFrame index (row labels) using one or more existing
columns or arrays (of the correct length). The index can replace the
existing index or expand on it.

:param keys:
    label or array-like or list of labels/arrays
        This parameter can be either a single column key, a single array of
        the same length as the calling DataFrame, or a list containing an
        arbitrary combination of column keys and arrays. Here, "array"
        encompasses :class:`Series`, :class:`Index`, ``np.ndarray``, and
        instances of :class:`~collections.abc.Iterator`.

:param drop:
    bool, default True
        Delete columns to be used as the new index.

:param append:
    bool, default False
        Whether to append columns to existing index.

:param inplace:
    bool, default False
        Modify the DataFrame in place (do not create a new object).

:param verify_integrity:
    bool, default False
        Check the new index for duplicates. Otherwise defer the check until
        necessary. Setting to False will improve the performance of this
        method.

:return: DataFrame
    Changed row labels.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

