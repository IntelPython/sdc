.. _pandas.DataFrame.memory_usage:

:orphan:

pandas.DataFrame.memory_usage
*****************************

Return the memory usage of each column in bytes.

The memory usage can optionally include the contribution of
the index and elements of `object` dtype.

This value is displayed in `DataFrame.info` by default. This can be
suppressed by setting ``pandas.options.display.memory_usage`` to False.

:param index:
    bool, default True
        Specifies whether to include the memory usage of the DataFrame's
        index in returned Series. If ``index=True``, the memory usage of
        the index is the first item in the output.

:param deep:
    bool, default False
        If True, introspect the data deeply by interrogating
        `object` dtypes for system-level memory consumption, and include
        it in the returned values.

:return: Series
    A Series whose index is the original column names and whose values
    is the memory usage of each column in bytes.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

