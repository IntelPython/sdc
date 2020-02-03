.. _pandas.Series.memory_usage:

:orphan:

pandas.Series.memory_usage
**************************

Return the memory usage of the Series.

The memory usage can optionally include the contribution of
the index and of elements of `object` dtype.

:param index:
    bool, default True
        Specifies whether to include the memory usage of the Series index.

:param deep:
    bool, default False
        If True, introspect the data deeply by interrogating
        `object` dtypes for system-level memory consumption, and include
        it in the returned value.

:return: int
    Bytes of memory consumed.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

