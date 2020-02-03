.. _pandas.DataFrame.to_parquet:

:orphan:

pandas.DataFrame.to_parquet
***************************

Write a DataFrame to the binary parquet format.

.. versionadded:: 0.21.0

This function writes the dataframe as a `parquet file
<https://parquet.apache.org/>`_. You can choose different parquet
backends, and have the option of compression. See
:ref:`the user guide <io.parquet>` for more details.

:param fname:
    str
        File path or Root Directory path. Will be used as Root Directory
        path while writing a partitioned dataset.

        .. versionchanged:: 0.24.0

:param engine:
    {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

:param compression:
    {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.

:param index:
    bool, default None
        If ``True``, include the dataframe's index(es) in the file output.
        If ``False``, they will not be written to the file. If ``None``,
        the behavior depends on the chosen engine.

        .. versionadded:: 0.24.0

:param partition_cols:
    list, optional, default None
        Column names by which to partition the dataset
        Columns are partitioned in the order they are given

        .. versionadded:: 0.24.0

        \*\*kwargs
        Additional arguments passed to the parquet library. See
        :ref:`pandas io <io.parquet>` for more details.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

